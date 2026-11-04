"""
RdiiModel — the central class of swmmrdii.

An ``RdiiModel`` wraps one or more sparsehydro ``EnsembleModel`` instances
(one per subcatchment) and holds all the data needed to calibrate and run the
rainfall-derived inflow and infiltration (RDII) model:

    - Subcatchment metadata (name, area, BFI, rain gage)
    - Forcing data (per-gage PRISM PPT and optional TMEAN DataFrames)
    - Observed flow time series (for calibration scoring)
    - Calibrated sparsehydro parameter bounds (set via RTKInitializer)

Two construction paths are provided:

    1. ``RdiiModel(area_acres=..., ...)``  — single subcatchment, no SWMM file needed.
    2. ``RdiiModel.from_swmm(inp_path=..., ...)``  — multi-subcatchment, reads
       subcatchment list from a SWMM .inp file (and optionally a PCSWMM .db file).

The model is lazy: sparsehydro templates are built on first call to
``_build_templates()``, which is triggered by ``CalOptimizer`` before
optimization starts.
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import numpy as np
import pandas as pd

from sparsehydro.rdii import make_rtk_ensemble
from sparsehydro.rdii.seasonality import compute_time_features

from .forcing import (
    load_dat,
    load_tsf,
    resolve_conversion,
    resolve_temp_unit,
    resolve_ppt_unit,
    parse_swmm_timeseries,
)
from .rtk_initializer import RTKInitializer
from .swmm_parse import load_swmm_subcatchments


class RdiiModel:
    """
    Unified RDII model container for single or multi-subcatchment watersheds.

    Parameters
    ----------
    area_acres : float
        Catchment area in acres.  For multi-subcatchment models constructed
        via ``from_swmm()``, this is the *total* watershed area and is
        computed automatically from the subcatchment table.
    n_models : int
        Number of IA pathway models in the sparsehydro EnsembleModel (almost
        always 1; the multi-model option exists for lumped RDII pathways).
    n_triangles : int
        Number of RTK triangles per IA pathway (typically 3: fast, medium,
        slow).
    units : str
        ``'imperial'`` (default, CFS / inches) or ``'metric'`` (m³/s / mm).
    ia_defaults : dict or None
        Keyword arguments forwarded to each ``IAModel`` constructor, e.g.
        ``{'snow': True}`` to enable the degree-day snow module.
    name : str
        Human-readable label for the model (e.g. gage ID or watershed name).
        Used in ``summary()`` output and JSON artifacts.
    """

    def __init__(
        self,
        area_acres: float,
        n_models: int = 1,
        n_triangles: int = 3,
        units: str = "imperial",
        ia_defaults: Optional[dict] = None,
        name: str = "",
    ) -> None:
        self.name = name
        self.n_models = n_models
        self.n_triangles = n_triangles
        self.units = units
        self.ia_defaults = ia_defaults or {}

        # Build a one-row subcatchment DataFrame so that single-SC and multi-SC
        # models share the same internal data model throughout the rest of the
        # library.
        self._sc_df = pd.DataFrame(
            [
                {
                    "name": name or "watershed",
                    "area_acres": float(area_acres),
                    "outlet": "",
                    "rain_gage": "default",
                    "ppt_file": "",
                    "temp_file": "",
                    "bfi": 1.0,
                    "area_weight": 1.0,
                    "bfi_weight": 1.0,
                }
            ]
        )

        self._is_multi: bool = False   # True once from_swmm() adds >1 row

        # Legacy combined forcing cache: {key: DataFrame with rainfall_in
        # (+ temperature_c)}.  Still populated by the manual load_forcing()
        # path for single-SC models and used as a fallback in _get_sc_df.
        self._forcing_cache: dict[str, pd.DataFrame] = {}

        # Series-identity caches.  Each unique precipitation / temperature time
        # series is loaded exactly once and stored under its own ID (a file
        # path or an embedded [TIMESERIES] name).  Subcatchments hold pointers
        # (_ppt_id / _temp_id columns in _sc_df) into these caches, so a series
        # shared by many subcatchments is parsed only once.
        self._ppt_cache:  dict[str, pd.DataFrame] = {}   # id -> rainfall_in frame
        self._temp_cache: dict[str, pd.DataFrame] = {}   # id -> temperature_c frame

        # Observed flow Series (indexed by datetime) — set by load_observed().
        self._observed: Optional[pd.Series] = None

        # sparsehydro EnsembleModel templates, one per subcatchment.
        # Built lazily by _build_templates().
        self._templates: list = []
        self._cal_params: list[str] = []       # calibratable parameter names
        self._param_meta: list[tuple] = []     # (name, lb, ub, use_log)
        self._ia_max_ub: float = 2.0
        self._uh1_t_ub: Optional[float] = None
        self._templates_built: bool = False

    # -----------------------------------------------------------------------
    # Alternate constructor: multi-subcatchment from SWMM INP + optional DB
    # -----------------------------------------------------------------------

    @classmethod
    def from_swmm(
        cls,
        inp_path: str,
        db_path: Optional[str] = None,
        name_field: str = "Name",
        area_field: str = "Area",
        bfi_field: Optional[str] = "BFI",
        where: Optional[str] = None,
        temp_dir: Optional[str] = None,
        temp_suffix: str = "_tmean",
        temp_field: Optional[str] = None,
        temp_unit: Optional[str] = None,
        n_models: int = 1,
        n_triangles: int = 3,
        units: str = "imperial",
        ia_defaults: Optional[dict] = None,
        name: str = "",
    ) -> "RdiiModel":
        """
        Construct an RdiiModel by reading subcatchment data from a SWMM .inp file.

        Parameters
        ----------
        inp_path : str
            Path to the SWMM .inp model file.
        db_path : str or None
            Path to the PCSWMM SQLite .db file containing user-defined
            subcatchment attributes (e.g. BFI).  When omitted (None),
            *bfi_field* and *temp_field* must also be None — a specified
            field with no DB to resolve it against raises a ``ValueError``
            rather than silently substituting defaults.  With
            ``bfi_field=None``, BFI defaults to 1.0 for all subcatchments
            (no BFI-based R_3 scaling).
        name_field : str
            Output column label for the subcatchment name.
        area_field : str
            Output column label for the area (default ``'Area'``).
        bfi_field : str or None
            Name of the BFI attribute in the PCSWMM DB.  Pass ``None`` to
            disable BFI loading entirely.
        where : str or None
            Optional Pandas ``.query()`` expression to filter which
            subcatchments are included, e.g. ``"area_acres > 50"``,
            ``"name in ['SC1','SC2']"``, or — referencing any DB user-field —
            ``"caltag == '03146000'"``.  When *db_path* is given, every
            ``Subcatchments_User_Fields`` column is attached automatically
            (lower-cased: ``CalTag`` -> ``caltag``, ``%Imperv`` ->
            ``pct_imperv``) so the query can use it without extra setup.
        temp_dir : str or None
            Directory containing temperature .dat files.  When provided, the
            loader derives temperature file paths from the PPT filename by
            appending *temp_suffix* to the basename.
        temp_suffix : str
            Suffix appended to the PPT basename when deriving temperature file
            paths (default ``'_tmean'``).
        temp_field : str or None
            Name of a PCSWMM DB attribute (in ``Subcatchments_User_Fields``)
            that, per subcatchment, names a SWMM ``[TIMESERIES]`` holding
            temperature data.  When set, temperature is read directly from the
            INP's embedded series (or its ``FILE`` reference) — no *temp_dir*
            needed.  Requires *db_path*.
        temp_unit : str or None
            Unit the temperature data is in (e.g. ``'degF'``, ``'degC'``,
            ``'K'``).  Used to convert to °C when forcing is loaded.  Required
            with *temp_field* unless the data is already in °C.
        n_models, n_triangles, units, ia_defaults, name :
            Forwarded to ``RdiiModel.__init__()`` — see class docstring.

        Returns
        -------
        RdiiModel
        """
        sc_df = load_swmm_subcatchments(
            inp_path=inp_path,
            db_path=db_path,
            name_field=name_field,
            area_field=area_field,
            bfi_field=bfi_field,
            temp_dir=temp_dir,
            temp_suffix=temp_suffix,
            temp_field=temp_field,
            temp_unit=temp_unit,
            where=where,
        )

        total_area = sc_df["area_acres"].sum()
        m = cls(
            area_acres=total_area,
            n_models=n_models,
            n_triangles=n_triangles,
            units=units,
            ia_defaults=ia_defaults,
            name=name,
        )
        # Replace the one-row placeholder with the real subcatchment table.
        m._sc_df = sc_df
        m._is_multi = len(sc_df) > 1
        return m

    # -----------------------------------------------------------------------
    # Alternate constructor: from a list of subcatchment records
    # -----------------------------------------------------------------------

    @classmethod
    def from_records(
        cls,
        records: "list[dict]",
        n_models: int = 1,
        n_triangles: int = 3,
        units: str = "imperial",
        ia_defaults: Optional[dict] = None,
        name: str = "",
    ) -> "RdiiModel":
        """
        Construct an RdiiModel from an explicit list of subcatchment records.

        This is the most flexible constructor — it bypasses SWMM/CSV parsing
        and takes the subcatchment table directly.  ``from_csv`` and
        ``from_list`` are thin wrappers that build *records* and delegate here.

        Each record is a dict describing one subcatchment.  Recognised keys:

        ===================  ========================================  ========
        Key                  Meaning                                   Required
        ===================  ========================================  ========
        ``name``             Subcatchment label                        yes
        ``area_acres``       Drainage area (acres)                     yes
        ``ppt_file``         Path to precipitation .dat file           yes
        ``ppt_unit``         Precip unit (``'in'``/``'mm'``/``'cm'``)  no (in)
        ``temp_file``        Path to temperature .dat file             no
        ``temp_unit``        Temp unit (``'degF'``/``'degC'``/``'K'``) no
        ``bfi``              Baseflow index for R₃ scaling             no (1.0)
        ``outlet``           Receiving node name (export only)         no
        ===================  ========================================  ========

        Precipitation and temperature are treated as external ``FILE`` sources;
        unit conversion to inches / °C is applied per row from *ppt_unit* /
        *temp_unit* when ``load_forcing_from_sc_df()`` is called.

        Parameters
        ----------
        records : list of dict
            One dict per subcatchment (see table above).
        n_models, n_triangles, units, ia_defaults, name :
            Forwarded to ``RdiiModel.__init__()``.

        Returns
        -------
        RdiiModel
        """
        if not records:
            raise ValueError("from_records() requires at least one subcatchment record.")

        rows = []
        for i, rec in enumerate(records):
            r = {str(k).strip().lower().replace(" ", "_"): v for k, v in rec.items()}
            # Accept a few friendly aliases for the area / path columns.
            area = r.get("area_acres", r.get("area"))
            ppt  = r.get("ppt_file", r.get("rainfall_path", r.get("rainfall_file", "")))
            temp = r.get("temp_file", r.get("temperature_path", r.get("temperature_file", "")))
            ppt_u  = r.get("ppt_unit", r.get("rainfall_units", r.get("rainfall_unit")))
            temp_u = r.get("temp_unit", r.get("temperature_units", r.get("temperature_unit")))
            nm = r.get("name", f"SC{i + 1}")

            if area is None:
                raise ValueError(f"Record {nm!r}: missing required 'area' / 'area_acres'.")
            if not ppt:
                raise ValueError(f"Record {nm!r}: missing required rainfall path.")

            # Normalise blank / NaN optional fields to empty / None.
            def _blank(v):
                if v is None:
                    return ""
                s = str(v).strip()
                return "" if s.lower() in ("", "nan", "none") else s

            temp_path = _blank(temp)
            rows.append(
                {
                    "name": str(nm),
                    "area_acres": float(area),
                    "outlet": _blank(r.get("outlet", "")),
                    "rain_gage": str(nm),       # one gage per SC (path-based)
                    "ppt_source": "FILE",
                    "ppt_file": str(ppt),
                    "ppt_series": "",
                    "ppt_unit": _blank(ppt_u),
                    "temp_source": "FILE" if temp_path else "",
                    "temp_file": temp_path,
                    "temp_series": "",
                    "temp_unit": _blank(temp_u),
                    "bfi": float(r.get("bfi", 1.0)) if _blank(r.get("bfi", "")) else 1.0,
                }
            )

        sc_df = pd.DataFrame(rows)

        # Derived weight columns (mirrors load_swmm_subcatchments()).
        total_area = sc_df["area_acres"].sum()
        sc_df["area_weight"] = sc_df["area_acres"] / total_area
        bfi_area = sc_df["area_acres"] * sc_df["bfi"]
        bfi_area_total = bfi_area.sum()
        sc_df["bfi_weight"] = (
            bfi_area / bfi_area_total if bfi_area_total > 0 else sc_df["area_weight"]
        )

        m = cls(
            area_acres=total_area,
            n_models=n_models,
            n_triangles=n_triangles,
            units=units,
            ia_defaults=ia_defaults,
            name=name,
        )
        m._sc_df = sc_df
        m._is_multi = len(sc_df) > 1
        return m

    # -----------------------------------------------------------------------
    # Alternate constructor: from a CSV table
    # -----------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        n_models: int = 1,
        n_triangles: int = 3,
        units: str = "imperial",
        ia_defaults: Optional[dict] = None,
        name: str = "",
    ) -> "RdiiModel":
        """
        Construct an RdiiModel from a CSV table of subcatchments.

        Expected column headers (case- and space-insensitive):

        ===========================  ==============================  ==========
        Column                       Meaning                         Required
        ===========================  ==============================  ==========
        ``Name``                     Subcatchment label              yes
        ``Area``                     Drainage area (acres)           yes
        ``Rainfall Path``            Path to precipitation .dat      yes
        ``Rainfall Units``           ``in`` / ``mm`` / ``cm``        no (in)
        ``Temperature Path``         Path to temperature .dat        no
        ``Temperature Units``        ``degF`` / ``degC`` / ``K``     no
        ``BFI``                      Baseflow index (R₃ scaling)     no (1.0)
        ``Outlet``                   Receiving node (export only)    no
        ===========================  ==============================  ==========

        Temperature columns are optional — omit them (or leave blank) for
        precipitation-only forcing.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        n_models, n_triangles, units, ia_defaults, name :
            Forwarded to ``RdiiModel.__init__()``.

        Returns
        -------
        RdiiModel
        """
        df = pd.read_csv(csv_path)
        records = df.to_dict(orient="records")
        return cls.from_records(
            records,
            n_models=n_models,
            n_triangles=n_triangles,
            units=units,
            ia_defaults=ia_defaults,
            name=name or os.path.splitext(os.path.basename(csv_path))[0],
        )

    # -----------------------------------------------------------------------
    # Alternate constructor: from parallel lists
    # -----------------------------------------------------------------------

    @classmethod
    def from_list(
        cls,
        names: "list[str]",
        areas: "list[float]",
        rainfall_paths: "list[str]",
        rainfall_units: "Optional[list]" = None,
        temperature_paths: "Optional[list]" = None,
        temperature_units: "Optional[list]" = None,
        bfi: "Optional[list]" = None,
        outlets: "Optional[list]" = None,
        n_models: int = 1,
        n_triangles: int = 3,
        units: str = "imperial",
        ia_defaults: Optional[dict] = None,
        name: str = "",
    ) -> "RdiiModel":
        """
        Construct an RdiiModel from parallel lists, one element per subcatchment.

        This is the maximally-explicit constructor: pass a list for each field.
        Only *names*, *areas*, and *rainfall_paths* are required; the rest are
        optional and default appropriately.  All provided lists must be the
        same length as *names*.

        Parameters
        ----------
        names : list of str
            Subcatchment labels.
        areas : list of float
            Drainage areas in acres.
        rainfall_paths : list of str
            Paths to per-subcatchment precipitation .dat files.
        rainfall_units : list or None
            Precip units (``'in'`` / ``'mm'`` / ``'cm'``).  Defaults to inches.
        temperature_paths : list or None
            Paths to per-subcatchment temperature .dat files.  ``None`` or an
            empty string entry means no temperature for that subcatchment.
        temperature_units : list or None
            Temperature units (``'degF'`` / ``'degC'`` / ``'K'``).
        bfi : list or None
            Per-subcatchment baseflow index for R₃ scaling (defaults to 1.0).
        outlets : list or None
            Receiving node names (used only by the SWMM export helpers).
        n_models, n_triangles, units, ia_defaults, name :
            Forwarded to ``RdiiModel.__init__()``.

        Returns
        -------
        RdiiModel
        """
        n = len(names)

        def _col(lst, default=None):
            if lst is None:
                return [default] * n
            if len(lst) != n:
                raise ValueError(
                    f"All from_list() columns must match len(names)={n}; "
                    f"got a list of length {len(lst)}."
                )
            return list(lst)

        areas_l = _col(areas)
        ppt_l   = _col(rainfall_paths)
        pptu_l  = _col(rainfall_units)
        temp_l  = _col(temperature_paths)
        tempu_l = _col(temperature_units)
        bfi_l   = _col(bfi, 1.0)
        out_l   = _col(outlets, "")

        records = [
            {
                "name": names[i],
                "area_acres": areas_l[i],
                "ppt_file": ppt_l[i],
                "ppt_unit": pptu_l[i],
                "temp_file": temp_l[i],
                "temp_unit": tempu_l[i],
                "bfi": bfi_l[i],
                "outlet": out_l[i],
            }
            for i in range(n)
        ]
        return cls.from_records(
            records,
            n_models=n_models,
            n_triangles=n_triangles,
            units=units,
            ia_defaults=ia_defaults,
            name=name,
        )

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    @property
    def area_acres(self) -> float:
        """Total watershed area in acres."""
        return float(self._sc_df["area_acres"].sum())

    def load_forcing(
        self,
        ppt_path: str,
        temp_path: Optional[str] = None,
        gage: Optional[str] = None,
        ppt_conversion=None,
        temp_conversion=None,
    ) -> None:
        """
        Load precipitation (and optional temperature) forcing and store it in
        the forcing cache under the given gage key.

        For single-subcatchment models, *gage* defaults to ``'default'`` and
        this method only needs to be called once.

        For multi-subcatchment models, call this method once per unique rain
        gage that appears in the subcatchment table.  The gage key should
        match the ``rain_gage`` column values in ``_sc_df`` (typically the
        gage name from the SWMM ``[RAINGAGES]`` section).  Alternatively,
        populate ``ppt_file`` / ``temp_file`` in ``_sc_df`` and call
        ``load_forcing_from_sc_df()`` as a convenience loader.

        Parameters
        ----------
        ppt_path : str
            Path to the precipitation .dat or .tsf file.
        temp_path : str or None
            Path to the temperature .dat or .tsf file.  If None, no
            temperature column is added; the IAModel falls back to its
            ``ia_T_ref`` parameter value throughout the run.
        gage : str or None
            Cache key for this forcing set.  Defaults to ``'default'`` for
            single-SC models; for multi-SC models use the gage name from
            the SWMM ``[RAINGAGES]`` section.
        ppt_conversion : str, callable, or None
            Optional conversion applied to the raw precipitation column before
            caching.  sparsehydro expects inches, so pass this when your .dat
            file is in another unit.

            Accepts a named preset string or a one-argument callable:

                - ``None``          — values already in inches, no change
                - ``'mm_to_in'``    — millimetres → inches
                - ``'cm_to_in'``    — centimetres → inches
                - callable          — e.g. ``lambda x: x / 25.4``

            See ``swmmrdii.forcing.CONVERSIONS`` for all built-in presets.
        temp_conversion : str, callable, or None
            Optional conversion applied to the raw temperature column before
            caching.  sparsehydro expects °C.

            Accepts a named preset string or a one-argument callable:

                - ``None``              — values already in °C, no change
                - ``'degF_to_degC'``    — PRISM TMEAN (°F) → °C
                - ``'kelvin_to_degC'``  — Kelvin → °C
                - callable              — e.g. ``lambda x: (x - 32) * 5/9``

            See ``swmmrdii.forcing.CONVERSIONS`` for all built-in presets.
        """
        key = gage or "default"
        ppt = load_dat(ppt_path, value_col="rainfall_in")
        temp = load_dat(temp_path, value_col="temperature_c") if temp_path else None
        self._assemble_forcing(
            key=key,
            ppt=ppt,
            temp=temp,
            ppt_conversion=ppt_conversion,
            temp_conversion=temp_conversion,
        )

    def _assemble_forcing(
        self,
        key: str,
        ppt: pd.DataFrame,
        temp: Optional[pd.DataFrame] = None,
        ppt_conversion=None,
        temp_conversion=None,
    ) -> None:
        """
        Apply unit conversions and cache a forcing set under *key* in the
        legacy combined ``_forcing_cache``.

        Used by the manual ``load_forcing()`` path (single-SC models and direct
        per-gage loading).  ``load_forcing_from_sc_df()`` does NOT use this — it
        caches each series independently in ``_ppt_cache`` / ``_temp_cache``.
        *ppt* must carry a ``rainfall_in`` column; *temp* (optional) a
        ``temperature_c`` column.  Both are datetime-indexed.
        """
        _ppt_conv = resolve_conversion(ppt_conversion)
        if _ppt_conv is not None:
            ppt = ppt.copy()
            ppt["rainfall_in"] = _ppt_conv(ppt["rainfall_in"])

        if temp is not None:
            _temp_conv = resolve_conversion(temp_conversion)
            if _temp_conv is not None:
                temp = temp.copy()
                temp["temperature_c"] = _temp_conv(temp["temperature_c"])
            forcing = (
                ppt[["rainfall_in"]]
                .join(temp[["temperature_c"]], how="inner")
                .dropna()
            )
        else:
            forcing = ppt[["rainfall_in"]].dropna()

        self._forcing_cache[key] = forcing

    def load_forcing_from_sc_df(
        self,
        ppt_conversion=None,
        temp_conversion=None,
    ) -> None:
        """
        Resolve each subcatchment's precipitation and temperature time series
        from the columns populated by ``from_swmm()`` and cache them by series
        identity.

        Each subcatchment's precipitation comes from its rain gage, which is
        either an external ``FILE`` (``ppt_file``) or an embedded SWMM
        ``[TIMESERIES]`` (``ppt_series``, read from the INP).  Temperature, if
        present, comes from a per-subcatchment ``[TIMESERIES]`` named in the
        PCSWMM DB (``temp_series``) or a derived ``FILE`` (``temp_file``).

        Each *unique* series is loaded exactly once into ``_ppt_cache`` /
        ``_temp_cache`` (keyed by its file path or series name), and the
        subcatchment table gains ``_ppt_id`` / ``_temp_id`` pointer columns.
        Subcatchments sharing a series therefore reference the same cached
        frame regardless of how rain and temperature line up — there is no
        gage-based grouping.

        Parameters
        ----------
        ppt_conversion : str, callable, or None
            Conversion for precipitation (sparsehydro expects inches).  Use
            ``'mm_to_in'`` for millimetre data.
        temp_conversion : str, callable, or None
            Explicit override for the temperature conversion.  When omitted,
            the conversion is derived from each subcatchment's ``temp_unit``
            column (set by ``from_swmm(temp_unit=...)``).
        """
        inp_path = self._sc_df.attrs.get("inp_path")

        def _ppt_identity(row):
            """Return (id, loader) for this row's precipitation series."""
            src = row.get("ppt_source", "")
            if src == "FILE" or row.get("ppt_file"):
                pid = str(row["ppt_file"])
                return pid, (lambda: load_dat(pid, value_col="rainfall_in"))
            if src == "TIMESERIES" and row.get("ppt_series"):
                if not inp_path:
                    raise ValueError(
                        "Embedded precipitation TIMESERIES requires the INP path; "
                        "build the model with from_swmm()."
                    )
                pid = str(row["ppt_series"])
                return pid, (
                    lambda: parse_swmm_timeseries(inp_path, pid, value_col="rainfall_in")
                )
            raise ValueError(
                f"Subcatchment {row['name']!r}: no precipitation source "
                "(ppt_file / ppt_series both empty)."
            )

        def _temp_identity(row):
            """Return (id, loader) for this row's temperature series, or (None, None)."""
            src = row.get("temp_source", "")
            if src == "FILE" or row.get("temp_file"):
                tp = row.get("temp_file") or row.get("tmean_file")
                if not tp:
                    return None, None
                tid = str(tp)
                return tid, (lambda: load_dat(tid, value_col="temperature_c"))
            if src == "TIMESERIES" and row.get("temp_series"):
                if not inp_path:
                    raise ValueError(
                        "Embedded temperature TIMESERIES requires the INP path; "
                        "build the model with from_swmm()."
                    )
                tid = str(row["temp_series"])
                return tid, (
                    lambda: parse_swmm_timeseries(inp_path, tid, value_col="temperature_c")
                )
            return None, None

        ppt_ids: list[str] = []
        temp_ids: list = []

        for _, row in self._sc_df.iterrows():
            # ---- Precipitation: load once per unique series id ----
            pid, ppt_loader = _ppt_identity(row)
            if pid not in self._ppt_cache:
                # Per-row conversion: explicit override, else derive from the
                # row's ppt_unit column (set by from_csv / from_list), else none.
                pc = (
                    ppt_conversion
                    if ppt_conversion is not None
                    else resolve_ppt_unit(row.get("ppt_unit") or None)
                )
                ppt = ppt_loader()
                _ppt_conv = resolve_conversion(pc)
                if _ppt_conv is not None:
                    ppt = ppt.copy()
                    ppt["rainfall_in"] = _ppt_conv(ppt["rainfall_in"])
                self._ppt_cache[pid] = ppt[["rainfall_in"]].dropna()
            ppt_ids.append(pid)

            # ---- Temperature: load once per unique series id ----
            tid, temp_loader = _temp_identity(row)
            if tid is not None and tid not in self._temp_cache:
                # Per-row conversion: explicit override, else derive from unit.
                tc = (
                    temp_conversion
                    if temp_conversion is not None
                    else resolve_temp_unit(row.get("temp_unit") or None)
                )
                temp = temp_loader()
                _tconv = resolve_conversion(tc)
                if _tconv is not None:
                    temp = temp.copy()
                    temp["temperature_c"] = _tconv(temp["temperature_c"])
                self._temp_cache[tid] = temp[["temperature_c"]].dropna()
            temp_ids.append(tid)

        # Write pointer columns so the simulator resolves forcing per series.
        self._sc_df["_ppt_id"] = ppt_ids
        self._sc_df["_temp_id"] = temp_ids

    def load_observed(self, tsf_path: str, gage_id: str) -> None:
        """
        Load observed flow from a multi-block PCSWMM/SWMM TSF file.

        Parameters
        ----------
        tsf_path : str
            Path to the .tsf file.
        gage_id : str
            Gage identifier string that appears in the TSF block header
            (e.g. ``'03145483'``).
        """
        flow_df = load_tsf(tsf_path, series_id=gage_id, value_col="flow_cfs")
        # Store as a Series indexed by datetime for easy slicing.
        self._observed = flow_df.iloc[:, 0] if "flow_cfs" not in flow_df.columns else flow_df["flow_cfs"]

    def set_observed(self, observed: pd.Series) -> None:
        """
        Set observed flow directly from a pandas Series indexed by datetime.

        Useful when the observed data come from a source other than a TSF file.
        """
        self._observed = observed

    # -----------------------------------------------------------------------
    # Template building (called lazily by CalOptimizer)
    # -----------------------------------------------------------------------

    def _build_templates(
        self,
        ia_max_ub: Optional[float] = None,
        uh1_t_ub: Optional[float] = None,
        ia_kt: float = 0.05,
        ia_t_ref: float = 20.0,
        ia_t_freeze: float = 0.0,
        calibrate_temperature_params: bool = False,
    ) -> None:
        """
        Instantiate and configure one sparsehydro EnsembleModel per subcatchment.

        This is an internal method called by ``CalOptimizer`` before
        optimization begins.  It:

        1. Resolves the IA_MAX upper bound automatically from the runoff ratio
           if not provided by the caller.
        2. Calls ``RTKInitializer`` to get area-scaled starting values and
           bounds for the R, T, K parameters.
        3. Sets all parameter bounds on a reference model, then deep-copies the
           reference for each subcatchment (substituting the per-subcatchment
           area).
        4. Extracts the list of calibratable parameters and their log-scale
           flags for use by the Optuna objective function.

        Parameters
        ----------
        ia_max_ub : float or None
            Upper bound for the IA_MAX parameter (inches for imperial models).
            If None, the value is derived automatically: 0.5 in for high-runoff
            (ratio > 0.7) catchments, or ``max(2.0, 1.5 × max_daily_rain)``
            otherwise.
        uh1_t_ub : float or None
            Optional upper bound override for the first (fastest) UH triangle's
            T parameter.  Useful to prevent the fast component from blurring
            into the medium one.
        ia_kt, ia_t_ref, ia_t_freeze : float
            Fixed values for temperature-sensitive IA parameters.  These are
            frozen by default (calibrate=False) to reduce dimensionality; set
            *calibrate_temperature_params=True* to allow them to vary.
        calibrate_temperature_params : bool
            If True, ``ia_kT``, ``ia_T_ref``, and ``ia_T_freeze`` are included
            in the Optuna search.  The notebooks use False for the single-SC
            model and True for the multi-SC model (where equifinality has been
            accepted and extra degrees of freedom help volume balance).
        """
        if not self._forcing_cache and not self._ppt_cache:
            raise RuntimeError(
                "No forcing data loaded. Call load_forcing() or "
                "load_forcing_from_sc_df() before building templates."
            )

        # ---- Resolve IA_MAX upper bound from runoff ratio ------------------
        # Collect all precipitation across the loaded series to compute an
        # aggregate runoff ratio.  Prefer the series-id cache; fall back to the
        # legacy combined cache.
        if self._ppt_cache:
            rain_frames = [df["rainfall_in"] for df in self._ppt_cache.values()]
        else:
            rain_frames = [df["rainfall_in"] for df in self._forcing_cache.values()]
        all_rain = pd.concat(rain_frames)
        mean_rain = float(all_rain.mean())

        if ia_max_ub is not None:
            _ia_max_ub = float(ia_max_ub)
        else:
            if self._observed is not None:
                mean_flow = float(self._observed.mean())
                # Convert mean daily rain (in/day) to CFS using the watershed area.
                _in_per_day_to_cfs = self.area_acres * 43560.0 / (12.0 * 86400.0)
                runoff_ratio = min(mean_flow / (mean_rain * _in_per_day_to_cfs), 0.90)
            else:
                # Without observed flow, default to the conservative auto rule.
                runoff_ratio = 0.5

            if runoff_ratio > 0.7:
                # High-runoff catchment: keep IA small so the model doesn't
                # absorb too much rainfall into initial abstraction.
                _ia_max_ub = 0.5
            else:
                # Standard: allow IA up to 1.5× the maximum daily rainfall.
                _ia_max_ub = max(2.0, float(all_rain.max()) * 1.5)

        self._ia_max_ub = _ia_max_ub
        self._uh1_t_ub = uh1_t_ub

        # ---- Build reference model and set parameter bounds ----------------
        ref_model = make_rtk_ensemble(
            n_models=self.n_models,
            n_triangles=self.n_triangles,
            units=self.units,
            area_acres=self.area_acres,
            ia_defaults=self.ia_defaults,
            output_name="rdii_cfs",
        )
        ref_model.validate()

        alias = ref_model.aliases[0]

        # RTKInitializer generates area-scaled initial values and bounds for all
        # R, T, K parameters across all triangles.
        rtk = RTKInitializer(
            n_triangles=self.n_triangles * self.n_models,
            area_acres=self.area_acres,
        )
        x0, bounds = rtk.flat_params()

        i = 0
        for j in range(1, self.n_triangles + 1):
            r_val, r_bounds = x0[i], bounds[i]; i += 1
            t_val, t_bounds = x0[i], bounds[i]; i += 1
            k_val, k_bounds = x0[i], bounds[i]; i += 1

            # Optionally cap the first UH triangle's T to keep the fast
            # component physically distinct from the medium one.
            if j == 1 and uh1_t_ub is not None:
                t_ub = min(t_bounds[1], float(uh1_t_ub))
                t_val = min(t_val, t_ub)
            else:
                t_ub = t_bounds[1]

            ref_model.set_parameter(
                f"{alias}_R_{j}", value=r_val,
                lower_bound=r_bounds[0], upper_bound=r_bounds[1],
            )
            ref_model.set_parameter(
                f"{alias}_uh{j}_T", value=t_val,
                lower_bound=t_bounds[0], upper_bound=t_ub,
            )
            ref_model.set_parameter(
                f"{alias}_uh{j}_K", value=k_val,
                lower_bound=k_bounds[0], upper_bound=k_bounds[1],
            )

        # IA parameters — temperature params are fixed by default to reduce
        # search dimensionality.  The calibrate flag controls whether Optuna
        # includes them in the search space.
        ref_model.set_parameter(
            f"{alias}_ia_max",
            value=min(0.10, _ia_max_ub),
            upper_bound=_ia_max_ub,
        )
        ref_model.set_parameter(f"{alias}_ia_k0",       value=0.0010)
        ref_model.set_parameter(
            f"{alias}_ia_kT", value=ia_kt,
            calibrate=calibrate_temperature_params,
        )
        ref_model.set_parameter(f"{alias}_ia_theta",    value=0.4974)
        ref_model.set_parameter(
            f"{alias}_ia_T_ref", value=ia_t_ref,
            calibrate=calibrate_temperature_params,
        )
        ref_model.set_parameter(f"{alias}_ia_k_dep",    value=2.0)
        ref_model.set_parameter(
            f"{alias}_ia_T_freeze", value=ia_t_freeze,
            calibrate=calibrate_temperature_params,
        )
        ref_model.set_parameter(f"{alias}_snow_T",      value=1.0)
        ref_model.set_parameter(f"{alias}_snow_ddf",    value=0.12)
        ref_model.validate()

        # ---- Extract calibratable parameter metadata -----------------------
        # Log-scale sampling is used for parameters spanning > 1 order of
        # magnitude (R fractions, IA capacity, IA rate constants) because
        # Optuna's log sampling is more efficient in that regime.
        LOG_SCALE_SUFFIXES = ("_R_", "_ia_max", "_ia_k0", "_ia_k_dep")

        def _use_log(name: str, lb: float, ub: float) -> bool:
            if lb <= 0:
                return False
            if ub / lb < 10:
                return False
            return any(s in name for s in LOG_SCALE_SUFFIXES)

        self._cal_params = [
            n for n in ref_model.scalar_parameter_names
            if ref_model.get_scalar_parameter(n).calibrate
        ]
        self._param_meta = [
            (
                name,
                float(ref_model.get_scalar_parameter(name).lower_bound),
                float(ref_model.get_scalar_parameter(name).upper_bound),
                _use_log(
                    name,
                    float(ref_model.get_scalar_parameter(name).lower_bound),
                    float(ref_model.get_scalar_parameter(name).upper_bound),
                ),
            )
            for name in self._cal_params
        ]

        # ---- Build per-subcatchment templates --------------------------------
        # The reference model above has the watershed-level area; each
        # subcatchment template is a deep copy with its own area substituted.
        # Deep-copying a validated EnsembleModel is much faster than calling
        # make_rtk_ensemble + all set_parameter calls again for every Optuna
        # trial.
        self._templates = []
        for _, sc_row in self._sc_df.iterrows():
            template = make_rtk_ensemble(
                n_models=self.n_models,
                n_triangles=self.n_triangles,
                units=self.units,
                area_acres=sc_row["area_acres"],
                ia_defaults=self.ia_defaults,
                output_name="rdii_cfs",
            )
            template.validate()
            _alias = template.aliases[0]

            # Mirror all bounds from the reference model onto each template.
            _i = 0
            for _j in range(1, self.n_triangles + 1):
                _r_val, _r_bounds = x0[_i], bounds[_i]; _i += 1
                _t_val, _t_bounds = x0[_i], bounds[_i]; _i += 1
                _k_val, _k_bounds = x0[_i], bounds[_i]; _i += 1

                if _j == 1 and uh1_t_ub is not None:
                    _t_ub = min(_t_bounds[1], float(uh1_t_ub))
                    _t_val = min(_t_val, _t_ub)
                else:
                    _t_ub = _t_bounds[1]

                template.set_parameter(
                    f"{_alias}_R_{_j}", value=_r_val,
                    lower_bound=_r_bounds[0], upper_bound=_r_bounds[1],
                )
                template.set_parameter(
                    f"{_alias}_uh{_j}_T", value=_t_val,
                    lower_bound=_t_bounds[0], upper_bound=_t_ub,
                )
                template.set_parameter(
                    f"{_alias}_uh{_j}_K", value=_k_val,
                    lower_bound=_k_bounds[0], upper_bound=_k_bounds[1],
                )

            template.set_parameter(
                f"{_alias}_ia_max",
                value=min(0.10, _ia_max_ub),
                upper_bound=_ia_max_ub,
            )
            template.set_parameter(f"{_alias}_ia_k0",       value=0.0010)
            template.set_parameter(
                f"{_alias}_ia_kT", value=ia_kt,
                calibrate=calibrate_temperature_params,
            )
            template.set_parameter(f"{_alias}_ia_theta",    value=0.4974)
            template.set_parameter(
                f"{_alias}_ia_T_ref", value=ia_t_ref,
                calibrate=calibrate_temperature_params,
            )
            template.set_parameter(f"{_alias}_ia_k_dep",    value=2.0)
            template.set_parameter(
                f"{_alias}_ia_T_freeze", value=ia_t_freeze,
                calibrate=calibrate_temperature_params,
            )
            template.set_parameter(f"{_alias}_snow_T",      value=1.0)
            template.set_parameter(f"{_alias}_snow_ddf",    value=0.12)
            template.validate()

            # Attach the subcatchment name to the template for diagnostics.
            template._name = sc_row["name"]
            self._templates.append(template)

        self._templates_built = True

    # -----------------------------------------------------------------------
    # Forcing DataFrame builder (used by CalOptimizer)
    # -----------------------------------------------------------------------

    def _get_sc_df(
        self,
        sc_row: pd.Series,
        observed: Optional[pd.Series],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Build a ready-to-use forcing DataFrame for one subcatchment and time
        window, merging PPT/TMEAN forcing with the observed flow series and
        adding sparsehydro seasonal time features.

        Parameters
        ----------
        sc_row : pd.Series
            A row from ``_sc_df`` (must have a ``rain_gage`` key).
        observed : pd.Series
            Observed flow (CFS), indexed by datetime.
        start, end : pd.Timestamp
            Inclusive date bounds for the slice.

        Returns
        -------
        pd.DataFrame
            Columns: datetime, rainfall_in, temperature_c (optional),
            flow_cfs, plus seasonal feature columns from compute_time_features.
        """
        # Preferred path: resolve precipitation + temperature by the series-id
        # pointers written by load_forcing_from_sc_df().  This joins whichever
        # unique cached series this subcatchment references — no gage grouping.
        ppt_id = sc_row.get("_ppt_id")
        forcing = None
        if ppt_id is not None and ppt_id in self._ppt_cache:
            forcing = self._ppt_cache[ppt_id]
            temp_id = sc_row.get("_temp_id")
            if temp_id is not None and temp_id in self._temp_cache:
                forcing = forcing.join(
                    self._temp_cache[temp_id], how="inner"
                ).dropna()

        # Fallback: legacy combined cache (manual load_forcing() / single-SC),
        # keyed by rain gage then the single-SC 'default' key.
        if forcing is None:
            key = sc_row.get("rain_gage", "default")
            forcing = self._forcing_cache.get(key)
            if forcing is None:
                forcing = self._forcing_cache.get("default")
        if forcing is None:
            raise KeyError(
                f"No forcing data cached for subcatchment {sc_row.get('name', '?')!r}. "
                "Call load_forcing() or load_forcing_from_sc_df() first."
            )

        if observed is not None:
            df = (
                forcing.join(observed.rename("flow_cfs"), how="inner")
                .dropna()
                .reset_index()
            )
        else:
            df = forcing.dropna().reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "datetime"})

        df = compute_time_features(df)

        mask = (df["datetime"] >= start) & (df["datetime"] <= end)
        return df.loc[mask].reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> None:
        """
        Print a human-readable summary of the model configuration.

        Shows: model name, number of subcatchments, total area, IA defaults,
        per-subcatchment table (name / area / BFI / gage), and forcing date
        ranges.
        """
        print(f"RdiiModel: {self.name or '(unnamed)'}")
        print(
            f"  {len(self._sc_df)} subcatchment(s)  |  "
            f"total area: {self.area_acres:,.1f} ac  |  "
            f"N_models={self.n_models}  N_triangles={self.n_triangles}  "
            f"units={self.units}"
        )
        print(f"  ia_defaults: {self.ia_defaults}")
        print()

        # Resolve a display label for each subcatchment's temperature source:
        # the embedded TIMESERIES name, or the temp-file basename, or '-'.
        def _temp_label(r) -> str:
            series = r.get("temp_series", "")
            if series:
                return str(series)
            tfile = r.get("temp_file", "") or r.get("tmean_file", "")
            if tfile:
                return os.path.splitext(os.path.basename(str(tfile)))[0]
            return "-"

        # Subcatchment table
        col_w = max(len(str(n)) for n in self._sc_df["name"]) + 2
        has_temp = any(_temp_label(r) != "-" for _, r in self._sc_df.iterrows())
        gage_w = max(
            [len("Rain gage")]
            + [len(str(r.get("rain_gage", ""))) for _, r in self._sc_df.iterrows()]
        ) + 2

        # Detect any imperviousness column present in _sc_df.
        _imperv_candidates = [c for c in self._sc_df.columns if "imperv" in c.lower() or c.lower() == "imperv"]
        imperv_col = _imperv_candidates[0] if _imperv_candidates else None

        header = (
            f"  {'Subcatchment':<{col_w}}  {'Area (ac)':>12}  {'BFI':>6}  "
            f"{'Area wt':>8}  {'BFI wt':>8}"
        )
        if imperv_col:
            header += f"  {'%Imperv':>8}"
        header += f"  {'Rain gage':<{gage_w}}"
        if has_temp:
            header += "Temp gage"
        print(header)
        sep_w = col_w + 52 + (10 if imperv_col else 0) + gage_w + (9 if has_temp else 0)
        print("  " + "-" * sep_w)
        for _, r in self._sc_df.iterrows():
            line = (
                f"  {str(r['name']):<{col_w}}  {r['area_acres']:>12,.2f}  "
                f"{r['bfi']:>6.3f}  {r['area_weight']:>8.4f}  "
                f"{r['bfi_weight']:>8.4f}"
            )
            if imperv_col:
                imp_val = r.get(imperv_col, None)
                if imp_val is not None and str(imp_val) not in ("", "nan", "None"):
                    line += f"  {float(imp_val):>8.2f}"
                else:
                    line += f"  {'—':>8}"
            line += f"  {str(r.get('rain_gage','')):<{gage_w}}"
            if has_temp:
                line += _temp_label(r)
            print(line)
        print()

        # Forcing summary — report the unique loaded series.
        def _basename(s: str) -> str:
            # Show a readable label: file basename, or series name as-is.
            return os.path.basename(s) if (os.sep in s or "/" in s) else s

        if self._ppt_cache or self._temp_cache:
            print("  Forcing data (unique series):")
            for sid, df in self._ppt_cache.items():
                dr = f"{df.index.min().date()} -> {df.index.max().date()}"
                print(f"    [ppt ] {_basename(sid):<22}  {dr}  ({len(df):,} rows)")
            for sid, df in self._temp_cache.items():
                dr = f"{df.index.min().date()} -> {df.index.max().date()}"
                print(f"    [temp] {_basename(sid):<22}  {dr}  ({len(df):,} rows)")
        elif self._forcing_cache:
            print("  Forcing data (forcing cache):")
            for key, df in self._forcing_cache.items():
                date_range = f"{df.index.min().date()} -> {df.index.max().date()}"
                cols = ", ".join(df.columns.tolist())
                print(f"    [{key}]  {date_range}  cols: {cols}")
        else:
            print("  Forcing data: NOT LOADED")

        # Observed flow summary
        if self._observed is not None:
            obs = self._observed
            print(
                f"\n  Observed flow: {obs.index.min().date()} → "
                f"{obs.index.max().date()}  "
                f"mean={obs.mean():.1f} CFS  max={obs.max():.1f} CFS"
            )
        else:
            print("\n  Observed flow: NOT LOADED")

    def __repr__(self) -> str:
        return (
            f"RdiiModel(name={self.name!r}, "
            f"n_sc={len(self._sc_df)}, "
            f"area_acres={self.area_acres:.1f}, "
            f"n_models={self.n_models}, "
            f"n_triangles={self.n_triangles})"
        )
