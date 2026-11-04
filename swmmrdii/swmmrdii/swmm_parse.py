"""
SWMM INP and PCSWMM SQLite DB parsers for swmmrdii.

Primary entry point: ``load_swmm_subcatchments()``, which returns a DataFrame
describing each subcatchment (name, area, rain gage, ppt/tmean file paths, BFI).
"""

import os
import re
import sqlite3

import pandas as pd


def _get_db_columns(db_path: str) -> list[str]:
    """Return the column names of Subcatchments_User_Fields (excluding 'Name')."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT * FROM Subcatchments_User_Fields LIMIT 0")
        return [d[0] for d in cursor.description if d[0] != "Name"]
    finally:
        conn.close()


def _sanitize_field_name(field: str) -> str:
    """
    Convert a DB user-field name into a query-friendly column name.

    Lower-cased; a leading ``%`` becomes ``pct_``; any remaining ``%`` and
    spaces become underscores.  Examples: ``'CalTag'`` -> ``'caltag'``,
    ``'%Imperv'`` -> ``'pct_imperv'``, ``'Land Use'`` -> ``'land_use'``.
    """
    f = field.strip()
    pct_prefix = f.startswith("%")
    f = f.lstrip("%")
    col = f.replace("%", "pct_").replace(" ", "_").lower()
    if pct_prefix:
        col = "pct_" + col
    return col


def _maybe_numeric(series: pd.Series) -> pd.Series:
    """
    Return *series* coerced to numeric only when it is safe to do so.

    "Safe" means every non-blank value round-trips: ``str(float(v))`` matches
    the original after dropping an insignificant trailing ``.0``.  This keeps
    genuine measurements numeric (``'1.954'`` -> ``1.954``) while leaving
    identifier-like fields untouched — a zero-padded ``CalTag`` such as
    ``'03146000'`` would lose its leading zero and break string equality, so it
    stays a string.
    """
    s = series.copy()
    num = pd.to_numeric(s, errors="coerce")
    nonblank = s.astype(str).str.strip()
    mask = ~nonblank.isin(["", "None", "nan", "NaN"])
    if not mask.any():
        return s
    # Every non-blank value must parse as a number...
    if num[mask].isna().any():
        return s
    # ...and none may be a zero-padded integer (an identifier like a CalTag),
    # which would silently lose its leading zero if coerced.
    for tok in nonblank[mask]:
        body = tok.lstrip("+-")
        if len(body) > 1 and body[0] == "0" and "." not in body:
            return s
    return num


def parse_swmm_section(inp_path: str, section_name: str) -> list[list[str]]:
    """
    Extract data rows from a named [SECTION] in a SWMM INP file.

    Returns a list of tokenised rows (each row is a list of strings).
    Comment lines (starting with ``;``) and blank lines are skipped.
    """
    rows: list[list[str]] = []
    in_section = False
    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.upper() == f"[{section_name.upper()}]":
                in_section = True
                continue
            if in_section:
                if stripped.startswith("["):
                    break
                if stripped.startswith(";;") or stripped.startswith(";") or not stripped:
                    continue
                rows.append(re.split(r"\s+", stripped))
    return rows


def scan_swmm_sections(
    inp_path: str, section_names
) -> dict[str, list[list[str]]]:
    """
    Extract data rows from several [SECTION]s in ONE pass over the file.

    Same row semantics as :func:`parse_swmm_section` (comments/blanks
    skipped, whitespace-tokenised), but reads the file once regardless of
    how many sections are requested and stops as soon as the last wanted
    section has been read — with per-section ``parse_swmm_section`` calls,
    every ABSENT section costs a full scan of the file, which matters for
    multi-hundred-MB PCSWMM exports.

    Returns
    -------
    dict
        ``{section_name: rows}`` for every requested name (empty list when
        the section is absent).  Keys match *section_names* as given.
    """
    wanted = {f"[{str(s).upper()}]": str(s) for s in section_names}
    rows: dict[str, list[list[str]]] = {name: [] for name in wanted.values()}
    pending = set(wanted)          # section headers not yet encountered
    current: str | None = None     # section currently being collected
    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("["):
                header = stripped.upper()
                pending.discard(header)
                current = wanted.get(header)
                if current is None and not pending:
                    break          # every wanted section seen and closed
                continue
            if current is None or not stripped or stripped.startswith(";"):
                continue
            rows[current].append(re.split(r"\s+", stripped))
    return rows


def _parse_raingages(inp_path: str) -> dict[str, dict]:
    """
    Parse the ``[RAINGAGES]`` section.

    Returns ``{gage_name: {"source": "FILE"|"TIMESERIES", "ref": <str>}}``
    where *ref* is the file path (FILE form) or the time-series name
    (TIMESERIES form).

    SWMM raingage line layouts:

        Name  Format  Interval  SCF  FILE       "path"  [station]  [units]
        Name  Format  Interval  SCF  TIMESERIES  Tseries

    The 5th token is the Source keyword; what follows depends on it.
    """
    from .forcing import _split_keep_quotes

    gages: dict[str, dict] = {}
    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    rg_section = re.search(r"\[RAINGAGES\](.*?)(?=^\[|\Z)", content, re.DOTALL | re.MULTILINE)
    if not rg_section:
        return gages

    for line in rg_section.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        tokens = _split_keep_quotes(line)  # keeps "quoted path" intact
        if len(tokens) < 6:
            continue
        name = tokens[0]
        source = tokens[4].upper()
        if source == "FILE":
            gages[name] = {"source": "FILE", "ref": tokens[5].strip('"')}
        elif source == "TIMESERIES":
            gages[name] = {"source": "TIMESERIES", "ref": tokens[5]}
    return gages


def load_swmm_subcatchments(
    inp_path: str,
    db_path: str | None = None,
    name_field: str = "Name",
    area_field: str = "Area",
    bfi_field: str | None = "BFI",
    temp_dir: str | None = None,
    temp_suffix: str = "_tmean",
    temp_field: str | None = None,
    temp_unit: str | None = None,
    where: str | None = None,
    # Backwards-compatible alias — prefer temp_dir in new code.
    tmean_dir: str | None = None,
) -> pd.DataFrame:
    """
    Load subcatchment metadata from a SWMM INP file (and optionally a PCSWMM DB).

    Parameters
    ----------
    inp_path : str
        Path to the SWMM .inp model file.
    db_path : str or None
        Path to the PCSWMM SQLite .db file.  Required only when *bfi_field* is
        not None and the BFI values live in the DB rather than in the INP itself.
    name_field : str
        Output column label for the subcatchment name (default ``'Name'``).
    area_field : str
        Output column label for the area column (default ``'Area'``).
    bfi_field : str or None
        Name of the BFI attribute in the PCSWMM DB table
        ``Subcatchments_User_Fields``.  Pass ``None`` to skip BFI loading; all
        subcatchments get ``bfi = 1.0`` (no BFI scaling).
    temp_dir : str or None
        Directory containing temperature .dat files.  When provided, the loader
        derives temperature file paths from the PPT filename by appending
        *temp_suffix* to the basename (e.g. ``PRISM_RG_022.dat`` →
        ``PRISM_RG_022_tmean.dat`` with the default suffix).  Pass ``None`` to
        skip; the ``temp_file`` column in the output will be empty strings.
    temp_suffix : str
        Suffix appended to the PPT file basename when deriving temperature file
        paths (default ``'_tmean'``).  Change this if your temperature files
        use a different naming convention.
    temp_field : str or None
        Name of a user-defined attribute in the PCSWMM DB table
        ``Subcatchments_User_Fields`` that, per subcatchment, names a SWMM
        time series in the INP's ``[TIMESERIES]`` section holding temperature
        data.  When provided, the loader resolves each subcatchment's temp
        series (inline rows *or* a ``FILE`` reference) directly from the INP —
        no *temp_dir* is needed.  Takes precedence over *temp_dir* derivation.
    temp_unit : str or None
        Unit the temperature data is stored in (e.g. ``'degF'``, ``'degC'``,
        ``'K'``).  Recorded in the output ``temp_unit`` column so the forcing
        loader can convert to °C.  Required when *temp_field* is used and the
        data is not already in °C.
    where : str or None
        Optional Pandas ``.query()`` expression applied after all joins, e.g.
        ``"area_acres > 100"``, ``"name in ['SC1', 'SC2']"``, or — referencing
        any DB user-field — ``"caltag == '03146000'"``.  When *db_path* is
        provided, every column in ``Subcatchments_User_Fields`` is attached
        automatically (sanitised to lower-case: ``CalTag`` -> ``caltag``,
        ``%Imperv`` -> ``pct_imperv``), so the query can reference them with no
        extra setup.
    tmean_dir : str or None
        Deprecated alias for *temp_dir*.  Accepted for backward compatibility;
        if both are provided, *temp_dir* takes precedence.

    Returns
    -------
    pd.DataFrame with columns:
        name, area_acres, outlet, rain_gage,
        imperv, width, slope (native INP [SUBCATCHMENTS] attributes),
        ppt_source ('FILE'|'TIMESERIES'), ppt_file, ppt_series,
        temp_source ('FILE'|'TIMESERIES'|''), temp_file, temp_series,
        temp_unit, bfi, area_weight, bfi_weight,
        plus every DB user-field under its sanitised lower-case name
        (a user-field whose sanitised name collides with an INP-native
        column above is shadowed by the INP value)

    Notes
    -----
    The INP path is recorded on the returned frame as ``df.attrs['inp_path']``
    so downstream loaders can read embedded ``[TIMESERIES]`` blocks without
    being passed the path again.
    """
    # Resolve deprecated alias.
    if temp_dir is None and tmean_dir is not None:
        temp_dir = tmean_dir

    # Fail-loud policy: a DB field the caller SPECIFIED must be resolvable.
    # Without a .db there is nothing to resolve against, so raise rather
    # than silently substituting defaults (BFI=1.0 / no temperature) — the
    # caller must opt out explicitly by passing None.
    if bfi_field is not None and db_path is None:
        raise ValueError(
            f"bfi_field={bfi_field!r} was specified but no PCSWMM .db is "
            f"available (db_path=None). Provide the .db containing the "
            f"{bfi_field!r} user-field, or pass bfi_field=None to use a "
            f"flat BFI of 1.0 for all subcatchments."
        )
    if temp_field is not None and db_path is None:
        raise ValueError(
            f"temp_field={temp_field!r} was specified but no PCSWMM .db is "
            f"available (db_path=None). Provide the .db containing the "
            f"{temp_field!r} user-field, or pass temp_field=None to skip "
            f"per-SC temperature series."
        )
    # --- Parse [SUBCATCHMENTS] section -----------------------------------------
    sc_rows = parse_swmm_section(inp_path, "SUBCATCHMENTS")

    def _tok_float(r, i):
        """Optional numeric token — NaN when absent or non-numeric."""
        try:
            return float(r[i])
        except (IndexError, ValueError):
            return float("nan")

    # Token order: name  rain_gage  outlet  area  %imperv  width  %slope  curb
    # The INP-native physical attributes (imperv, width, slope) are kept as
    # columns so field lookups (imperv_field=, WEIGHTS, `where` queries) can
    # reference them exactly like DB user-fields.  A DB user-field with the
    # same sanitised name is shadowed by the INP column (see merge below).
    subcatchments = pd.DataFrame(
        [
            {
                name_field: r[0],
                "rain_gage": r[1],
                "outlet": r[2],
                area_field: float(r[3]),
                "imperv": _tok_float(r, 4),
                "width": _tok_float(r, 5),
                "slope": _tok_float(r, 6),
            }
            for r in sc_rows
        ]
    ).rename(columns={name_field: "name", area_field: "area_acres"})

    # --- Map rain gages to their precipitation source --------------------------
    # Each gage is either a FILE (external .dat path) or a TIMESERIES (named
    # series embedded in the INP [TIMESERIES] block).
    gage_map = _parse_raingages(inp_path)

    def _ppt_source(g):
        return gage_map.get(g, {}).get("source", "")

    def _ppt_file(g):
        info = gage_map.get(g, {})
        return info["ref"] if info.get("source") == "FILE" else ""

    def _ppt_series(g):
        info = gage_map.get(g, {})
        return info["ref"] if info.get("source") == "TIMESERIES" else ""

    subcatchments["ppt_source"] = subcatchments["rain_gage"].map(_ppt_source)
    subcatchments["ppt_file"]   = subcatchments["rain_gage"].map(_ppt_file)
    subcatchments["ppt_series"] = subcatchments["rain_gage"].map(_ppt_series)

    # --- Resolve temperature source --------------------------------------------
    # Priority:
    #   1. temp_field (DB attribute naming a per-SC [TIMESERIES]) — preferred.
    #   2. temp_dir derivation from the PPT filename (legacy convention).
    #   3. none.
    subcatchments["temp_source"] = ""
    subcatchments["temp_file"]   = ""
    subcatchments["temp_series"] = ""
    subcatchments["temp_unit"]   = temp_unit or ""

    if temp_field is not None and db_path is not None:
        # Validate the column exists before querying.
        _db_cols = _get_db_columns(db_path)
        if temp_field not in _db_cols:
            raise ValueError(
                f"temp_field={temp_field!r} not found in Subcatchments_User_Fields.\n"
                f"Available columns: {_db_cols}"
            )
        # Read the per-SC temp-series-name attribute from the PCSWMM DB.
        conn = sqlite3.connect(db_path)
        df_temp = pd.read_sql(
            f"SELECT Name, \"{temp_field}\" AS temp_series FROM Subcatchments_User_Fields",
            conn,
        )
        conn.close()
        df_temp = df_temp.rename(columns={"Name": "name"})
        df_temp["temp_series"] = df_temp["temp_series"].astype(str).str.strip()
        subcatchments = subcatchments.merge(
            df_temp[["name", "temp_series"]], on="name", how="left",
            suffixes=("", "_db"),
        )
        # Prefer the DB column when present.
        subcatchments["temp_series"] = (
            subcatchments.pop("temp_series_db")
            if "temp_series_db" in subcatchments.columns
            else subcatchments["temp_series"]
        ).fillna("").replace({"None": "", "nan": ""})
        # Each named series is resolved from the INP [TIMESERIES] block.
        subcatchments.loc[
            subcatchments["temp_series"].astype(bool), "temp_source"
        ] = "TIMESERIES"
    elif temp_dir is not None:
        def _ppt_to_temp(ppt_path: str) -> str:
            if not ppt_path:
                return ""
            base = os.path.splitext(os.path.basename(ppt_path))[0]
            return os.path.join(temp_dir, f"{base}{temp_suffix}.dat")

        subcatchments["temp_file"] = subcatchments["ppt_file"].map(_ppt_to_temp)
        subcatchments.loc[
            subcatchments["temp_file"].astype(bool), "temp_source"
        ] = "FILE"

    # --- Load BFI from PCSWMM DB -----------------------------------------------
    if bfi_field is not None and db_path is not None:
        # Validate the column exists before querying.
        _db_cols = _get_db_columns(db_path)
        if bfi_field not in _db_cols:
            raise ValueError(
                f"bfi_field={bfi_field!r} not found in Subcatchments_User_Fields.\n"
                f"Available columns: {_db_cols}"
            )
        conn = sqlite3.connect(db_path)
        df_db = pd.read_sql(
            f"SELECT Name, {bfi_field} FROM Subcatchments_User_Fields", conn
        )
        conn.close()
        df_db = df_db.rename(columns={"Name": "name", bfi_field: "bfi"})
        df_db["bfi"] = pd.to_numeric(df_db["bfi"], errors="coerce")
        subcatchments = subcatchments.merge(df_db[["name", "bfi"]], on="name", how="left")
    else:
        subcatchments["bfi"] = 1.0

    # Fill any remaining NaN BFI values with the mean (or 1.0 if all NaN)
    if subcatchments["bfi"].isna().any():
        fill_val = subcatchments["bfi"].mean()
        if pd.isna(fill_val):
            fill_val = 1.0
        subcatchments["bfi"] = subcatchments["bfi"].fillna(fill_val)

    # --- Auto-attach all DB user-fields so `where` can reference any of them ---
    # When a DB is provided, every column in Subcatchments_User_Fields is merged
    # in under a sanitised, lower-cased name (CalTag -> caltag, %Imperv ->
    # pct_imperv) so the definition query can filter on it without the caller
    # having to name the fields up front.  Columns already present (name, bfi,
    # the temp-series field) are not duplicated.
    if db_path is not None:
        try:
            conn = sqlite3.connect(db_path)
            df_uf = pd.read_sql("SELECT * FROM Subcatchments_User_Fields", conn)
            conn.close()
        except Exception as exc:
            print(
                f"[swmm_parse] Warning: could not read Subcatchments_User_Fields "
                f"({exc}). DB attributes will be unavailable to the where query."
            )
            df_uf = None

        if df_uf is not None and "Name" in df_uf.columns:
            df_uf = df_uf.rename(columns={"Name": "name"})
            existing = set(subcatchments.columns)
            keep = ["name"]
            for raw in df_uf.columns:
                if raw == "name":
                    continue
                col = _sanitize_field_name(raw)
                if col in existing or col in keep:
                    continue  # don't clobber name/bfi/temp_* etc.
                series = df_uf[raw]
                # Coerce to numeric only when it is SAFE — i.e. every non-blank
                # value round-trips through float without changing form.  This
                # keeps real measurements numeric (so `where` can use > / <)
                # while preserving identifier-like fields such as a zero-padded
                # CalTag ('03146000' must stay a string, not become 3146000.0).
                df_uf[raw] = _maybe_numeric(series)
                df_uf = df_uf.rename(columns={raw: col})
                keep.append(col)
            if len(keep) > 1:
                subcatchments = subcatchments.merge(
                    df_uf[keep], on="name", how="left"
                )

    # --- Optional definition query ---------------------------------------------
    if where is not None:
        subcatchments = subcatchments.query(where).reset_index(drop=True)

    if subcatchments.empty:
        raise ValueError(
            f"No subcatchments remain after applying where={where!r}. "
            "Check your definition query."
        )

    # --- Derived weight columns ------------------------------------------------
    total_area = subcatchments["area_acres"].sum()
    subcatchments["area_weight"] = subcatchments["area_acres"] / total_area

    bfi_area = subcatchments["area_acres"] * subcatchments["bfi"]
    bfi_area_total = bfi_area.sum()
    subcatchments["bfi_weight"] = (
        bfi_area / bfi_area_total if bfi_area_total > 0 else subcatchments["area_weight"]
    )

    out = subcatchments.reset_index(drop=True)
    # Record the INP path so downstream loaders can read embedded [TIMESERIES].
    out.attrs["inp_path"] = inp_path
    return out
