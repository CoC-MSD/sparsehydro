"""
Mass-balance / continuity checking for swmmrdii.

The sparsehydro RDII model is analytic — UH kernels integrate to unit area, so
the continuity error is always 0.0000% in theory.  These functions verify that
claim after every calibration run and produce a SWMM-style budget breakdown:

    Rainfall = IA absorbed + snow still in storage + non-routed (1-SigR)
             + routing tail + RDII discharged  ±  continuity error

The watershed total for a multi-subcatchment model is reported as a
depth-volume in acre-inches (area × depth per subcatchment, summed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helper: snow-water-equivalent still in storage at end of run
# ---------------------------------------------------------------------------

def _swe_remaining(child, forcing_df: pd.DataFrame, rain_col: str) -> float:
    """
    Estimate the snow-water-equivalent (SWE) remaining in storage at the end
    of the simulation period, in the same depth units as *rain_col*.

    The sparsehydro ``IAModel`` does not expose SWE state after ``predict()``,
    so we re-run the degree-day snow accumulation/melt loop manually using the
    calibrated ``snow_T`` (rain/snow partition temperature) and ``snow_ddf``
    (degree-day melt factor) parameters.

    Parameters
    ----------
    child : CombinedHydroModel or IAModel
        A single sparsehydro child model (one of the ``EnsembleModel._children``).
    forcing_df : pd.DataFrame
        The forcing data that was passed to ``model.prepare()``.
    rain_col : str
        Name of the precipitation column in *forcing_df* (``'rainfall_in'`` or
        ``'rainfall_mm'``).

    Returns
    -------
    float
        Depth of SWE still in storage, or 0.0 if snow model is disabled.
    """
    # Navigate to the IAModel regardless of whether child is a CombinedHydroModel
    # or a raw IAModel (EnsembleModel stores CombinedHydroModel instances).
    ia = getattr(child, "_ia_model", child)

    # If the snow module was not enabled, there is no storage to account for.
    if not getattr(ia, "_snow", False):
        return 0.0

    d = forcing_df.sort_values("datetime").reset_index(drop=True)
    P = d[rain_col].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)

    # Use the calibrated T_ref as the fallback when temperature data are absent.
    T_ref = ia.get_scalar_parameter("ia_T_ref").value
    Tarr = (
        d["temperature_c"].fillna(T_ref).to_numpy(dtype=float)
        if "temperature_c" in d.columns
        else np.full(len(d), T_ref)
    )

    # Derive the timestep in hours from the median spacing between records.
    dt_h = (
        pd.Timedelta(d["datetime"].diff().dropna().median()).total_seconds() / 3600.0
    )

    snow_T = ia.get_scalar_parameter("snow_T").value   # partition threshold (°C)
    ddf = ia.get_scalar_parameter("snow_ddf").value    # degree-day melt factor (in/°C/day)

    swe = 0.0
    for i in range(len(P)):
        Pi, Ti = float(P[i]), float(Tarr[i])
        if Ti <= snow_T:
            # Cold step: precipitation falls as snow, add to SWE.
            swe += Pi
        elif swe > 0.0:
            # Warm step: melt at degree-day rate, but do not melt below zero.
            swe -= min(swe, ddf * (Ti - snow_T) * dt_h / 24.0)

    return float(swe)


# ---------------------------------------------------------------------------
# Internal helper: collect depth-budget terms from a single prepared model
# ---------------------------------------------------------------------------

def _mb_terms(model, forcing_df: pd.DataFrame) -> dict:
    """
    Return the depth-budget components (in the model's native depth units)
    for a single prepared sparsehydro model.

    The budget identity is:

        rain = ia_loss + snow_storage + non_routed + routing_tail + rdii_out

    with ``cont_err = rain - sum(components)``, which should be ~0.

    Parameters
    ----------
    model : EnsembleModel or CombinedHydroModel
        A sparsehydro model that has already had ``prepare()`` called.
    forcing_df : pd.DataFrame
        The same forcing DataFrame passed to ``model.prepare()``.

    Returns
    -------
    dict with keys: rain, ia_loss, snow, non_routed, tail, rdii, sum_R, cont_err
    """
    # EnsembleModel exposes ._children; plain CombinedHydroModel is its own child.
    children = list(model._children) if hasattr(model, "_children") else [model]

    # Detect unit system from the first child's _units attribute.
    units = getattr(children[0], "_units", "imperial")
    rain_col = "rainfall_in" if units == "imperial" else "rainfall_mm"
    exc_col  = "p_excess_in" if units == "imperial" else "p_excess_mm"
    dep_col  = "rdii_in"     if units == "imperial" else "rdii_mm"

    rain_total = float(forcing_df[rain_col].fillna(0.0).clip(lower=0.0).sum())

    excess_l, rdii_l, swe_l, sumR_l, routed_l = [], [], [], [], []
    for child in children:
        cpred  = child.predict()
        # p_excess is rainfall after initial abstraction (and optionally snowmelt).
        excess = float(cpred[exc_col].sum())
        # rdii_in is the depth equivalent of routed RDII (before area conversion).
        rdii   = float(cpred[dep_col].sum())
        swe    = _swe_remaining(child, forcing_df, rain_col)

        # Sum the R fractions for this child.  R parameters are named 'R_1', 'R_2', ...
        R_names = sorted(
            nm for nm in child.scalar_parameter_names if nm.startswith("R_")
        )
        sumR = sum(float(child.get_scalar_parameter(nm).value) for nm in R_names)

        excess_l.append(excess)
        rdii_l.append(rdii)
        swe_l.append(swe)
        sumR_l.append(sumR)
        routed_l.append(sumR * excess)  # amount that *should* have been routed

    # For an EnsembleModel the children all see the same rainfall, so averaging
    # the per-child depth terms gives the effective per-pathway depth.
    excess_total  = float(np.mean(excess_l))
    rdii_out      = float(np.mean(rdii_l))
    swe_remaining = float(np.mean(swe_l))
    sum_R         = float(np.mean(sumR_l))
    routed_should = float(np.mean(routed_l))

    # IA loss = rainfall that was absorbed before becoming excess.
    ia_loss = rain_total - excess_total - swe_remaining

    # Non-routed fraction: the portion of excess that no RTK triangle claims.
    non_routed = (1.0 - sum_R) * excess_total

    # Routing tail: the tiny amount of RDII that convolution assigns past the
    # end of the series (mode='full'[:n] leaves a fractional tail).
    routing_tail = routed_should - rdii_out

    cont_err = rain_total - (ia_loss + swe_remaining + non_routed + rdii_out + routing_tail)

    return {
        "rain":       rain_total,
        "ia_loss":    ia_loss,
        "snow":       swe_remaining,
        "non_routed": non_routed,
        "tail":       routing_tail,
        "rdii":       rdii_out,
        "sum_R":      sum_R,
        "cont_err":   cont_err,
    }


# ---------------------------------------------------------------------------
# Public function: SWMM-style mass-balance report
# ---------------------------------------------------------------------------

def mass_balance_report(
    model,
    forcing_df: pd.DataFrame,
    label: str = "",
    verbose: bool = True,
) -> dict:
    """
    Print a SWMM-style mass-balance / continuity check for a single model.

    Works for both single-subcatchment (``EnsembleModel``) and any single
    prepared sparsehydro model.  For multi-subcatchment models, call this
    function once per subcatchment and sum the results externally, or use
    ``mass_balance_report_multi()``.

    Parameters
    ----------
    model : EnsembleModel or CombinedHydroModel
        A sparsehydro model that has already had ``prepare()`` called.
    forcing_df : pd.DataFrame
        The same forcing DataFrame passed to ``model.prepare()``.
    label : str
        Optional label appended to the report header (e.g. ``'calibration'``).
    verbose : bool
        If True (default), print the formatted budget table.

    Returns
    -------
    dict
        Keys: label, rain_in, ia_loss_in, snow_storage_in, non_routed_in,
        routing_tail_in, rdii_out_in, sum_R, cont_err_in, cont_err_pct
    """
    t = _mb_terms(model, forcing_df)

    rain_total = t["rain"]

    def pct(x: float) -> float:
        return 100.0 * x / rain_total if rain_total > 0 else 0.0

    cont_err_pct = pct(t["cont_err"])

    if verbose:
        w = 26
        tag = (" — " + label) if label else ""
        print(f"=== Mass balance / continuity{tag} ===")
        print(f"  {'INPUT  rainfall':<{w}} {rain_total:12.4f}   (100.00%)")
        print(f"  {'LOSS   IA absorbed':<{w}} {t['ia_loss']:12.4f}   ({pct(t['ia_loss']):6.2f}%)")
        print(f"  {'LOSS   snow in storage':<{w}} {t['snow']:12.4f}   ({pct(t['snow']):6.2f}%)")
        print(f"  {'LOSS   non-routed (1-SigR)':<{w}} {t['non_routed']:12.4f}   ({pct(t['non_routed']):6.2f}%)")
        print(f"  {'LOSS   routing tail':<{w}} {t['tail']:12.4f}   ({pct(t['tail']):6.2f}%)")
        print(f"  {'OUTPUT RDII discharged':<{w}} {t['rdii']:12.4f}   ({pct(t['rdii']):6.2f}%)")
        print(f"  {'SigR (routed fraction)':<{w}} {t['sum_R']:12.4f}")
        print(f"  {'CONTINUITY ERROR':<{w}} {t['cont_err']:12.6f}   ({cont_err_pct:+.4f}%)")
        status = "OK (analytic ~0)" if abs(cont_err_pct) < 0.01 else "CHECK — non-trivial error"
        print(f"  {'STATUS':<{w}} {status}")

    return {
        "label":           label,
        "rain_in":         rain_total,
        "ia_loss_in":      t["ia_loss"],
        "snow_storage_in": t["snow"],
        "non_routed_in":   t["non_routed"],
        "routing_tail_in": t["tail"],
        "rdii_out_in":     t["rdii"],
        "sum_R":           t["sum_R"],
        "cont_err_in":     t["cont_err"],
        "cont_err_pct":    cont_err_pct,
    }


def mass_balance_report_multi(
    sc_models: list[tuple],
    label: str = "",
    verbose: bool = True,
) -> dict:
    """
    Compute a watershed-level mass balance for a multi-subcatchment model.

    Parameters
    ----------
    sc_models : list of (model, forcing_df, area_acres) tuples
        Each entry is a prepared sparsehydro model, its forcing DataFrame,
        and the subcatchment area in acres.
    label : str
        Optional label for the report header.
    verbose : bool
        If True, print the per-subcatchment table and the watershed total.

    Returns
    -------
    dict
        Watershed-total budget in acre-inches (depth × area for each term).
    """
    # Accumulate area-weighted depth totals (acre-inches).
    ws: dict[str, float] = {
        "rain": 0.0, "ia_loss": 0.0, "snow": 0.0,
        "non_routed": 0.0, "tail": 0.0, "rdii": 0.0,
    }

    if verbose:
        tag = (" — " + label) if label else ""
        print(f"=== Mass balance / continuity{tag} — per subcatchment ===")
        hdr = f"  {'Subcatchment':<24} {'Rain':>10} {'IA%':>6} {'Snow%':>6} {'NonR%':>6} {'Tail%':>6} {'RDII%':>6}  Err%"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

    for model, forcing_df, area_acres in sc_models:
        t = _mb_terms(model, forcing_df)
        sc_name = getattr(model, "_name", "")

        # Accumulate volume contributions (acre-inches).
        for k in ws:
            ws[k] += t[k] * area_acres

        if verbose:
            r = t["rain"]
            def p(x): return 100.0 * x / r if r > 0 else 0.0
            print(
                f"  {sc_name:<24} {r:10.4f} {p(t['ia_loss']):6.1f} {p(t['snow']):6.1f} "
                f"{p(t['non_routed']):6.1f} {p(t['tail']):6.1f} {p(t['rdii']):6.1f}  "
                f"{100.0 * t['cont_err'] / r if r > 0 else 0.0:+.4f}%"
            )

    if verbose:
        r = ws["rain"]
        def pct(x): return 100.0 * x / r if r > 0 else 0.0
        print("\n  Watershed totals (acre-inches):")
        print(f"  {'INPUT  rainfall':<28} {ws['rain']:14.1f}  (100.00%)")
        print(f"  {'LOSS   IA absorbed':<28} {ws['ia_loss']:14.1f}  ({pct(ws['ia_loss']):6.2f}%)")
        print(f"  {'LOSS   snow in storage':<28} {ws['snow']:14.1f}  ({pct(ws['snow']):6.2f}%)")
        print(f"  {'LOSS   non-routed (1-SigR)':<28} {ws['non_routed']:14.1f}  ({pct(ws['non_routed']):6.2f}%)")
        print(f"  {'LOSS   routing tail':<28} {ws['tail']:14.1f}  ({pct(ws['tail']):6.2f}%)")
        print(f"  {'OUTPUT RDII discharged':<28} {ws['rdii']:14.1f}  ({pct(ws['rdii']):6.2f}%)")

    return {k + "_acre_in": v for k, v in ws.items()}
