"""Rain-aware bimodality detection for storm sub-event splitting.

Ported (rain burst detector faithfully; flow crest detector streamlined) from the
Parsimonious Functions bimodality code.  Detects whether an event window contains
a meaningful secondary rain-driven flow crest, so multi-peak storms can be split
into single-peak sub-events.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_prominences

from .smoothing import savgol_curvature

# --- Default thresholds (from Parsimonious data_processing.py) --------------
BIMODALITY_USE_RAW_FLOW = True
BIMODALITY_FLAG_THRESHOLD = 0.12
BIMODALITY_MIN_SECONDARY_REL_HEIGHT = 0.08
BIMODALITY_MIN_SECONDARY_ABS_RISE = 0.04
BIMODALITY_MIN_VALLEY_REL_DIP = 0.025
BIMODALITY_MIN_VALLEY_ABS_DIP = 0.02
BIMODALITY_MIN_PEAK_DISTANCE = 6      # 5-min data: ≈ 30 min
BIMODALITY_MIN_PEAK_WIDTH = 2
BIMODALITY_MIN_PROM_FRAC = 0.025
BIMODALITY_MIN_PROM_ABS = 0.025
BIMODALITY_MIN_SEC_CURV_FRAC = 0.20
BIMODALITY_RAIN_LEAD_HRS = 8.0
BIMODALITY_RAIN_LAG_HRS = 0.75
BIMODALITY_MIN_RAIN_NEAR_PEAK = 0.01
BIMODALITY_MIN_RAIN_BURST = 0.01
BIMODALITY_RAIN_BURST_GAP_HRS = 1.0
BIMODALITY_SPLIT_MIN_SEPARATION_HRS = 2.0
BIMODALITY_SPLIT_MAX_VALLEY_FRAC = 0.65
BIMODALITY_RAIN_TARGET_RATIO = 0.30
BIMODALITY_RAIN_INDEX_WEIGHT = 0.15


def calculate_rain_bimodality(
    rain,
    event_start_idx: int,
    event_end_idx: int,
    *,
    dt_minutes: float = 5.0,
    burst_gap_hours: float = BIMODALITY_RAIN_BURST_GAP_HRS,
    min_rain_burst: float = BIMODALITY_MIN_RAIN_BURST,
    min_separation_hours: float = BIMODALITY_SPLIT_MIN_SEPARATION_HRS,
    target_ratio: float = BIMODALITY_RAIN_TARGET_RATIO,
) -> dict:
    """Score whether the rain within an event window has two distinct bursts.

    :param rain: Rain series on the full dataset grid.
    :type rain: array-like
    :param event_start_idx: Inclusive event-window start index into *rain*.
    :type event_start_idx: int
    :param event_end_idx: Inclusive event-window end index into *rain*.
    :type event_end_idx: int
    :param dt_minutes: Time step in minutes.
    :type dt_minutes: float
    :param burst_gap_hours: Dry gap (hours) up to which wet runs merge into one burst.
    :type burst_gap_hours: float
    :param min_rain_burst: Minimum burst total to count.
    :type min_rain_burst: float
    :param min_separation_hours: Separation (hours) at which the two bursts score
        a full separation factor.
    :type min_separation_hours: float
    :param target_ratio: Second/largest volume ratio scoring a full magnitude.
    :type target_ratio: float
    :returns: ``rain_bimodality_index`` (float), ``rain_burst_count`` (int),
        ``rain_is_bimodal`` (bool).
    :rtype: dict
    """
    out = {"rain_bimodality_index": 0.0, "rain_burst_count": 0, "rain_is_bimodal": False}
    if rain is None:
        return out
    rain_arr = np.asarray(rain, dtype=float)
    n = len(rain_arr)
    if n == 0:
        return out
    s0 = int(max(0, min(event_start_idx, n - 1)))
    s1 = int(max(0, min(event_end_idx, n - 1)))
    if s1 < s0:
        s0, s1 = s1, s0
    seg = np.clip(np.nan_to_num(rain_arr[s0:s1 + 1], nan=0.0), 0.0, None)
    if seg.size == 0 or seg.sum() <= 0:
        return out

    steps_per_hour = max(1, int(round(60.0 / float(dt_minutes)))) if dt_minutes else 12
    gap_steps = max(1, int(round(float(burst_gap_hours) * steps_per_hour)))

    wet = np.flatnonzero(seg > 0)
    bursts = []
    if wet.size:
        b0 = last = int(wet[0])
        for w in wet[1:]:
            w = int(w)
            if w - last <= gap_steps:
                last = w
            else:
                bursts.append((b0, last))
                b0 = last = w
        bursts.append((b0, last))

    sig = [(a, b, float(seg[a:b + 1].sum())) for (a, b) in bursts]
    sig = [t for t in sig if t[2] >= float(min_rain_burst)]
    out["rain_burst_count"] = len(sig)
    if len(sig) < 2:
        return out

    sig.sort(key=lambda t: t[2], reverse=True)
    (a1, b1, t1), (a2, b2, t2) = sig[0], sig[1]
    rel = (t2 / t1) if t1 > 0 else 0.0
    mag_score = min(rel / max(float(target_ratio), 1e-9), 1.0)
    gap = (a2 - b1) if a2 >= a1 else (a1 - b2)
    gap_hours = max(0.0, float(gap)) / float(steps_per_hour)
    sep_factor = min(gap_hours / max(float(min_separation_hours), 1e-9), 1.0)
    index = float(mag_score * sep_factor)
    out["rain_bimodality_index"] = index
    out["rain_is_bimodal"] = index > 0.0
    return out


def _rain_linked(rain, peak_idx, steps_per_hour, lead_hours, lag_hours, min_rain) -> bool:
    """Return whether enough rain falls in ``[peak - lead, peak + lag]``."""
    if rain is None:
        return True
    rain_arr = np.asarray(rain, dtype=float)
    n = len(rain_arr)
    lo = max(0, peak_idx - int(round(lead_hours * steps_per_hour)))
    hi = min(n, peak_idx + int(round(lag_hours * steps_per_hour)) + 1)
    if lo >= hi:
        return False
    return float(np.nansum(rain_arr[lo:hi])) >= float(min_rain)


def calculate_peak_zone_bimodality(
    sg,
    flow,
    event_start_idx: int,
    primary_peak_idx: int,
    event_end_idx: int,
    *,
    rain=None,
    curv=None,
    dt_minutes: float = 5.0,
    score_threshold: float = BIMODALITY_FLAG_THRESHOLD,
    min_sec_curv_frac: float = BIMODALITY_MIN_SEC_CURV_FRAC,
    min_secondary_rel_height: float = BIMODALITY_MIN_SECONDARY_REL_HEIGHT,
    min_secondary_abs_rise: float = BIMODALITY_MIN_SECONDARY_ABS_RISE,
    min_valley_rel_dip: float = BIMODALITY_MIN_VALLEY_REL_DIP,
    min_valley_abs_dip: float = BIMODALITY_MIN_VALLEY_ABS_DIP,
    min_peak_distance: int = BIMODALITY_MIN_PEAK_DISTANCE,
    min_peak_width: int = BIMODALITY_MIN_PEAK_WIDTH,
    min_prominence_frac: float = BIMODALITY_MIN_PROM_FRAC,
    min_prominence_abs: float = BIMODALITY_MIN_PROM_ABS,
    rain_lead_hours: float = BIMODALITY_RAIN_LEAD_HRS,
    rain_lag_hours: float = BIMODALITY_RAIN_LAG_HRS,
    min_rain_near_peak: float = BIMODALITY_MIN_RAIN_NEAR_PEAK,
) -> dict:
    """Detect meaningful secondary rain-response crests within an event window.

    Candidate crests are found on the raw flow (default) with prominence, width,
    distance, curvature-sharpness and rain-link gates; those passing all gates are
    returned as accepted secondary peaks.

    :param sg: Smoothed flow on the full dataset grid.
    :type sg: array-like
    :param flow: Raw flow on the full dataset grid.
    :type flow: array-like
    :param event_start_idx: Inclusive event-window start index.
    :type event_start_idx: int
    :param primary_peak_idx: Index of the primary (largest) peak.
    :type primary_peak_idx: int
    :param event_end_idx: Inclusive event-window end index.
    :type event_end_idx: int
    :param rain: Optional rain series (enables the rain-link gate).
    :type rain: array-like | None
    :param curv: Optional precomputed smoothed-flow curvature.
    :type curv: array-like | None
    :param dt_minutes: Time step in minutes.
    :type dt_minutes: float
    :returns: ``bimodality_index`` (float), ``is_bimodal`` (bool),
        ``secondary_peak_list`` (accepted extra peaks, global indices),
        ``peak_zone_peak_list`` (primary + accepted, sorted),
        ``rain_bimodality_index`` (float), ``bimodality_threshold`` (float).
    :rtype: dict
    """
    sg = np.asarray(sg, dtype=float)
    flow = np.asarray(flow, dtype=float)
    n = len(sg)
    rain_bi = calculate_rain_bimodality(rain, event_start_idx, event_end_idx, dt_minutes=dt_minutes)
    base_out = {
        "bimodality_index": 0.0,
        "is_bimodal": False,
        "secondary_peak_list": [],
        "peak_zone_peak_list": [int(primary_peak_idx)] if n else [],
        "rain_bimodality_index": rain_bi["rain_bimodality_index"],
        "bimodality_threshold": float(score_threshold),
    }
    if n == 0:
        base_out["peak_zone_peak_list"] = []
        return base_out

    s0 = int(max(0, min(event_start_idx, n - 1)))
    s1 = int(max(0, min(event_end_idx, n - 1)))
    if s1 < s0:
        s0, s1 = s1, s0
    p0 = int(max(s0, min(primary_peak_idx, s1)))
    if s1 - s0 < 2 * int(min_peak_distance):
        return base_out

    steps_per_hour = max(1, int(round(60.0 / float(dt_minutes)))) if dt_minutes else 12

    if curv is None or len(np.asarray(curv)) != n:
        curv = savgol_curvature(sg)
    else:
        curv = np.asarray(curv, dtype=float)
    curv_abs_event = np.abs(curv[s0:s1 + 1])
    curv_scale = float(np.nanpercentile(curv_abs_event, 95)) if curv_abs_event.size else 0.0

    scan_src = flow if BIMODALITY_USE_RAW_FLOW else sg
    scan = scan_src[s0:s1 + 1]
    baseline = float(np.nanpercentile(scan, 10))
    scale = max(float(np.nanmax(scan)) - baseline, 1e-9)
    prom_gate = max(min_prominence_frac * scale, min_prominence_abs)

    idx_local, _ = find_peaks(scan, distance=int(min_peak_distance), width=int(min_peak_width), prominence=prom_gate)
    if idx_local.size == 0:
        return base_out
    proms = peak_prominences(scan, idx_local)[0]
    globals_ = idx_local + s0

    primary_height = max(float(scan_src[p0]) - baseline, 1e-9)
    accepted: list[int] = []
    best_score = 0.0
    for loc, gidx, prom in zip(idx_local, globals_, proms):
        if abs(int(gidx) - p0) < int(min_peak_distance):
            continue  # this is (or is adjacent to) the primary
        sec_height = float(scan_src[gidx]) - baseline
        if sec_height < max(min_secondary_rel_height * scale, min_secondary_abs_rise):
            continue
        # valley between primary and this secondary
        a, b = sorted((p0, int(gidx)))
        valley = float(np.min(scan_src[a:b + 1]))
        smaller_crest = min(float(scan_src[p0]), float(scan_src[gidx]))
        valley_dip = smaller_crest - valley
        if valley_dip < max(min_valley_rel_dip * scale, min_valley_abs_dip):
            continue
        # curvature sharpness (concave-down => negative second derivative)
        if curv_scale > 0 and abs(float(curv[gidx])) < min_sec_curv_frac * curv_scale:
            continue
        # rain link
        if not _rain_linked(rain, int(gidx), steps_per_hour, rain_lead_hours, rain_lag_hours, min_rain_near_peak):
            continue
        accepted.append(int(gidx))
        height_ratio = min(sec_height / primary_height, 1.0)
        dip_frac = min(valley_dip / max(smaller_crest - baseline, 1e-9), 1.0)
        best_score = max(best_score, height_ratio * dip_frac)

    if not accepted:
        return base_out

    flow_index = float(best_score)
    combined = (1.0 - BIMODALITY_RAIN_INDEX_WEIGHT) * flow_index + BIMODALITY_RAIN_INDEX_WEIGHT * rain_bi["rain_bimodality_index"]
    base_out.update({
        "bimodality_index": combined,
        "is_bimodal": combined >= float(score_threshold),
        "secondary_peak_list": accepted,
        "peak_zone_peak_list": sorted({p0, *accepted}),
    })
    return base_out


__all__ = [
    "calculate_rain_bimodality",
    "calculate_peak_zone_bimodality",
    "BIMODALITY_FLAG_THRESHOLD",
    "BIMODALITY_SPLIT_MIN_SEPARATION_HRS",
    "BIMODALITY_SPLIT_MAX_VALLEY_FRAC",
]
