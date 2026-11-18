"""Event-based, stratified, interwoven train/validation partitioning.

The entry point is :func:`stratified_event_split`, which consumes a per-timestep
``event_ids`` array (as produced by the notebook's ``events_to_event_ids`` /
:func:`sparsehydro.events.events_to_mask` machinery) and returns an
:class:`EventSplit` holding the per-event assignment and every boolean mask the
calibration and scoring steps need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["EventSplit", "stratified_event_split", "estimate_recession_hours"]


# ---------------------------------------------------------------------------
# Recession estimate (drives the "auto" warmup length)
# ---------------------------------------------------------------------------

def estimate_recession_hours(
    flow,
    baseflow,
    *,
    dt_hours: float = 1.0,
    min_excess_frac: float = 0.05,
) -> float:
    """Estimate the storm-recession time constant :math:`\\tau` in hours.

    A continuous rainfall-runoff model carries state (soil moisture, base-flow
    storage) between storms, so the warmup lead-in ahead of a validation event
    should scale with how long that memory persists.  This estimates
    :math:`\\tau` from the median step-to-step decay ratio on **falling limbs**
    of the base-flow-removed hydrograph, assuming ``excess[t+1] ≈ excess[t]·e^{-Δt/τ}``.

    :param flow: Total observed flow series.
    :param baseflow: Base-flow series (same length) subtracted before analysis.
    :param dt_hours: Timestep length in hours.
    :type dt_hours: float
    :param min_excess_frac: Only use falling-limb steps whose starting excess
        exceeds ``min_excess_frac × max(excess)`` — ignores near-baseflow noise.
    :type min_excess_frac: float
    :returns: Estimated recession time constant in hours; ``nan`` if it cannot be
        estimated (e.g. no clean falling limbs).
    :rtype: float
    """
    excess = np.clip(np.asarray(flow, float) - np.asarray(baseflow, float), 0.0, None)
    if excess.size < 3 or not np.isfinite(excess).any():
        return float("nan")
    peak = float(np.nanmax(excess))
    if peak <= 0:
        return float("nan")
    cur, nxt = excess[:-1], excess[1:]
    falling = (nxt < cur) & (cur > min_excess_frac * peak) & (nxt > 0)
    if not falling.any():
        return float("nan")
    ratios = nxt[falling] / cur[falling]
    ratios = ratios[(ratios > 0) & (ratios < 1)]
    if ratios.size == 0:
        return float("nan")
    r = float(np.median(ratios))
    tau = -dt_hours / np.log(r)
    return float(tau) if np.isfinite(tau) and tau > 0 else float("nan")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EventSplit:
    """An event-based train/validation partition and its boolean masks.

    All masks are 1-D boolean arrays aligned to the original timeline (same
    length as ``event_ids``).  The four *exclusion* regions
    (``val_mask``, ``warmup_mask``, ``buffer_mask``) and ``train_mask`` form a
    partition of the timeline: every timestep is in exactly one of them.

    :ivar event_ids: The per-timestep event label array this split was built
        from (``-1`` between storms).
    :ivar train_events: Sorted event indices assigned to training.
    :ivar val_events: Sorted event indices assigned to validation.
    :ivar val_mask: Timesteps inside a validation storm window (the **scored**
        validation region; used by per-event scoring for all methods).
    :ivar warmup_mask: Unscored spin-up steps immediately *before* each
        validation storm — excluded from training so no fitted step seeds a
        validation storm's initial state.
    :ivar buffer_mask: Dynamic recession guard *after* each validation storm —
        excluded from training so a following training storm does not begin
        inside a validation storm's recession.
    :ivar train_mask: Whole-series training steps for a Method-A-style fit
        (everything not held out: ``~(val | warmup | buffer)``).
    :ivar train_event_mask: Training-storm windows only (``train_mask`` ∩ train
        events) — the calibration mask for event-based methods (B/C).
    :ivar events_table: One row per event: ``start``, ``magnitude`` (storm
        volume), ``mag_bin``, ``season_bin``, ``split``.
    :ivar meta: Parameters and summary counts for reproducibility/reporting.
    """

    event_ids: np.ndarray
    train_events: np.ndarray
    val_events: np.ndarray
    val_mask: np.ndarray
    warmup_mask: np.ndarray
    buffer_mask: np.ndarray
    train_mask: np.ndarray
    train_event_mask: np.ndarray
    events_table: pd.DataFrame
    meta: dict = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def val_event_mask(self) -> np.ndarray:
        """Alias for :attr:`val_mask` (validation-storm windows are the scored set)."""
        return self.val_mask

    def counts(self) -> dict:
        """Return a small dict of event/step counts for logging."""
        return {
            "n_train_events": int(self.train_events.size),
            "n_val_events": int(self.val_events.size),
            "train_event_steps": int(self.train_event_mask.sum()),
            "val_event_steps": int(self.val_mask.sum()),
            "warmup_steps": int(self.warmup_mask.sum()),
            "buffer_steps": int(self.buffer_mask.sum()),
            "train_steps_A": int(self.train_mask.sum()),
        }


# ---------------------------------------------------------------------------
# Core split
# ---------------------------------------------------------------------------

def _event_bounds(event_ids: np.ndarray) -> dict[int, tuple[int, int]]:
    """Map each event index to its ``(first_idx, last_idx)`` span."""
    bounds: dict[int, tuple[int, int]] = {}
    for k in np.unique(event_ids[event_ids >= 0]):
        pos = np.flatnonzero(event_ids == k)
        bounds[int(k)] = (int(pos[0]), int(pos[-1]))
    return bounds


def _assign_interwoven(
    order: list[int],
    strata: dict[int, tuple],
    val_fraction: float,
    seed: int,
) -> set[int]:
    """Stratified, time-interwoven selection of validation events.

    Within each stratum, events (already in time order) are picked with an
    accumulator that fires every ``1/val_fraction`` steps, so validation events
    are spread evenly across the record rather than clustered.  A seeded phase
    per stratum decorrelates which events are chosen without breaking the even
    spacing.
    """
    rng = np.random.default_rng(seed)
    by_stratum: dict[tuple, list[int]] = {}
    for k in order:  # order is time-sorted, so groups stay time-sorted
        by_stratum.setdefault(strata[k], []).append(k)

    val: set[int] = set()
    for _stratum, members in sorted(by_stratum.items(), key=lambda kv: str(kv[0])):
        acc = float(rng.random())
        for k in members:
            acc += val_fraction
            if acc >= 1.0:
                acc -= 1.0
                val.add(k)
    return val


def stratified_event_split(
    event_ids,
    datetime,
    flow,
    baseflow,
    *,
    val_fraction: float = 0.30,
    n_mag_bins: int = 3,
    n_season_bins: int = 4,
    warmup_hours="auto",
    warmup_factor: float = 3.0,
    buffer_tol: float = 0.10,
    max_buffer_factor: float = 3.0,
    dt_hours: float = 1.0,
    seed: int = 42,
) -> EventSplit:
    """Partition storms into interwoven train/validation sets with guard bands.

    :param event_ids: Per-timestep event label array (``-1`` outside events),
        e.g. from the notebook's ``events_to_event_ids``.  Event indices need not
        be contiguous; each present index is treated as one storm.
    :param datetime: Timestamps aligned to *event_ids* (array-like coercible by
        :func:`pandas.to_datetime`).  Used for season strata and reporting.
    :param flow: Total observed flow series (for storm-magnitude strata and the
        recession-based warmup / buffer).
    :param baseflow: Base-flow series subtracted to get storm excess.
    :param val_fraction: Target fraction of events assigned to validation.
    :type val_fraction: float
    :param n_mag_bins: Number of storm-magnitude (volume) quantile strata.
    :type n_mag_bins: int
    :param n_season_bins: Number of season strata (12 months split evenly; 4 →
        meteorological quarters).
    :type n_season_bins: int
    :param warmup_hours: Unscored spin-up length before each validation storm.
        ``"auto"`` derives it as ``warmup_factor × estimate_recession_hours``.
    :param warmup_factor: Multiplier on the recession constant when
        ``warmup_hours="auto"`` (3 ≈ 95 % state decay).
    :type warmup_factor: float
    :param buffer_tol: Recession buffer extends after a validation storm until
        storm excess falls below ``buffer_tol × baseflow``.
    :type buffer_tol: float
    :param max_buffer_factor: Cap the recession buffer at
        ``max_buffer_factor × warmup_steps`` steps.
    :type max_buffer_factor: float
    :param dt_hours: Timestep length in hours.
    :type dt_hours: float
    :param seed: Seed for the (reproducible) stratified assignment.
    :type seed: int
    :returns: The :class:`EventSplit` partition.
    :rtype: EventSplit
    :raises ValueError: If *event_ids* contains no events.
    """
    event_ids = np.asarray(event_ids)
    flow = np.asarray(flow, float)
    baseflow = np.asarray(baseflow, float)
    dt = pd.to_datetime(pd.Series(datetime)).reset_index(drop=True)
    n = event_ids.shape[0]
    if not (flow.shape[0] == baseflow.shape[0] == len(dt) == n):
        raise ValueError("event_ids, datetime, flow, baseflow must be equal length.")

    bounds = _event_bounds(event_ids)
    if not bounds:
        raise ValueError("event_ids contains no events (all -1).")
    events = sorted(bounds)  # event indices, ascending == time order (disjoint spans)

    excess = np.clip(flow - baseflow, 0.0, None)

    # Per-event magnitude (storm volume) and season (start month).
    magnitude = {k: float(excess[a : b + 1].sum() * dt_hours) for k, (a, b) in bounds.items()}
    start_ts = {k: dt.iloc[a] for k, (a, b) in bounds.items()}
    order = sorted(events, key=lambda k: start_ts[k])  # explicit time order

    # Strata: magnitude quantile bin × season bin.
    mags = pd.Series({k: magnitude[k] for k in events})
    try:
        mag_bin = pd.qcut(mags.rank(method="first"), q=min(n_mag_bins, mags.nunique()),
                          labels=False, duplicates="drop")
    except (ValueError, IndexError):
        mag_bin = pd.Series(0, index=mags.index)
    mag_bin = mag_bin.fillna(0).astype(int).to_dict()

    span = max(1, 12 // max(1, n_season_bins))
    season_bin = {k: (start_ts[k].month - 1) // span for k in events}
    strata = {k: (mag_bin[k], season_bin[k]) for k in events}

    val_events = _assign_interwoven(order, strata, val_fraction, seed)
    # Guard against an empty validation set on small catalogs.
    if not val_events and len(order) >= 2:
        val_events = {order[len(order) // 2]}
    train_events = [k for k in events if k not in val_events]

    # Warmup length in steps.
    if warmup_hours == "auto":
        tau = estimate_recession_hours(flow, baseflow, dt_hours=dt_hours)
        wh = warmup_factor * tau if np.isfinite(tau) else 3 * 24.0
    else:
        wh = float(warmup_hours)
    warmup_steps = max(1, int(round(wh / dt_hours)))
    max_buffer_steps = max(1, int(round(max_buffer_factor * warmup_steps)))

    # Base masks.
    val_arr = np.array(sorted(val_events), dtype=int)
    train_arr = np.array(sorted(train_events), dtype=int)
    val_mask = np.isin(event_ids, val_arr) if val_arr.size else np.zeros(n, bool)

    warmup_mask = np.zeros(n, bool)
    buffer_mask = np.zeros(n, bool)
    thresh = buffer_tol * baseflow

    for k in val_arr:
        a, b = bounds[int(k)]
        # Warmup: fixed spin-up immediately before the storm window.
        w0 = max(0, a - warmup_steps)
        warmup_mask[w0:a] = True
        # Buffer: dynamic recession guard after the storm, until near baseflow.
        i, steps = b + 1, 0
        while i < n and steps < max_buffer_steps and excess[i] > thresh[i]:
            if val_mask[i]:  # ran into the next validation storm
                break
            buffer_mask[i] = True
            i += 1
            steps += 1

    # A validation window always wins over warmup/buffer bookkeeping.
    warmup_mask &= ~val_mask
    buffer_mask &= ~val_mask & ~warmup_mask

    excluded = val_mask | warmup_mask | buffer_mask
    train_mask = ~excluded                                   # Method A whole-series fit
    train_event_mask = np.isin(event_ids, train_arr) & ~excluded  # B/C calibration mask

    events_table = pd.DataFrame(
        {
            "start": [start_ts[k] for k in order],
            "magnitude": [magnitude[k] for k in order],
            "mag_bin": [mag_bin[k] for k in order],
            "season_bin": [season_bin[k] for k in order],
            "split": ["val" if k in val_events else "train" for k in order],
        },
        index=pd.Index(order, name="event_id"),
    )

    meta = {
        "val_fraction_target": val_fraction,
        "val_fraction_actual": (val_arr.size / len(events)) if events else 0.0,
        "n_events": len(events),
        "n_mag_bins": n_mag_bins,
        "n_season_bins": n_season_bins,
        "warmup_hours": warmup_steps * dt_hours,
        "warmup_steps": warmup_steps,
        "recession_hours": estimate_recession_hours(flow, baseflow, dt_hours=dt_hours),
        "buffer_tol": buffer_tol,
        "max_buffer_steps": max_buffer_steps,
        "dt_hours": dt_hours,
        "seed": seed,
    }

    return EventSplit(
        event_ids=event_ids,
        train_events=train_arr,
        val_events=val_arr,
        val_mask=val_mask,
        warmup_mask=warmup_mask,
        buffer_mask=buffer_mask,
        train_mask=train_mask,
        train_event_mask=train_event_mask,
        events_table=events_table,
        meta=meta,
    )
