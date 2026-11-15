"""Spin-up warmup and train / validation splits for continuous-simulation calibration.

A store-based model (IA store ``S``, deep store ``G``, RTK convolution tails) starts
from empty, so its first weeks are wrong because of the *initial condition*, not
the parameters.  Those steps must be simulated but never scored.  This module
builds the boolean masks that choose which steps enter an objective:

* :func:`prepend_spinup` manufactures a synthetic pre-history (a copy of the
  record relabelled to sit before the real start), so no *real* record is
  spent on warmup.  Only forcing is invented; the copied rows are always
  masked out of every score.
* :func:`build_masks` returns ``warmup`` / ``train`` / ``val`` masks, either a
  ``continuous`` split (first ``train_frac`` trains, tail validates) or a
  ``stratified`` seasonal holdout (every ``val_every``-th block of
  ``block_days`` is validation, so validation spans every season).  Block
  edges can be snapped into dry periods (:func:`dry_edge_flags`) so no storm
  is cut between train and validation.
* :func:`label_events_by_split` assigns each detected storm to warmup / train /
  val by the majority of its scored window and counts edge-straddlers.

The model is still simulated in **one continuous pass**; the masks only select
steps, so store state carries across every block edge and no buffer between
blocks is needed.

All knobs live on :class:`SplitConfig`, so nothing reads notebook globals.
Every step count is derived from days / hours and ``dt_hours``, so the same
configuration works at any model timestep.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, NamedTuple

import numpy as np
import pandas as pd

SPINUP_MODES = ("wrap", "repeat_head", "none")
SPLIT_MODES = ("continuous", "stratified")
SPLIT_LABELS = ("warmup", "train", "val")

_DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class SplitConfig:
    """Knobs for the spin-up warmup and the train / validation split.

    :param dt_hours: Model timestep [h].  Every step count is derived from it.
    :type dt_hours: float
    :param warmup_days: Unscored warmup length [d].  A multiple of 7 keeps both
        hour-of-day and day-of-week aligned in the copied block.
    :type warmup_days: float
    :param spinup_mode: ``"wrap"`` prepends the record's last ``warmup_days``;
        ``"repeat_head"`` prepends its first ``warmup_days`` (classic cyclic
        spin-up); ``"none"`` spends the first ``warmup_days`` of real record.
    :type spinup_mode: str
    :param split_mode: ``"continuous"`` or ``"stratified"``.
    :type split_mode: str
    :param train_frac: Continuous mode: training fraction of the post-warmup window.
    :type train_frac: float
    :param block_days: Stratified mode: interleave block length [d].
    :type block_days: float
    :param val_every: Stratified mode: every ``val_every``-th block is validation.
    :type val_every: int
    :param snap_edges_to_dry: Stratified mode: move block edges into dry periods.
    :type snap_edges_to_dry: bool
    :param edge_dry_hours: An edge is "clean" with no rain in the preceding window [h].
    :type edge_dry_hours: float
    :param edge_snap_max_days: Furthest an edge may move [d]; beyond it the
        calendar edge is kept.
    :type edge_snap_max_days: float
    :param rain_eps: Per-step depth above which a step counts as wet.
    :type rain_eps: float
    :raises ValueError: On an unknown mode or an out-of-range knob.
    """

    dt_hours: float
    warmup_days: float = 91.0
    spinup_mode: str = "wrap"
    split_mode: str = "stratified"
    train_frac: float = 0.70
    block_days: float = 21.0
    val_every: int = 4
    snap_edges_to_dry: bool = True
    edge_dry_hours: float = 48.0
    edge_snap_max_days: float = 5.0
    rain_eps: float = 1e-3

    def __post_init__(self) -> None:
        if self.spinup_mode not in SPINUP_MODES:
            raise ValueError(f"spinup_mode must be one of {SPINUP_MODES}, got {self.spinup_mode!r}")
        if self.split_mode not in SPLIT_MODES:
            raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {self.split_mode!r}")
        if not self.dt_hours > 0:
            raise ValueError(f"dt_hours must be > 0, got {self.dt_hours!r}")
        if self.warmup_days < 0:
            raise ValueError(f"warmup_days must be >= 0, got {self.warmup_days!r}")
        if not 0.0 < self.train_frac <= 1.0:
            raise ValueError(f"train_frac must be in (0, 1], got {self.train_frac!r}")
        if not self.block_days > 0:
            raise ValueError(f"block_days must be > 0, got {self.block_days!r}")
        if int(self.val_every) != self.val_every or self.val_every < 2:
            raise ValueError(f"val_every must be an integer >= 2, got {self.val_every!r}")

    def steps(self, days: float = 0.0, hours: float = 0.0) -> int:
        """Convert a duration to a whole number of model steps (rounded).

        :param days: Duration in days.
        :type days: float
        :param hours: Additional duration in hours.
        :type hours: float
        :returns: ``round((24*days + hours) / dt_hours)``.
        :rtype: int
        """
        return int(round((days * 24.0 + hours) / self.dt_hours))

    @property
    def warmup_steps(self) -> int:
        """Warmup length in model steps."""
        return self.steps(days=self.warmup_days)

    def describe(self) -> str:
        """One-line summary of the configuration (for notebook prints).

        :rtype: str
        """
        head = (f"SPLIT_MODE={self.split_mode}   WARMUP_DAYS={self.warmup_days:g}  "
                f"SPINUP_MODE={self.spinup_mode}")
        if self.split_mode == "continuous":
            return head + f"   TRAIN_FRAC={self.train_frac:.0%} (auto TRAIN_END)"
        return (head + f"   block={self.block_days:g}d, 1-in-{self.val_every} -> val"
                + (", dry-snapped" if self.snap_edges_to_dry else ""))


class SplitMasks(NamedTuple):
    """Boolean step masks returned by :func:`build_masks`.

    Unpacks as ``warmup, train, val, warmup_steps``.  ``train`` and ``val``
    both exclude the warmup and never overlap; together with ``warmup`` they
    partition the record.
    """

    warmup: np.ndarray
    train: np.ndarray
    val: np.ndarray
    warmup_steps: int

    @property
    def scored(self) -> np.ndarray:
        """Every non-warmup step (``train | val``)."""
        return self.train | self.val


def wrap_seasonal_slip_days(record_days: float) -> float:
    """Seasonal misplacement of a ``wrap`` spin-up block [d].

    ``wrap`` shifts the record's tail back by the record *length*, so a copied
    step lands ``record_days`` earlier than its real date.  Relative to the
    same calendar day the slip is the signed distance of ``record_days`` to the
    nearest whole number of years: ``+0`` for a 1- or 4-year record, ``-65``
    for a 300-day record (the copy sits 65 days early in the season).

    :param record_days: Length of the real record [d].
    :type record_days: float
    :returns: Signed slip in days, in ``(-182.6, +182.6]``.
    :rtype: float
    """
    years = round(record_days / _DAYS_PER_YEAR)
    return float(record_days - years * _DAYS_PER_YEAR)


def prepend_spinup(
    df: pd.DataFrame,
    cfg: SplitConfig,
    datetime_col: str = "datetime",
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Prepend a synthetic spin-up block so no real record is spent on warmup.

    A block of ``cfg.warmup_days`` is copied from the record and its timestamps
    are relabelled to end exactly one step before the real start.  Every other
    column (flow, rain, temperature, ...) is copied as-is, so the prepended rows
    must be masked out of every score (:func:`build_masks` does this when given
    the returned step count).

    ``wrap`` copies the record's tail; its seasonal slip is
    :func:`wrap_seasonal_slip_days` of the record length, which matters because
    temperature drives the IA recovery rate.  ``repeat_head`` copies the head
    and is always available; prefer it when the record is shorter than about
    twice the warmup.

    :param df: Model frame on a regular grid, oldest first.
    :type df: pandas.DataFrame
    :param cfg: Split configuration.
    :type cfg: SplitConfig
    :param datetime_col: Timestamp column.
    :type datetime_col: str
    :param verbose: Print what was prepended.
    :type verbose: bool
    :returns: ``(df_extended, n_spinup_steps)``.  ``n_spinup_steps`` is 0 for
        ``spinup_mode="none"``, a zero warmup, or a warmup at least as long as
        the record (which also warns).
    :rtype: tuple[pandas.DataFrame, int]
    """
    d = df.reset_index(drop=True).copy()
    if cfg.spinup_mode == "none" or cfg.warmup_days <= 0:
        return d, 0

    dt = pd.to_datetime(d[datetime_col])
    step = dt.diff().median()
    n_spin = cfg.warmup_steps
    if n_spin >= len(d):
        warnings.warn(f"spin-up ({n_spin:,} steps) >= record ({len(d):,}); "
                      "falling back to spinup_mode='none'.", stacklevel=2)
        return d, 0

    src = (d.iloc[-n_spin:] if cfg.spinup_mode == "wrap" else d.iloc[:n_spin]).copy()
    # Relabel so the block ends exactly one step before the real record starts.
    src[datetime_col] = pd.date_range(end=dt.iloc[0] - step, periods=n_spin, freq=step)
    out = pd.concat([src, d], ignore_index=True)

    if verbose:
        print(f"Spin-up: prepended {n_spin:,} synthetic steps ({cfg.warmup_days:g} d, "
              f"mode={cfg.spinup_mode}) covering {out[datetime_col].iloc[0]} -> "
              f"{out[datetime_col].iloc[n_spin - 1]}")
        msg = f"         all {len(d):,} real steps are now scorable"
        if cfg.spinup_mode == "wrap":
            record_days = len(d) * cfg.dt_hours / 24.0
            msg += (f"; seasonal slip of the wrap = {wrap_seasonal_slip_days(record_days):+.0f} d "
                    f"(record {record_days:,.0f} d = {record_days / _DAYS_PER_YEAR:.2f} yr)")
        print(msg)
    return out, n_spin


def dry_edge_flags(rain: Any, cfg: SplitConfig) -> np.ndarray:
    """True where a block edge may be placed: no rain in the preceding window.

    An edge is "clean" once nothing above ``cfg.rain_eps`` has fallen for
    ``cfg.edge_dry_hours``, so the previous storm's response has decayed.  The
    test looks backward only: an edge just *before* a storm is ideal, since the
    storm then falls wholly inside the next block.

    :param rain: Per-step rainfall depth.
    :type rain: array-like
    :param cfg: Split configuration.
    :type cfg: SplitConfig
    :returns: Boolean array, same length as *rain*.
    :rtype: numpy.ndarray
    """
    wet = pd.Series((np.asarray(rain, dtype=float) > cfg.rain_eps).astype(float))
    look = max(1, cfg.steps(hours=cfg.edge_dry_hours))
    return wet.rolling(look, min_periods=1).max().to_numpy() == 0.0


def _snap_edges(edges: list[int], dry: np.ndarray, n: int, warmup_steps: int,
                block_steps: int, cfg: SplitConfig) -> tuple[list[int], int]:
    """Move each calendar edge to the nearest dry step within ``edge_snap_max_days``."""
    snap = cfg.steps(days=cfg.edge_snap_max_days)
    min_block = max(1, block_steps // 3)
    snapped, prev, n_moved = [], warmup_steps - min_block, 0
    for e in edges:
        lo, hi = max(e - snap, 0), min(e + snap + 1, n)
        cand = np.flatnonzero(dry[lo:hi]) + lo
        cand = cand[cand > prev + min_block]   # keep blocks ordered and sane
        best = int(cand[np.argmin(np.abs(cand - e))]) if cand.size else e
        n_moved += int(bool(cand.size) and best != e)
        if best > prev:
            snapped.append(best)
            prev = best
    return snapped, n_moved


def build_masks(
    datetimes: Any,
    cfg: SplitConfig,
    *,
    rain: Any = None,
    warmup_steps: int | None = None,
    verbose: bool = True,
) -> SplitMasks:
    """Warmup, train and validation masks over a regular-grid record.

    ``continuous``: the first ``cfg.train_frac`` of the post-warmup window
    trains and the contiguous tail validates.

    ``stratified``: the post-warmup window is cut into ``cfg.block_days``
    blocks and every ``cfg.val_every``-th block is validation, so validation
    spans all seasons.  When ``cfg.snap_edges_to_dry`` is set and *rain* is
    given, each edge moves to the nearest dry step (:func:`dry_edge_flags`), so
    blocks come out slightly uneven but never cut a storm in half.

    :param datetimes: Timestamps of the (possibly spin-up-extended) record;
        only its length is used.
    :type datetimes: array-like
    :param cfg: Split configuration.
    :type cfg: SplitConfig
    :param rain: Per-step rainfall, needed for dry-snapped edges.
    :type rain: array-like or None
    :param warmup_steps: Leading unscored steps.  Pass the step count returned
        by :func:`prepend_spinup`; ``None`` uses ``cfg.warmup_steps`` (real
        record spent on warmup).
    :type warmup_steps: int or None
    :param verbose: Print the snap count and the validation share.
    :type verbose: bool
    :returns: :class:`SplitMasks` (unpacks as ``warmup, train, val, warmup_steps``).
    :rtype: SplitMasks
    """
    n = len(datetimes)
    if warmup_steps is None:
        warmup_steps = cfg.warmup_steps
    warmup_steps = int(min(max(warmup_steps, 0), n))
    if warmup_steps > n // 3:
        warnings.warn(f"warmup is {warmup_steps:,} steps ({warmup_steps * cfg.dt_hours / 24.0:.0f} d "
                      f"@ {cfg.dt_hours:.4g} h), over 1/3 of the {n:,}-step record; only "
                      f"{n - warmup_steps:,} steps remain for train+val.", stacklevel=2)
    warmup = np.zeros(n, dtype=bool)
    warmup[:warmup_steps] = True

    if cfg.split_mode == "continuous":
        # Regular grid, so the fraction is taken by step count.
        n_train = int(round(cfg.train_frac * (n - warmup_steps)))
        val = np.zeros(n, dtype=bool)
        val[warmup_steps + n_train:] = True
    else:
        block_steps = max(1, cfg.steps(days=cfg.block_days))
        edges = list(range(warmup_steps, n, block_steps))
        if cfg.snap_edges_to_dry and rain is not None and edges:
            rain = np.asarray(rain, dtype=float)
            if len(rain) != n:
                raise ValueError(f"rain has {len(rain):,} steps, datetimes {n:,}")
            snapped, n_moved = _snap_edges(edges, dry_edge_flags(rain, cfg), n,
                                           warmup_steps, block_steps, cfg)
            if snapped:
                edges = snapped
                if verbose:
                    print(f"Dry-snapped {n_moved}/{len(edges)} block edges "
                          f"(<= {cfg.edge_snap_max_days:g} d move, "
                          f"{cfg.edge_dry_hours:g} h dry lookback)")
        if edges:
            block = np.searchsorted(np.asarray(edges), np.arange(n), side="right") - 1
            val = (block % cfg.val_every) == (cfg.val_every - 1)
        else:
            val = np.zeros(n, dtype=bool)

    val = val & ~warmup
    train = ~val & ~warmup
    if verbose and cfg.split_mode == "stratified":
        scored = max(1, int(train.sum() + val.sum()))
        print(f"Validation share of scored steps: {val.sum() / scored:.1%} "
              f"(nominal 1-in-{cfg.val_every} = {1 / cfg.val_every:.0%})")
    return SplitMasks(warmup, train, val, warmup_steps)


def _event_bounds(event: Any) -> tuple[int, int]:
    """``(start_idx, end_idx)`` from an event object or a ``(start, end)`` pair."""
    if hasattr(event, "start_idx"):
        return int(event.start_idx), int(event.end_idx)
    start, end = event
    return int(start), int(end)


def label_events_by_split(
    events: Iterable[Any],
    warmup: Any,
    train: Any,
    val: Any,
    *,
    pre_steps: int = 0,
    tail_steps: int = 0,
) -> tuple[np.ndarray, int]:
    """Assign each storm to warmup / train / val and count edge-straddlers.

    A storm's window is ``[start_idx - pre_steps, end_idx + tail_steps]``
    (clipped to the record).  Its label is the mask covering the largest share
    of that window (ties go to the earlier label in :data:`SPLIT_LABELS`).  A
    storm *straddles* when its window holds both train and val steps; with
    dry-snapped edges this should be rare.

    :param events: Events with ``start_idx`` / ``end_idx`` attributes (e.g.
        MSDGC ``find_events`` output) or ``(start, end)`` index pairs.
    :type events: iterable
    :param warmup: Warmup mask.
    :type warmup: array-like of bool
    :param train: Train mask.
    :type train: array-like of bool
    :param val: Validation mask.
    :type val: array-like of bool
    :param pre_steps: Steps added before each event start.
    :type pre_steps: int
    :param tail_steps: Recession steps added after each event end.
    :type tail_steps: int
    :returns: ``(labels, n_straddle)`` where *labels* is an object array of
        ``"warmup"`` / ``"train"`` / ``"val"``, one per event.
    :rtype: tuple[numpy.ndarray, int]
    """
    masks = [np.asarray(m, dtype=bool) for m in (warmup, train, val)]
    n = len(masks[1])
    labels, n_straddle = [], 0
    for ev in events:
        start, end = _event_bounds(ev)
        sl = slice(max(0, start - pre_steps), min(n - 1, end + tail_steps) + 1)
        frac = [m[sl].mean() if m[sl].size else 0.0 for m in masks]
        labels.append(SPLIT_LABELS[int(np.argmax(frac))])
        n_straddle += int(masks[1][sl].any() and masks[2][sl].any())
    return np.array(labels, dtype=object), n_straddle


__all__ = [
    "SPINUP_MODES",
    "SPLIT_MODES",
    "SPLIT_LABELS",
    "SplitConfig",
    "SplitMasks",
    "wrap_seasonal_slip_days",
    "prepend_spinup",
    "dry_edge_flags",
    "build_masks",
    "label_events_by_split",
]
