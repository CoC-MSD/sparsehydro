"""Per-event (event-averaged) skill metrics.

The notebook's existing event columns (``*_val_evt``) apply a point-wise KGE/CCC
over the **concatenation** of every validation-event timestep.  Across
discontiguous storms that pools between-storm level differences into the
correlation/variance terms and lets the largest storm dominate — so it is not a
faithful "typical event" indicator.

:func:`per_event_scores` instead evaluates each objective **within one storm at a
time**, and :func:`summarize_event_scores` averages across storms (reporting the
median and IQR too, since per-event KGE is noisy for short storms).  Objective
definitions are reused from :mod:`sparsehydro.calibration`, so the numbers stay
consistent with the rest of the notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparsehydro.calibration import KGE, ConcordanceCorrelationCoefficient

__all__ = ["per_event_scores", "summarize_event_scores", "DEFAULT_OBJECTIVES"]


def DEFAULT_OBJECTIVES() -> dict:
    """Return a fresh ``{"kge": KGE(), "ccc": CCC()}`` mapping."""
    return {"kge": KGE(), "ccc": ConcordanceCorrelationCoefficient()}


def per_event_scores(
    observed,
    predicted,
    event_ids,
    *,
    objectives: dict | None = None,
    restrict_to=None,
    datetime=None,
    min_steps: int = 3,
) -> pd.DataFrame:
    """Score every storm event independently — one row per event.

    :param observed: Observed series (e.g. ``flow_cfs``).
    :param predicted: Modeled series (e.g. ``q_total_cfs``), same length.
    :param event_ids: Per-timestep event label array (``-1`` outside events).
    :param objectives: Mapping ``name -> objective`` with a
        ``compute(obs, pred, mask=...)`` method.  Defaults to KGE + CCC.
    :param restrict_to: Optional iterable of event indices to score (e.g. the
        validation events); ``None`` scores every event present.
    :param datetime: Optional timestamps (adds a ``start`` column for context).
    :param min_steps: Events with fewer than this many finite paired steps are
        skipped (returns ``nan`` metrics — too short to score meaningfully).
    :type min_steps: int
    :returns: DataFrame indexed by event id with columns ``n_steps`` (+ ``start``
        if *datetime* given) and one column per objective.  A degenerate window
        (e.g. constant observed → KGE undefined) yields ``nan`` for that metric
        rather than raising.
    :rtype: pandas.DataFrame
    """
    obs = np.asarray(observed, float)
    pred = np.asarray(predicted, float)
    ids = np.asarray(event_ids)
    if objectives is None:
        objectives = DEFAULT_OBJECTIVES()

    present = np.unique(ids[ids >= 0])
    if restrict_to is not None:
        wanted = set(int(k) for k in np.asarray(restrict_to).ravel())
        present = np.array([k for k in present if int(k) in wanted], dtype=int)

    dt = None if datetime is None else pd.to_datetime(pd.Series(datetime)).reset_index(drop=True)

    rows: list[dict] = []
    index: list[int] = []
    for k in present:
        m = (ids == k) & np.isfinite(obs) & np.isfinite(pred)
        n_ok = int(m.sum())
        row: dict = {"n_steps": n_ok}
        if dt is not None:
            pos = np.flatnonzero(ids == k)
            row["start"] = dt.iloc[int(pos[0])] if pos.size else pd.NaT
        for name, obj in objectives.items():
            if n_ok < min_steps:
                row[name] = np.nan
                continue
            try:
                row[name] = float(obj.compute(obs, pred, mask=m))
            except (ValueError, ZeroDivisionError, FloatingPointError):
                row[name] = np.nan  # degenerate window (constant/zero) — undefined
        rows.append(row)
        index.append(int(k))

    cols = (["n_steps"] + (["start"] if dt is not None else []) + list(objectives))
    if not rows:
        return pd.DataFrame(columns=cols, index=pd.Index([], name="event_id"))
    return pd.DataFrame(rows, index=pd.Index(index, name="event_id"))[cols]


def summarize_event_scores(scores: pd.DataFrame, metrics=("kge", "ccc")) -> dict:
    """Average per-event scores across storms (mean, median, IQR, valid count).

    :param scores: Output of :func:`per_event_scores`.
    :param metrics: Metric column names to summarize.
    :returns: Flat dict with, per metric ``m``: ``{m}_mean``, ``{m}_median``,
        ``{m}_iqr`` (Q3−Q1) and ``{m}_n`` (number of storms with a finite value).
        ``nan`` values are ignored so a few unscoreable storms don't poison the
        average.
    :rtype: dict
    """
    out: dict = {}
    for m in metrics:
        if m not in scores.columns:
            out[f"{m}_mean"] = out[f"{m}_median"] = out[f"{m}_iqr"] = np.nan
            out[f"{m}_n"] = 0
            continue
        vals = scores[m].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[f"{m}_mean"] = out[f"{m}_median"] = out[f"{m}_iqr"] = np.nan
            out[f"{m}_n"] = 0
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        out[f"{m}_mean"] = float(vals.mean())
        out[f"{m}_median"] = float(np.median(vals))
        out[f"{m}_iqr"] = float(q3 - q1)
        out[f"{m}_n"] = int(vals.size)
    return out
