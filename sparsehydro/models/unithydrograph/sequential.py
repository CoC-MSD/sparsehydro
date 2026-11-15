"""Sequential event-by-event unit hydrograph fitting.

Uses :class:`~sparsehydro.calibration.CalibrationProblem` for data preparation
and objective evaluation, and constructs a
:class:`~sparsehydro.calibration.CalibrationResult` per event.

Each fit starts from the model's current parameter values: the previous
event's fitted values when warm starting, otherwise the model's defaults. This
calls ``scipy.optimize.minimize`` directly (``ScipySolver`` always restarts
from the bounds midpoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize as scipy_minimize
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter

from ...calibration.objectives import RMSE, KGE, NashSutcliffe, WeightedRMSE, IObjective
from ...calibration.problem import CalibrationProblem
from ...calibration.result import CalibrationResult
from ..base import IModel
from .kernels import infer_dt_hours_from_values
from ...events import EventRecord


def _start_point(problem: CalibrationProblem, model: IModel) -> np.ndarray:
    """The model's current values of the calibrated parameters, clipped to bounds."""
    xl, xu = problem.bounds
    x0 = np.array([model.read_flat_parameter(name) for name in problem.param_names], dtype=float)
    return np.clip(x0, xl, xu)


def _zonal_weight_vector(event, datetimes, peak_weight: float, tail_weight: float) -> np.ndarray:
    """Build a per-timestep zonal weight vector for one event window.

    Points inside the peak zone ``[rise_start_datetime, tail_start_datetime]``
    receive *peak_weight*; all others receive *tail_weight*.  Events without the
    zone attributes (e.g. a plain :class:`~sparsehydro.events.EventRecord`) get a
    uniform *tail_weight* vector.

    :param event: Event exposing optional ``rise_start_datetime`` /
        ``tail_start_datetime`` attributes.
    :type event: object
    :param datetimes: Timestamps of the event window.
    :type datetimes: array-like
    :param peak_weight: Weight applied inside the peak zone.
    :type peak_weight: float
    :param tail_weight: Weight applied outside the peak zone.
    :type tail_weight: float
    :returns: Per-timestep weight vector aligned with *datetimes*.
    :rtype: numpy.ndarray
    """
    dts = pd.to_datetime(pd.Series(np.asarray(datetimes)).reset_index(drop=True))
    w = np.full(len(dts), float(tail_weight), dtype=float)
    rs = getattr(event, "rise_start_datetime", None)
    ts = getattr(event, "tail_start_datetime", None)
    if rs is not None and ts is not None and not pd.isna(rs) and not pd.isna(ts):
        peak_mask = (dts >= pd.Timestamp(rs)) & (dts <= pd.Timestamp(ts))
        w[peak_mask.to_numpy()] = float(peak_weight)
    return w


@dataclass
class SequentialFitSummary:
    """Results of a sequential unit hydrograph fitting run.

    :ivar events: Events that were fitted, in chronological order.
    :vartype events: list[EventRecord]
    :ivar calibration_results: One
        :class:`~sparsehydro.calibration.CalibrationResult` per event.
        ``pareto_X[0]`` holds fitted parameters;
        ``objective_display_values()[0]`` returns [RMSE, NSE, KGE].
    :vartype calibration_results: list[CalibrationResult]
    :ivar global_predicted: Full-domain accumulated predicted flow.
        Columns: ``datetime``, ``Q_pred``.
    :vartype global_predicted: pandas.DataFrame
    :ivar global_observed: Full-domain stormflow.
        Columns: ``datetime``, ``stormflow``.
    :vartype global_observed: pandas.DataFrame
    :ivar model_class_name: Name of the model class produced by ``model_factory``.
    :vartype model_class_name: str
    :ivar fitted_effective_areas: Fitted effective area per event:
        ``sum(Q_pred) / sum(rain)`` over the event window.  Parallel to
        *events* and *calibration_results*.
    :vartype fitted_effective_areas: list[float]
    """

    events: list[EventRecord]
    calibration_results: list[CalibrationResult]
    global_predicted: pd.DataFrame
    global_observed: pd.DataFrame
    model_class_name: str
    fitted_effective_areas: list[float]
    dt_hours: float = 1.0 / 12.0

    def parameter_evolution(self) -> pd.DataFrame:
        """Return fitted parameters indexed by event.

        Columns: ``event_id``, ``start_datetime``, ``end_datetime``,
        then one column per calibrated parameter.

        :returns: One row per fitted event with its calibrated parameters.
        :rtype: pandas.DataFrame
        """
        rows = []
        for event, result in zip(self.events, self.calibration_results):
            row: dict = {
                "event_id": event.event_id,
                "start_datetime": event.start_datetime,
                "end_datetime": event.end_datetime,
            }
            row.update(dict(zip(result.param_names, result.pareto_X[0])))
            rows.append(row)
        return pd.DataFrame(rows)

    def metrics_summary(self) -> pd.DataFrame:
        """Return per-event objective values in display form.

        Columns: ``event_id``, then one column per objective name.

        :returns: One row per fitted event with display-form objective values.
        :rtype: pandas.DataFrame
        """
        rows = []
        for event, result in zip(self.events, self.calibration_results):
            display_F = result.objective_display_values()[0]
            row: dict = {"event_id": event.event_id}
            for name, val in zip(result.objective_names, display_F):
                row[name] = val
            rows.append(row)
        return pd.DataFrame(rows)

    def calibrated_areas(self) -> list[float]:
        """Return the calibrated UH area parameter per event.

        For a single UH model this is the ``A`` parameter.
        For an ensemble (aliases like ``fast_A``, ``slow_A``) it is the sum of all
        ``*_A`` parameters — consistent with ``mode="sum"`` and fixed weights of 1.0.

        ``A`` is a flow per unit rainfall *rate* and does not depend on the time
        step. It is divided by :attr:`dt_hours` here so the result is on the same
        per-step scale as the observed effective area ``sum(flow) / sum(rain)``.

        :returns: Calibrated total ``A / dt_hours`` per fitted event.
        :rtype: list[float]
        :raises ValueError: If no ``A``-bearing parameter is found in a
            calibration result.
        """
        result: list[float] = []
        for cal in self.calibration_results:
            params = dict(zip(cal.param_names, cal.pareto_X[0]))
            a_params = {k: v for k, v in params.items()
                        if k == "A" or k.endswith("_A")}
            if not a_params:
                raise ValueError(
                    f"No 'A' parameter found among calibrated params: {list(params.keys())}"
                )
            result.append(float(sum(a_params.values())) / self.dt_hours)
        return result

    def effective_area_summary(self) -> pd.DataFrame:
        """Return observed and fitted effective area per event.

        Columns: ``event_id``, ``start_datetime``, ``observed_ae``, ``fitted_ae``.

        :returns: One row per fitted event comparing observed vs fitted area.
        :rtype: pandas.DataFrame
        """
        rows = [
            {
                "event_id": e.event_id,
                "start_datetime": e.start_datetime,
                "observed_ae": e.effective_area,
                "fitted_ae": ae,
            }
            for e, ae in zip(self.events, self.fitted_effective_areas)
        ]
        return pd.DataFrame(rows)


class SequentialFitter:
    """Fit an :class:`~sparsehydro.models.IModel` to storm events one by one.

    :param model_factory: Zero-argument callable returning a fresh
        :class:`~sparsehydro.models.IModel` in CREATED state.  Works with single
        UH models and :class:`~sparsehydro.models.EnsembleModel`::

            SequentialFitter(lambda: GammaUH(), data, events)
            SequentialFitter(lambda: make_ensemble(), data, events)

    :type model_factory: Callable[[], IModel]
    :param rain_stormflow: Full time series with columns ``datetime``,
        ``rain``, ``stormflow``.
    :type rain_stormflow: pandas.DataFrame
    :param events: Events to fit (sorted internally by ``start_datetime``).
    :type events: list[EventRecord]
    :param output_column: Column name produced by ``model.predict()``.
        Default ``"Q_pred"``.
    :type output_column: str
    """

    def __init__(
        self,
        model_factory: Callable[[], IModel],
        rain_stormflow: pd.DataFrame,
        events: list[EventRecord],
        output_column: str = "Q_pred",
    ) -> None:
        self._factory = model_factory
        self._data = rain_stormflow.copy()
        self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        self._data["stormflow"] = np.maximum(self._data["stormflow"].values, 0.0)
        self._events = sorted(events, key=lambda e: e.start_datetime)
        self._output_column = output_column

    def fit(
        self,
        time_range: tuple[str, str] | None = None,
        objectives: list[IObjective] | None = None,
        calibration_objective_index: int = 0,
        method: str = "Nelder-Mead",
        smooth_obs: bool = True,
        savgol_options: dict | None = None,
        zonal_weights: bool = False,
        peak_weight: float = 3.0,
        tail_weight: float = 1.0,
        verbose: bool = True,
    ) -> SequentialFitSummary:
        """Run the sequential fitting loop.

        :param time_range: Restrict fitting to this ``(start, end)`` date range.
        :type time_range: tuple[str, str] | None
        :param objectives: Objectives to evaluate; defaults to
            ``[RMSE(), NashSutcliffe(), KGE()]``.
        :type objectives: list[IObjective] | None
        :param calibration_objective_index: Index of *objectives* to minimise
            (default 0 = RMSE).  Ignored when *zonal_weights* is ``True`` (the
            weighted objective is always minimised).
        :type calibration_objective_index: int
        :param method: ``scipy.optimize.minimize`` method, or
            ``"differential_evolution"`` for a global search.
        :type method: str
        :param smooth_obs: Apply a light Savitzky-Golay smooth to observed flow
            within each event.
        :type smooth_obs: bool
        :param savgol_options: Options for the observed smoothing —
            ``{"window_length": int, "polyorder": int}`` (defaults 5 / 2).
        :type savgol_options: dict | None
        :param zonal_weights: When ``True`` and events expose
            ``rise_start_datetime`` / ``tail_start_datetime`` (i.e.
            :class:`~sparsehydro.events.SubEventRecord`), minimise a
            :class:`~sparsehydro.calibration.WeightedRMSE` that up-weights the
            peak zone.
        :type zonal_weights: bool
        :param peak_weight: Per-timestep weight applied inside the peak zone.
        :type peak_weight: float
        :param tail_weight: Per-timestep weight applied outside the peak zone.
        :type tail_weight: float
        :param verbose: Print per-event progress.
        :type verbose: bool
        :returns: Summary of the sequential fitting run.
        :rtype: SequentialFitSummary
        """
        if objectives is None:
            objectives = [RMSE(), NashSutcliffe(), KGE()]
        sg_opts = savgol_options or {}
        sg_wlen = int(sg_opts.get("window_length", 5))
        sg_poly = int(sg_opts.get("polyorder", 2))

        data = self._filter_data(time_range)
        events = self._filter_events(time_range)

        if not events:
            return SequentialFitSummary(
                events=[],
                calibration_results=[],
                global_predicted=pd.DataFrame({"datetime": data["datetime"], "Q_pred": np.zeros(len(data))}),
                global_observed=data[["datetime", "stormflow"]].copy(),
                model_class_name=self._factory().__class__.__name__,
                fitted_effective_areas=[],
                dt_hours=infer_dt_hours_from_values(data["datetime"].values),
            )

        datetime_arr = data["datetime"].values
        Q_residual = data["stormflow"].values.copy()
        Q_global = np.zeros(len(data))

        cal_results: list[CalibrationResult] = []
        fitted_events: list[EventRecord] = []
        fitted_areas: list[float] = []
        last_x: np.ndarray | None = None
        last_param_names: list[str] = []

        n_events = len(events)
        end_dt = pd.Timestamp(datetime_arr[-1])

        for i, event in enumerate(events):
            if verbose:
                print(f"[{i+1}/{n_events}] Event {event.event_id}: {event.start_datetime} → {event.end_datetime}")

            fit_mask = (data["datetime"] >= event.start_datetime) & (data["datetime"] <= event.end_datetime)
            if not fit_mask.any():
                if verbose:
                    print("  Skipped: no data in event window.")
                continue

            fit_indices = np.where(fit_mask.values)[0]
            idx_s, idx_e = int(fit_indices[0]), int(fit_indices[-1]) + 1

            app_end = events[i + 1].start_datetime if i < n_events - 1 else end_dt
            app_mask = (data["datetime"] >= event.start_datetime) & (data["datetime"] < app_end)
            app_indices = np.where(app_mask.values)[0]

            event_data = data[fit_mask].copy().reset_index(drop=True)
            obs = Q_residual[idx_s:idx_e].copy()
            if smooth_obs and len(obs) >= max(sg_wlen, 3):
                wlen = sg_wlen if sg_wlen % 2 == 1 else sg_wlen + 1
                wlen = min(wlen, len(obs) if len(obs) % 2 == 1 else len(obs) - 1)
                if wlen >= sg_poly + 2:
                    obs = savgol_filter(obs, window_length=wlen, polyorder=sg_poly, mode="interp")
            obs = np.maximum(obs, 0.0)
            event_data["stormflow"] = obs

            event_objectives = objectives
            obj_idx = calibration_objective_index
            if zonal_weights:
                w_vec = _zonal_weight_vector(event, event_data["datetime"], peak_weight, tail_weight)
                event_objectives = [WeightedRMSE(weights=w_vec), *objectives]
                obj_idx = 0

            model_i = self._factory()
            model_i.initialize()
            if last_x is not None and last_param_names:
                for name, val in zip(last_param_names, last_x):
                    if name in model_i.scalar_parameter_names:
                        p = model_i.get_scalar_parameter(name)
                        p.update(value=float(np.clip(val, p.lower_bound, p.upper_bound)))
            model_i.validate()

            try:
                problem = CalibrationProblem(
                    model=model_i,
                    data=event_data,
                    objectives=event_objectives,
                    column_map={"observed": "stormflow", "predicted": self._output_column},
                )
            except Exception as exc:
                if verbose:
                    print(f"  CalibrationProblem failed: {exc!r} — skipped.")
                continue

            xl, xu = problem.bounds
            x0 = _start_point(problem, model_i)

            best_x, success = self._optimise(problem, x0, xl, xu, method, obj_idx)

            all_F = problem.evaluate(best_x).reshape(1, -1)
            cal_result = CalibrationResult(
                history=[],
                pareto_X=best_x.reshape(1, -1),
                pareto_F=all_F,
                param_names=problem.param_names,
                objective_names=problem.objective_names,
                minimize_flags=problem.minimize_flags,
            )

            if verbose:
                disp = cal_result.objective_display_values()[0]
                obj_strs = ", ".join(f"{n}={v:.4f}" for n, v in zip(cal_result.objective_names, disp))
                print(f"  {obj_strs}, success={success}")

            for name, val in zip(problem.param_names, best_x):
                if name in model_i.scalar_parameter_names:
                    model_i.get_scalar_parameter(name).update(value=float(val))

            orig_rain = float(data[fit_mask]["rain"].values.sum())
            orig_flow = float(np.maximum(data[fit_mask]["stormflow"].values, 0.0).sum())
            event.effective_area = orig_flow / orig_rain if orig_rain > 0 else 0.0

            app_data = data[app_mask].copy().reset_index(drop=True)
            fitted_ae = 0.0
            if len(app_data) > 0:
                try:
                    model_i.prepare(app_data)
                    app_pred = model_i.predict()
                    q_app = app_pred[self._output_column].values
                    pred_len = min(len(q_app), len(app_indices))
                    Q_global[app_indices[:pred_len]] += q_app[:pred_len]
                    Q_residual[app_indices[:pred_len]] -= q_app[:pred_len]
                    Q_residual = np.maximum(Q_residual, 0.0)

                    n_event = len(event_data)
                    rain_event_sum = float(event_data["rain"].values.sum())
                    if rain_event_sum > 0 and n_event <= len(q_app):
                        fitted_ae = float(q_app[:n_event].sum()) / rain_event_sum
                except Exception as exc:
                    if verbose:
                        print(f"  Application window prediction failed: {exc!r}")

            last_x = best_x
            last_param_names = problem.param_names
            cal_results.append(cal_result)
            fitted_events.append(event)
            fitted_areas.append(fitted_ae)

        return SequentialFitSummary(
            events=fitted_events,
            calibration_results=cal_results,
            global_predicted=pd.DataFrame({"datetime": datetime_arr, "Q_pred": Q_global}),
            global_observed=data[["datetime", "stormflow"]].copy(),
            model_class_name=self._factory().__class__.__name__,
            fitted_effective_areas=fitted_areas,
            dt_hours=infer_dt_hours_from_values(datetime_arr),
        )

    def _optimise(self, problem, x0, xl, xu, method, obj_idx):
        """Minimise a single objective of *problem* from start point *x0*.

        Delegates to :func:`_optimise_problem`; kept as a method because it is
        part of this class's established surface.

        :param problem: Calibration problem exposing ``evaluate``.
        :type problem: CalibrationProblem
        :param x0: Initial parameter vector.
        :type x0: numpy.ndarray
        :param xl: Lower parameter bounds.
        :type xl: numpy.ndarray
        :param xu: Upper parameter bounds.
        :type xu: numpy.ndarray
        :param method: ``scipy.optimize.minimize`` method or
            ``"differential_evolution"``.
        :type method: str
        :param obj_idx: Index of the objective to minimise.
        :type obj_idx: int
        :returns: Tuple ``(best_x, success)``.
        :rtype: tuple[numpy.ndarray, bool]
        """
        return _optimise_problem(problem, x0, xl, xu, method, obj_idx)

    def _filter_data(self, time_range):
        """Return the forcing data optionally restricted to *time_range*.

        :param time_range: ``(start, end)`` date range, or ``None`` for all data.
        :type time_range: tuple[str, str] | None
        :returns: Filtered forcing DataFrame with a reset index.
        :rtype: pandas.DataFrame
        """
        data = self._data.copy()
        if time_range is not None:
            s, e = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
            data = data[(data["datetime"] >= s) & (data["datetime"] <= e)]
        return data.reset_index(drop=True)

    def _filter_events(self, time_range):
        """Return the events optionally restricted to *time_range*.

        :param time_range: ``(start, end)`` date range, or ``None`` for all events.
        :type time_range: tuple[str, str] | None
        :returns: Events fully contained within the range.
        :rtype: list[EventRecord]
        """
        evts = self._events
        if time_range is not None:
            s, e = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
            evts = [ev for ev in evts if ev.start_datetime >= s and ev.end_datetime <= e]
        return evts


def _optimise_problem(problem, x0, xl, xu, method, obj_idx):
    """Minimise a single objective of *problem* from start point *x0*.

    :param problem: Calibration problem exposing ``evaluate``.
    :type problem: CalibrationProblem
    :param x0: Initial parameter vector.
    :type x0: numpy.ndarray
    :param xl: Lower parameter bounds.
    :type xl: numpy.ndarray
    :param xu: Upper parameter bounds.
    :type xu: numpy.ndarray
    :param method: ``scipy.optimize.minimize`` method or ``"differential_evolution"``.
    :type method: str
    :param obj_idx: Index of the objective to minimise.
    :type obj_idx: int
    :returns: Tuple ``(best_x, success)``.
    :rtype: tuple[numpy.ndarray, bool]
    """
    bounds_list = list(zip(xl.tolist(), xu.tolist()))

    # Only one objective is minimised here, but a problem typically carries
    # several; evaluate_single() skips the ones whose values are discarded.
    # Resolved once rather than per call, and falls back for any problem-like
    # object that predates the method.
    evaluate_single = getattr(problem, "evaluate_single", None)
    if evaluate_single is not None:
        def scalar_fn(x):
            return evaluate_single(x, obj_idx)
    else:
        def scalar_fn(x):
            return float(problem.evaluate(x)[obj_idx])

    if method == "differential_evolution":
        res = differential_evolution(scalar_fn, bounds=bounds_list, rng=42, polish=True)
        return res.x, bool(res.success)
    try:
        res = scipy_minimize(
            scalar_fn, x0=x0, bounds=Bounds(lb=xl, ub=xu), method=method,
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8},
        )
        return res.x, bool(res.success)
    except Exception:
        return x0.copy(), False


class GlobalSequentialFitter:
    """Fit one model per global event over its padded window (joint sub-event fit).

    Unlike :class:`SequentialFitter` (residual, one fit per event), this fits a
    single unit hydrograph per :class:`~sparsehydro.events.GlobalEvent` window,
    minimising a :class:`~sparsehydro.calibration.WeightedRMSE` over the union of
    the member sub-events' peak zones, and warm-starts each global event from the
    previous one.  Padding outside the member events is given zero weight.

    :param model_factory: Zero-argument callable returning a fresh model.
    :type model_factory: Callable[[], IModel]
    :param rain_stormflow: Full time series with ``datetime``, ``rain``,
        ``stormflow`` (and optionally ``temperature_c``).
    :type rain_stormflow: pandas.DataFrame
    :param global_events: Global events to fit.
    :type global_events: list
    :param sub_events: Sub-events providing member peak zones (matched by ``global_id``).
    :type sub_events: list
    :param output_column: Column produced by ``model.predict()``.  Default ``"Q_pred"``.
    :type output_column: str
    """

    def __init__(
        self,
        model_factory: Callable[[], IModel],
        rain_stormflow: pd.DataFrame,
        global_events: list,
        sub_events: list,
        output_column: str = "Q_pred",
    ) -> None:
        self._factory = model_factory
        self._data = rain_stormflow.copy()
        self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        self._data["stormflow"] = np.maximum(self._data["stormflow"].values, 0.0)
        self._global_events = sorted(global_events, key=lambda g: g.window_start_datetime)
        self._subs_by_global: dict = {}
        for s in sub_events:
            self._subs_by_global.setdefault(s.global_id, []).append(s)
        self._output_column = output_column

    def fit(
        self,
        objectives: list[IObjective] | None = None,
        method: str = "Nelder-Mead",
        smooth_obs: bool = True,
        savgol_options: dict | None = None,
        peak_weight: float = 3.0,
        tail_weight: float = 1.0,
        verbose: bool = True,
    ) -> SequentialFitSummary:
        """Fit each global event's window and accumulate predictions.

        :param objectives: Reporting objectives; defaults to
            ``[RMSE(), NashSutcliffe(), KGE()]``.  A weighted RMSE is always
            prepended and minimised.
        :type objectives: list[IObjective] | None
        :param method: ``scipy.optimize.minimize`` method or ``"differential_evolution"``.
        :type method: str
        :param smooth_obs: Light Savitzky-Golay smooth of observed flow per window.
        :type smooth_obs: bool
        :param savgol_options: ``{"window_length": int, "polyorder": int}``.
        :type savgol_options: dict | None
        :param peak_weight: Weight inside member peak zones.
        :type peak_weight: float
        :param tail_weight: Weight inside member events but outside peak zones.
        :type tail_weight: float
        :param verbose: Print per-global-event progress.
        :type verbose: bool
        :returns: Summary whose ``events`` are the fitted global events.
        :rtype: SequentialFitSummary
        """
        if objectives is None:
            objectives = [RMSE(), NashSutcliffe(), KGE()]
        sg_opts = savgol_options or {}
        sg_wlen = int(sg_opts.get("window_length", 5))
        sg_poly = int(sg_opts.get("polyorder", 2))

        data = self._data
        datetime_arr = data["datetime"].values
        Q_global = np.zeros(len(data))
        cal_results: list[CalibrationResult] = []
        fitted_events: list = []
        fitted_areas: list[float] = []
        last_x: np.ndarray | None = None
        last_names: list[str] = []

        for gi, ge in enumerate(self._global_events):
            mask = (data["datetime"] >= ge.window_start_datetime) & (data["datetime"] <= ge.window_end_datetime)
            idxs = np.where(mask.values)[0]
            if idxs.size == 0:
                continue
            win = data[mask].copy().reset_index(drop=True)
            obs = win["stormflow"].values.copy()
            if smooth_obs and len(obs) >= max(sg_wlen, 3):
                wlen = sg_wlen if sg_wlen % 2 == 1 else sg_wlen + 1
                wlen = min(wlen, len(obs) if len(obs) % 2 == 1 else len(obs) - 1)
                if wlen >= sg_poly + 2:
                    obs = savgol_filter(obs, wlen, sg_poly, mode="interp")
            win["stormflow"] = np.maximum(obs, 0.0)

            members = self._subs_by_global.get(ge.global_id, [])
            w = np.zeros(len(win), dtype=float)
            dts = win["datetime"]
            for s in members:
                w[((dts >= s.start_datetime) & (dts <= s.end_datetime)).to_numpy()] = tail_weight
                w[((dts >= s.rise_start_datetime) & (dts <= s.tail_start_datetime)).to_numpy()] = peak_weight
            if not np.any(w > 0):
                w[:] = tail_weight

            model = self._factory()
            model.initialize()
            if last_x is not None and last_names:
                for nm, val in zip(last_names, last_x):
                    if nm in model.scalar_parameter_names:
                        p = model.get_scalar_parameter(nm)
                        p.update(value=float(np.clip(val, p.lower_bound, p.upper_bound)))
            model.validate()

            try:
                problem = CalibrationProblem(
                    model=model, data=win,
                    objectives=[WeightedRMSE(weights=w), *objectives],
                    column_map={"observed": "stormflow", "predicted": self._output_column},
                )
            except Exception as exc:
                if verbose:
                    print(f"  GE{ge.global_id} problem failed: {exc!r} — skipped.")
                continue

            xl, xu = problem.bounds
            x0 = _start_point(problem, model)
            best_x, success = _optimise_problem(problem, x0, xl, xu, method, 0)

            all_F = problem.evaluate(best_x).reshape(1, -1)
            cal = CalibrationResult(
                history=[], pareto_X=best_x.reshape(1, -1), pareto_F=all_F,
                param_names=problem.param_names, objective_names=problem.objective_names,
                minimize_flags=problem.minimize_flags,
            )
            for nm, val in zip(problem.param_names, best_x):
                if nm in model.scalar_parameter_names:
                    model.get_scalar_parameter(nm).update(value=float(val))

            fitted_ae = 0.0
            try:
                model.prepare(win)
                pred = model.predict()
                q = pred[self._output_column].values
                pred_len = min(len(q), idxs.size)
                Q_global[idxs[:pred_len]] += q[:pred_len]
                rain_sum = float(win["rain"].values.sum())
                if rain_sum > 0:
                    fitted_ae = float(q[:pred_len].sum()) / rain_sum
            except Exception as exc:
                if verbose:
                    print(f"  GE{ge.global_id} prediction failed: {exc!r}")

            if verbose:
                disp = cal.objective_display_values()[0]
                s = ", ".join(f"{n}={v:.4f}" for n, v in zip(cal.objective_names, disp))
                print(f"[{gi+1}/{len(self._global_events)}] GE{ge.global_id} ({len(members)} sub): {s}, success={success}")

            last_x = best_x
            last_names = problem.param_names
            cal_results.append(cal)
            fitted_events.append(ge)
            fitted_areas.append(fitted_ae)

        return SequentialFitSummary(
            events=fitted_events,
            calibration_results=cal_results,
            global_predicted=pd.DataFrame({"datetime": datetime_arr, "Q_pred": Q_global}),
            global_observed=data[["datetime", "stormflow"]].copy(),
            model_class_name=self._factory().__class__.__name__,
            fitted_effective_areas=fitted_areas,
            dt_hours=infer_dt_hours_from_values(datetime_arr),
        )


__all__ = ["SequentialFitter", "SequentialFitSummary", "GlobalSequentialFitter"]
