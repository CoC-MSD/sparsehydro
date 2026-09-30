"""SparseHydro viewer: a Streamlit app for browsing parsimonious-UH fits by global event.

Run with ``sparsehydro-viewer [path/to/rain_stormflow.csv]`` or
``streamlit run sparsehydro/viewer/app.py -- [path]``.

Loads a ``rain_stormflow.csv`` (``datetime, rain, stormflow``), runs
:func:`~sparsehydro.events.detect_event_hierarchy` and
:class:`~sparsehydro.models.unithydrograph.GlobalSequentialFitter` once per
(file, settings) pair, and steps through global events in four views:
Overview, Smoothing, UH Shape and Convolution.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from sparsehydro.events.hierarchy import GLOBAL_EVENT_GAP_DAYS
from sparsehydro.models.unithydrograph import GammaUH, NashUH, PeakTailUH, TriangleUH
from sparsehydro.viewer.data import (
    SEASON_ORDER,
    ViewerResult,
    content_hash,
    events_in_window,
    filter_sort_events,
    fit_details,
    fit_index,
    fitted_model,
    global_event_table,
    kernel_display_steps,
    member_table,
    read_rain_stormflow,
    run_analysis,
    slice_savgol,
)
from sparsehydro.visualization.unithydrograph import (
    plot_convolution,
    plot_event_hierarchy,
    plot_uh_shapes,
    plot_variable_savgol,
)

MODELS = {
    "PeakTailUH": PeakTailUH,
    "GammaUH": GammaUH,
    "TriangleUH": TriangleUH,
    "NashUH": NashUH,
}

SORT_COLUMNS = {
    "Date": "start",
    "Total rain": "total_rain",
    "Peak flow": "peak_flow",
    "NSE": "nse",
    "Sub-events": "n_sub",
}


def _default_path() -> str:
    """Initial CSV path: ``$SPARSEHYDRO_VIEWER_CSV``, else the command line
    (``streamlit run app.py -- <path>``), else the in-repo sample if present."""
    if os.environ.get("SPARSEHYDRO_VIEWER_CSV"):
        return os.environ["SPARSEHYDRO_VIEWER_CSV"]
    args = [a for a in sys.argv[1:] if a.lower().endswith(".csv")]
    if args:
        return args[0]
    sample = Path("data/TC-NB-001/rain_stormflow.csv")
    return str(sample) if sample.exists() else ""


@st.cache_resource(show_spinner=False, max_entries=4)
def _load(key: str, _raw: bytes) -> pd.DataFrame:
    return read_rain_stormflow(_raw)


@st.cache_resource(show_spinner=False, max_entries=8)
def _run(key: str, _df: pd.DataFrame, model_name: str, gap_days: float,
         peak_weight: float, tail_weight: float) -> ViewerResult:
    # Objectives emit divide-by-zero RuntimeWarnings whenever an optimizer
    # trial predicts a constant series; they are harmless and flood the console.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return run_analysis(_df, MODELS[model_name], global_event_gap_days=gap_days,
                            peak_weight=peak_weight, tail_weight=tail_weight)


def _sidebar_data() -> "tuple[str, bytes, str] | None":
    """Dataset picker. Returns ``(label, raw_bytes, hash)`` or ``None``."""
    st.sidebar.header("Dataset")
    source = st.sidebar.radio("Source", ["Local path", "Upload"], horizontal=True, label_visibility="collapsed")
    if source == "Upload":
        up = st.sidebar.file_uploader("rain_stormflow.csv", type=["csv"])
        if up is None:
            return None
        raw = up.getvalue()
        return up.name, raw, content_hash(raw)
    path = st.sidebar.text_input("CSV path", value=_default_path(),
                                 help="Columns: datetime, rain, stormflow")
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        st.sidebar.error(f"File not found: {path}")
        return None
    raw = p.read_bytes()
    return p.parent.name or p.name, raw, content_hash(raw)


def _sidebar_settings() -> "tuple[str, float, float, float]":
    with st.sidebar.expander("Fit settings"):
        model_name = st.selectbox("UH model", list(MODELS))
        gap_days = st.number_input("Global-event gap (days)", min_value=0.05, max_value=30.0,
                                   value=float(GLOBAL_EVENT_GAP_DAYS), step=0.25)
        peak_weight = st.number_input("Peak-zone weight", min_value=0.0, value=3.0, step=0.5)
        tail_weight = st.number_input("Tail weight", min_value=0.0, value=1.0, step=0.5)
    return model_name, float(gap_days), float(peak_weight), float(tail_weight)


def _sidebar_navigation(table: pd.DataFrame) -> "int | None":
    """Filter/sort controls plus prev/next navigation. Returns the selected ``global_id``."""
    st.sidebar.header("Global events")
    seasons = st.sidebar.multiselect("Season", SEASON_ORDER, placeholder="All seasons")
    max_rain = float(table["total_rain"].max()) if not table.empty else 0.0
    min_rain = st.sidebar.slider("Min total rain", 0.0, max(max_rain, 0.01), 0.0)
    c1, c2 = st.sidebar.columns([3, 2])
    sort_label = c1.selectbox("Sort by", list(SORT_COLUMNS))
    descending = c2.toggle("Desc.", value=sort_label != "Date")

    view = filter_sort_events(table, seasons=seasons, min_rain=min_rain,
                              sort_by=SORT_COLUMNS[sort_label], descending=descending)
    if view.empty:
        st.sidebar.warning("No global events match the filters.")
        return None

    options = view["global_id"].tolist()
    if st.session_state.get("gev") not in options:
        st.session_state["gev"] = options[0]
    pos = options.index(st.session_state["gev"])

    def _step(delta: int) -> None:
        i = options.index(st.session_state["gev"])
        st.session_state["gev"] = options[min(max(i + delta, 0), len(options) - 1)]

    labels = {row.global_id: f"G{row.global_id} · {row.start:%Y-%m-%d} · "
                             f"{row.n_sub} sub · {row.total_rain:.2f} rain"
              for row in view.itertuples()}
    b1, b2, b3 = st.sidebar.columns([1, 2, 1])
    b1.button("◀", on_click=_step, args=(-1,), disabled=pos == 0, width="stretch", help="Previous")
    b2.markdown(f"<div style='text-align:center;padding-top:0.45rem'>{pos + 1} / {len(options)}</div>",
                unsafe_allow_html=True)
    b3.button("▶", on_click=_step, args=(1,), disabled=pos == len(options) - 1, width="stretch",
              help="Next")
    st.sidebar.selectbox("Global event", options, key="gev", format_func=labels.get)
    return int(st.session_state["gev"])


def main() -> None:
    st.set_page_config(page_title="SparseHydro Viewer", page_icon="🌧️", layout="wide")

    picked = _sidebar_data()
    model_name, gap_days, peak_weight, tail_weight = _sidebar_settings()
    if picked is None:
        st.title("SparseHydro Viewer")
        st.info("Choose a `rain_stormflow.csv` (columns `datetime, rain, stormflow`) in the sidebar.")
        return
    name, raw, key = picked

    try:
        df = _load(key, raw)
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.spinner(f"Detecting events and fitting {model_name} on {name} ({len(df):,} rows)…"):
        result = _run(key, df, model_name, gap_days, peak_weight, tail_weight)

    table = global_event_table(result)
    if table.empty:
        st.title(name)
        st.warning("No events were detected in this dataset.")
        return

    gid = _sidebar_navigation(table)
    if gid is None:
        st.title(name)
        return

    row = table.set_index("global_id").loc[gid]
    window = (row.window_start, row.window_end)
    st.title(f"{name} — G{gid}")
    st.caption(f"{row.start:%Y-%m-%d %H:%M} → {row.end:%Y-%m-%d %H:%M} · {row.season} · "
               f"{len(table)} global events, {len(result.sub_events)} sub-events, "
               f"{len(result.summary.events)} fitted ({result.summary.model_class_name})")
    m = st.columns(4)
    m[0].metric("Sub-events", f"{row.n_sub}")
    m[1].metric("Total rain", f"{row.total_rain:.2f}")
    m[2].metric("Peak flow", f"{row.peak_flow:.2f}")
    m[3].metric("NSE (event span)", "—" if pd.isna(row.nse) else f"{row.nse:.2f}",
                help="Nash-Sutcliffe efficiency of the fitted prediction over the member "
                     "sub-events' span (excluding the padded lead-in / recession window).")

    t0, t1 = window
    in_win = (df["datetime"] >= t0) & (df["datetime"] <= t1)
    idx = fit_index(result, gid)
    tab_overview, tab_smooth, tab_uh, tab_conv = st.tabs(["Overview", "Smoothing", "UH Shape", "Convolution"])

    with tab_overview:
        fig = plot_event_hierarchy(
            df[in_win],
            events_in_window(result.global_events, t0, t1),
            events_in_window(result.sub_events, t0, t1),
            savgol_result=slice_savgol(result.savgol, t0, t1),
            title=f"Global event G{gid}",
        )
        st.plotly_chart(fig, width="stretch")

    with tab_smooth:
        st.plotly_chart(plot_variable_savgol(slice_savgol(result.savgol, t0, t1), df[in_win],
                                             title=f"Variable-window smoothing — G{gid}"),
                        width="stretch")

    with tab_uh:
        if idx is None:
            st.info("This global event was not fitted.")
        else:
            factory = MODELS[model_name]
            items = [(f"G{gid}", fitted_model(result.summary, idx, factory))]
            if st.toggle("Compare with neighbouring events", value=False):
                for j in (idx - 1, idx + 1):
                    if 0 <= j < len(result.summary.events):
                        items.append((f"G{result.summary.events[j].global_id}",
                                      fitted_model(result.summary, j, factory)))
            fig = plot_uh_shapes(items, dt_hours=result.dt_hours,
                                 title=f"Fitted {result.summary.model_class_name} — G{gid}")
            steps = kernel_display_steps([m.get_kernel(result.dt_hours) for _, m in items])
            fig.update_xaxes(range=[0, steps * result.dt_hours])
            st.plotly_chart(fig, width="stretch")

    with tab_conv:
        if idx is None:
            st.info("This global event was not fitted.")
        else:
            span = SimpleNamespace(start_datetime=t0, end_datetime=t1)
            st.plotly_chart(plot_convolution(result.summary, event=span, title=f"Convolution — G{gid}"),
                            width="stretch")

    st.subheader("Fit")
    details = fit_details(result, gid)
    if details.empty:
        st.caption("Not fitted.")
    else:
        st.dataframe(details, hide_index=True, width="stretch")
    st.subheader("Sub-events")
    st.dataframe(member_table(result, gid), hide_index=True, width="stretch")


main()
