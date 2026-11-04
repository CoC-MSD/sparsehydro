"""
swmmrdii — subcatchment RDII calibration and SWMM integration library.

This package wraps sparsehydro's RDII model into a workflow-oriented API for
calibrating multi-subcatchment RDII models and exporting results to SWMM.

Quick start
-----------
Single subcatchment::

    from swmmrdii import RdiiModel, CalOptimizer

    m = RdiiModel(area_acres=51_053, n_triangles=3, ia_defaults={'snow': True}, name='03145483')
    m.load_forcing(ppt_path='...\\PRISM_RG_022.dat',
                   temp_path='...\\PRISM_RG_022_tmean.dat',
                   temp_conversion='degF_to_degC')
    m.load_observed(tsf_path='...\\USGS_DV.tsf', gage_id='03145483')

    cal = CalOptimizer(m, objective='KGE', sampler='CMA-ES', n_trials=10_000)
    cal.set_windows(cal_start='2008-01-01', cal_end='2015-12-31',
                    val_start='2016-01-01', val_end='2022-12-31')
    cal.optimize()
    cal.summary()
    cal.save_params('artifacts/best_params.json')

Multi-subcatchment (from SWMM)::

    m = RdiiModel.from_swmm(
        inp_path='...\\model.inp',
        db_path='...\\model.db',
        bfi_field='BFI',
        n_triangles=3,
        ia_defaults={'snow': True},
    )
    m.load_forcing_from_sc_df()
    m.load_observed(tsf_path='...\\USGS_DV.tsf', gage_id='03145483')
    m.summary()

    cal = CalOptimizer(m, objective='KGE', n_trials=5_000)
    cal.set_windows('2008-01-01', '2015-12-31', '2016-01-01', '2022-12-31')
    cal.optimize()

See the module docstrings for full API documentation.
"""

# Core workflow classes
from .model import RdiiModel

# Calibration is imported lazily to avoid pulling in calibration/joblib in
# environments where only the SWMM runner is needed (e.g. PCSWMM scripts).
def __getattr__(name):
    if name in ("CalOptimizer", "OBJECTIVE_REGISTRY", "SAMPLER_REGISTRY"):
        from .calibration import CalOptimizer, OBJECTIVE_REGISTRY, SAMPLER_REGISTRY
        globals()["CalOptimizer"] = CalOptimizer
        globals()["OBJECTIVE_REGISTRY"] = OBJECTIVE_REGISTRY
        globals()["SAMPLER_REGISTRY"] = SAMPLER_REGISTRY
        return globals()[name]
    raise AttributeError(f"module 'swmmrdii' has no attribute {name!r}")

# Standalone utilities — importable directly from swmmrdii
from .forcing import (
    load_dat,
    load_tsf,
    load_prism_dat,
    load_tsf_gage,
    resolve_conversion,
    resolve_temp_unit,
    resolve_ppt_unit,
    parse_swmm_timeseries,
    CONVERSIONS,
)
from .swmm_parse import load_swmm_subcatchments, parse_swmm_section
from .diagnostics import mass_balance_report, mass_balance_report_multi
from .io import write_tsf, write_tsf_multi, write_swmm_inflows, write_swmm_timeseries
from .plots import plot_timeseries, plot_aggregated, plot_temporal_scatter, plot_cumulative_volume
from .rtk_initializer import RTKInitializer
from .swmm_runner import (
    check_inp_window,
    compute_network_lags,
    bc_node_lag,
    prepare_swmm_run,
    run_sparsehydro,
    patch_inp,
    rescale_imperv,
    rescale_bfi,
    write_series_tsf,
)

__all__ = [
    # Core classes
    "RdiiModel",
    "CalOptimizer",
    "OBJECTIVE_REGISTRY",
    "SAMPLER_REGISTRY",
    # Forcing loaders
    "load_dat",
    "load_tsf",
    "load_prism_dat",   # backwards-compatible alias
    "load_tsf_gage",    # backwards-compatible alias
    "resolve_conversion",
    "resolve_temp_unit",
    "resolve_ppt_unit",
    "parse_swmm_timeseries",
    "CONVERSIONS",
    # SWMM parsing
    "load_swmm_subcatchments",
    "parse_swmm_section",
    # Diagnostics
    "mass_balance_report",
    "mass_balance_report_multi",
    # SWMM/TSF export
    "write_tsf",
    "write_tsf_multi",
    "write_swmm_inflows",
    "write_swmm_timeseries",
    # Plots
    "plot_timeseries",
    "plot_aggregated",
    "plot_temporal_scatter",
    "plot_cumulative_volume",
    # RTK initializer (exposed for power users)
    "RTKInitializer",
    # SWMM network integration
    "check_inp_window",
    "compute_network_lags",
    "bc_node_lag",
    "prepare_swmm_run",
    "run_sparsehydro",
    "patch_inp",
    "rescale_imperv",
    "rescale_bfi",
    "write_series_tsf",
]
