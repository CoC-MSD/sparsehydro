# Examples

Short, runnable scripts on the sample data bundled with sparsehydro (one month
of 5-minute rain and stormflow from meter TC-NB-001). Run them from the
repository root:

```bash
python examples/01_fit_unit_hydrograph.py
```

| Script | Shows |
|---|---|
| `01_fit_unit_hydrograph.py` | The basic loop: data, model, calibration problem, solver, result. Setting parameter bounds for a site. |
| `02_compare_unit_hydrographs.py` | Swapping formulations by registered name and comparing fits in a table. |
| `03_composite_from_config.py` | Wiring components (abstraction, then fast and slow unit hydrographs) from a configuration dictionary or JSON file. |
| `04_events_and_event_scores.py` | Detecting and screening storm events, then scoring a fit storm by storm. |

Each takes 5 to 30 seconds and prints its results. `01` also accepts `--plot`
(needs `plotly`). Nothing is written into the repository.

`tests/test_examples.py` runs every script, so they stay in step with the
library.
