# Contributing to sparsehydro

Thank you for helping improve sparsehydro. This guide explains how the project
is organised, how to set up a development environment, and what a pull request
needs before it can be merged. If anything here is unclear, open an issue and
ask; that is a contribution too.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Contents

1. [What sparsehydro is for](#1-what-sparsehydro-is-for)
2. [Maintainers and decisions](#2-maintainers-and-decisions)
3. [Licensing](#3-licensing)
4. [Development setup](#4-development-setup)
5. [Repository rules](#5-repository-rules)
6. [Branches](#6-branches)
7. [Adding a model formulation](#7-adding-a-model-formulation)
8. [Tests](#8-tests)
9. [Code style](#9-code-style)
10. [Documentation and changelog](#10-documentation-and-changelog)
11. [Pull requests](#11-pull-requests)
12. [Reporting bugs and requesting features](#12-reporting-bugs-and-requesting-features)
13. [Versioning and deprecation](#13-versioning-and-deprecation)
14. [Security issues](#14-security-issues)

## 1. What sparsehydro is for

sparsehydro is a library of small, interchangeable hydrological model
formulations (abstraction stores, unit hydrographs, baselines, snow, and so on)
that are **registered by name and composed through configuration**. A new idea
should usually arrive as a new component that plugs into the existing ones,
not as a new end-to-end model class that hard-codes one particular
combination.

Code that is specific to one study, client, or monitoring network belongs in
`sparsehydro/studies/<study>/`. Nothing outside `sparsehydro/studies` may
import from it.

## 2. Maintainers and decisions

The maintainers are listed in [`pyproject.toml`](pyproject.toml). Caleb
Buahin ([@cbuahin](https://github.com/cbuahin)) is the project lead and has the
final say on design questions and releases.

- Small fixes and additions can go straight to a pull request.
- Changes to a public interface (`IModel`, the registry, calibration
  interfaces, configuration format) or to the numerical behaviour of an
  existing formulation should start as an issue so the approach can be agreed
  before you write the code.

## 3. Licensing

sparsehydro is released under the [MIT License](LICENSE). By submitting a
contribution you agree that it is licensed under the same terms
(inbound = outbound). No separate contributor agreement is required.

Only add code you have the right to contribute. Credit any source you adapt in
a comment or docstring. New dependencies must have an MIT-compatible licence
(MIT, BSD, Apache-2.0, PSF and similar).

## 4. Development setup

You need Python 3.10 or newer and Git. Git LFS is **not** needed.

```bash
git clone https://github.com/CoC-MSD/sparsehydro.git
cd sparsehydro
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

`pip install -e ".[dev]"` installs the package in editable mode with the test,
documentation, numba and joblib extras. The full test suite runs in about a
minute.

To build the documentation locally:

```bash
sphinx-build -b html docs docs/_build/html
```

A small sample dataset for experiments is bundled with the package:

```python
from sparsehydro.sample_data import load_sample
df = load_sample("tc-nb-001")
```

## 5. Repository rules

sparsehydro is used by people who are new to Python and Git, so a fresh clone
must stay small and simple. CI enforces these rules:

- **No notebooks.** Do not commit `.ipynb` files. Keep notebooks locally or in
  a separate examples repository. Runnable examples go in `examples/*.py` as
  plain scripts.
- **No generated outputs.** Plots, HTML reports, CSV exports and similar files
  are written to a temporary or ignored directory, never committed. Tests write
  to `tmp_path`.
- **No large files.** Tracked files must be under 1 MiB. Test data is trimmed
  to the window the tests actually use.
- **No Git LFS.**
- **No credentials.** Never commit passwords, tokens or API keys. Read them
  from function arguments or environment variables.

## 6. Branches

| Branch | Purpose |
|---|---|
| `main` | Released code. Only the project lead merges into `main`, at release time. |
| `dev` | Integration branch. **Open pull requests against `dev`.** |
| `feature/<short-name>` | New functionality. |
| `bugfix/<issue-number>-<short-name>` | Bug fixes. |
| `docs/<short-name>` | Documentation-only changes. |

Keep each branch focused on one change. A pull request that adds a
formulation, refactors calibration and updates an unrelated study is hard to
review; split it.

## 7. Adding a model formulation

Every formulation is a subclass of `sparsehydro.IModel` (or of a more specific
base such as `IUnitHydroComponent`) and follows the same lifecycle:

| Method | Responsibility |
|---|---|
| `__init__` | Store configuration only. Do not take data here. |
| `initialize()` | Reset the registries, then register parameters (`ScalarParameter` / `VectorParameter`), output fields (`FieldRecord`) and constraints (`ConstraintRecord`). |
| `validate()` | Return `True` when the parameter values are usable. |
| `prepare(data)` | Receive forcing data and anything fitted beforehand; cache arrays. |
| `predict_arrays()` | The fast path: return `{field_name: numpy array}`. |
| `predict()` | Return `pd.DataFrame(self.predict_arrays())`. |
| `inequality_constraints()` | Residuals `g`, feasible when `g <= 0`. |
| `finalize()` | Release cached data. |

Then:

1. **Register it** with `@registry.register` and give it a unique,
   lower-case, hyphenated `model_name` (for example `"nash-uh"`). Registered
   components can be built from configuration.
2. **Use physical units** and state them in each parameter's `units`. Time
   parameters are in hours, not time steps, so results do not depend on the
   model time step.
3. **Give every parameter sensible bounds** and a default inside them.
   Express relationships between parameters (for example `T1 < T2`) as a
   `ConstraintRecord` plus an `inequality_constraints()` residual, so that
   optimisers can see them.
4. **Reuse what exists.** Use `convolve_causal` for convolution,
   the unit constants in the package, and existing components inside your own.
   Do not copy code from another module.
5. **Keep observed data out of `predict`.** Anything estimated from
   observations (dry-weather patterns, backgrounds) is fitted beforehand and
   passed to `prepare()`.
6. **Do not print.** Use `logging.getLogger(__name__)`. Raise exceptions for
   invalid input instead of using `assert` or `numpy.testing`.
7. **Add tests and documentation** as described below.

### Unit hydrographs

A unit hydrograph only needs its cumulative unit response: subclass
`UnitHydrographBase`, implement `_cdf(t)` (``t`` in hours) and
`_register_shape_parameters()`, and register it. Discretisation, truncation,
the optional delay, the lifecycle and use inside `RDIIModel` and `MixtureUH`
come from the base class. Add the class to the expected set in
`tests/test_uh_library.py` so the shared conformance tests (unit area,
timestep invariance, registry round-trip) cover it. See
[docs/unit_hydrographs.md](docs/unit_hydrographs.md#adding-a-formulation) for a
worked example.

## 8. Tests

- Every new formulation needs tests for its lifecycle, its outputs on a small
  synthetic case with a known answer (mass balance, unit area, analytical
  limits), and its parameter bounds and constraints.
- When you fix a bug, first add a test that fails because of the bug.
- **Never delete or weaken a test for code you did not change.** If a test
  blocks your change, discuss it in the pull request.
- Tests must not need network access, notebooks, or files outside the
  repository. Use synthetic data or the bundled sample.
- `tests/golden/` pins exact numerical results. If your change is meant to
  alter them, regenerate with `python tests/golden/generate.py`, explain why
  in the pull request, and include a short before/after comparison.

Run a single file or test while working:

```bash
pytest tests/test_rdii_model.py -k triangle
```

## 9. Code style

- Follow PEP 8 with a line length of about 100 characters.
- Add type hints to public functions and methods.
- Write docstrings for public classes and functions, including units for
  physical quantities. The documentation is built from them.
- Prefer clear names and short functions over comments that explain clever
  code.

## 10. Documentation and changelog

- Update `docs/` when you add or change public behaviour. New formulations get
  an entry in the API reference with their equations and parameter table.
- Add a line to the `[Unreleased]` section of [`CHANGELOG.md`](CHANGELOG.md)
  under Added, Changed, Deprecated, Removed or Fixed. Mark breaking changes
  with **BREAKING**.

## 11. Pull requests

1. Open the pull request against `dev`.
2. Fill in the pull request template, including the checklist.
3. Link the issue it resolves (`Closes #123`).
4. Make sure CI passes: tests on Linux, macOS and Windows, the documentation
   build, and the repository hygiene check.
5. At least one maintainer who is not the author must approve. Authors do not
   merge their own pull requests.
6. Pull requests are squash-merged into `dev` unless the commits are already
   clean, logical steps.

Reviewers aim to respond within two weeks. If a pull request has had no
activity from its author for 60 days it may be closed; it can be reopened at
any time.

## 12. Reporting bugs and requesting features

Use the issue templates on GitHub. For bugs, include the sparsehydro version
(`python -c "import sparsehydro; print(sparsehydro.__version__)"`), your Python
version and operating system, a minimal script that reproduces the problem
(synthetic data or the bundled sample if possible), and what you expected to
happen.

## 13. Versioning and deprecation

sparsehydro uses [Semantic Versioning](https://semver.org/). While the version
is below 1.0.0 or carries a pre-release tag, minor releases may contain
breaking changes; they are always listed in the changelog. When an interface
is going to be removed, it is first deprecated with a `DeprecationWarning` for
at least one release where practical.

## 14. Security issues

Do not open a public issue for a security problem. Email the project lead at
[cbuahin@hazenandsawyer.com](mailto:cbuahin@hazenandsawyer.com) with details,
and you will receive a reply within five business days.
