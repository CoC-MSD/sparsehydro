# Releasing sparsehydro

Releases are published to [PyPI](https://pypi.org/p/sparsehydro) automatically
when a GitHub release is published in `CoC-MSD/sparsehydro`. The workflow
(`.github/workflows/publish.yml`) uses PyPI *trusted publishing*: GitHub proves
its identity to PyPI directly, so there are no API tokens to create, store or
rotate.

## One-time setup

Do this once, as an owner of the `sparsehydro` project on PyPI and an admin of
`CoC-MSD/sparsehydro` on GitHub.

### 1. PyPI: trust the workflow

1. Sign in to <https://pypi.org> (two-factor authentication must be on).
2. Your projects → **sparsehydro** → Manage → **Publishing** →
   *Add a new publisher* → **GitHub**.
3. Enter exactly:

   | Field | Value |
   |---|---|
   | Owner | `CoC-MSD` |
   | Repository name | `sparsehydro` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

4. Under Manage → **Releases**, open `0.0.0` (the placeholder that reserved the
   name) → Options → **Yank**. Until it is yanked, a plain
   `pip install sparsehydro` installs that empty placeholder instead of the
   newest pre-release.

### 2. GitHub: create the `pypi` environment

In `CoC-MSD/sparsehydro` → Settings → **Environments** → *New environment* →
`pypi`:

- **Required reviewers**: the maintainers who may approve a release. Every
  publish then waits for one of them to click *Approve*.
- **Deployment branches and tags**: *Selected branches and tags* → add a tag
  rule `v*`.

Remove any old `PYPI_API_TOKEN` or similar secret; it is no longer used.

### 3. Optional: rehearse on TestPyPI

1. On <https://test.pypi.org> (a separate account) → Account settings →
   **Publishing** → *Add a new pending publisher*: project name `sparsehydro`,
   the same owner, repository and workflow, environment `testpypi`.
2. On GitHub, create an environment `testpypi` the same way as `pypi`.
3. Settings → Secrets and variables → Actions → **Variables** → new repository
   variable `USE_TESTPYPI` = `true`.

With the variable set, every release goes to TestPyPI first and to PyPI only
after that succeeds. Delete the variable to skip TestPyPI.

## Cutting a release

1. **Version.** Set `__version__` in `sparsehydro/__init__.py` (the only place
   it is defined; `pyproject.toml` reads it). Use
   [PEP 440](https://peps.python.org/pep-0440/) spellings: `1.0.0b2`,
   `1.0.0rc1`, `1.0.0`.
2. **Changelog.** In `CHANGELOG.md`, rename `## [Unreleased]` to
   `## [<version>] - <YYYY-MM-DD>`, add a fresh empty `## [Unreleased]` above
   it, and update the comparison links at the bottom.
3. **Merge to `main`** through a pull request, with CI green.
4. **Check locally (optional):**

   ```bash
   python -m build
   python .github/scripts/check_release.py v<version> --dist dist
   ```

5. **Publish the GitHub release.** Releases → *Draft a new release*:
   - *Choose a tag*: type `v<version>` (for example `v1.0.0b1`) and create it
     on `main`.
   - Title `sparsehydro <version>`; paste that version's changelog section as
     the description.
   - Tick **Set as a pre-release** for alpha, beta and rc versions, and leave
     it unticked for final versions.
   - *Publish release*.
6. **Approve.** Actions → *Publish to PyPI*. The workflow:
   1. runs the full test matrix;
   2. builds the sdist and wheel, and checks that the tag matches the
      version, that the pre-release tick matches the version, that the
      changelog has the section, and that the metadata is valid
      (`twine check`); it then installs the wheel and imports it;
   3. publishes to TestPyPI if enabled;
   4. waits for a reviewer to approve the `pypi` environment, then publishes.
7. **Verify** from a clean environment:

   ```bash
   pip install --pre "sparsehydro==<version>"
   python -c "import sparsehydro; print(sparsehydro.__version__)"
   ```

## When something goes wrong

- **A check fails before publishing** (wrong tag, missing changelog section):
  delete the GitHub release *and* its tag, fix the problem on `main`, and
  publish the release again.
- **A bad version reached PyPI**: PyPI never allows the same version number to
  be uploaded twice, even after deletion. **Yank** the bad release on PyPI
  (Manage → Releases → Options → Yank; pip then skips it unless it is pinned)
  and release a new version, for example `1.0.0b2`.
- **"Trusted publishing exchange failure"**: the owner, repository, workflow
  file name or environment name on PyPI does not match the workflow exactly,
  or the release was published from a fork. Releases must be published in
  `CoC-MSD/sparsehydro`.
