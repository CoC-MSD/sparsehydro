"""Package structure rules: layers, the study boundary, and thin __init__ files.

Layers (a module may import only from its own layer or lower ones):

  0  core       enums, parameters, registry, units, _numba, _columns, _validation
  1  leaves     metrics, filters, data, sample_data
  2  events
  3  models
  4  calibration
  5  configuration
  6  visualization
  7  apps       viewer, studies

Imports inside ``if TYPE_CHECKING:`` blocks are ignored (they never run).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "sparsehydro"

LAYERS = {
    "enums": 0, "parameters": 0, "registry": 0, "units": 0, "_numba": 0, "_columns": 0,
    "_validation": 0,
    "metrics": 1, "filters": 1, "data": 1, "sample_data": 1,
    "events": 2,
    "models": 3,
    "calibration": 4,
    "configuration": 5,
    "visualization": 6,
    "viewer": 7, "studies": 7,
}

#: Known upward imports, each with the reason it is acceptable.
ALLOWED = {
    # ModelRegistry checks that registered classes subclass IModel; the import
    # runs lazily inside the check, never at import time.
    ("registry", "models"),
}


def modules():
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).with_suffix("")
        yield path, ("sparsehydro",) + rel.parts


def top_package(dotted: tuple[str, ...]) -> str | None:
    return dotted[1] if len(dotted) > 1 and dotted[1] != "__init__" else None


def runtime_imports(path: Path, module: tuple[str, ...]):
    """Yield absolute dotted targets of every import that can run."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            skip.update(id(n) for n in ast.walk(node))
    package = module[:-1] if module[-1] != "__init__" else module[:-1]
    if module[-1] == "__init__":
        package = module[:-1]
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - (node.level - 1)] if node.level > 1 else package
                target = base + tuple((node.module or "").split(".")) if node.module else base
            else:
                target = tuple((node.module or "").split("."))
            if target and target[0] == "sparsehydro":
                yield target, node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                target = tuple(alias.name.split("."))
                if target[0] == "sparsehydro":
                    yield target, node.lineno


def test_every_package_has_a_layer():
    unknown = {top_package(m) for _, m in modules()} - set(LAYERS) - {None}
    assert not unknown, f"add these packages to LAYERS: {sorted(unknown)}"


def test_no_upward_imports():
    problems = []
    for path, module in modules():
        source = top_package(module)
        if source is None:
            continue  # the package root may expose anything
        for target, line in runtime_imports(path, module):
            dest = top_package(target)
            if dest is None or dest == source:
                continue
            if LAYERS[dest] > LAYERS[source] and (source, dest) not in ALLOWED:
                problems.append(f"{path.relative_to(PACKAGE)}:{line} ({source}) imports {'.'.join(target)} ({dest})")
    assert not problems, "upward imports:\n" + "\n".join(problems)


def test_nothing_imports_the_study_package():
    problems = [
        f"{path.relative_to(PACKAGE)}:{line}"
        for path, module in modules() if top_package(module) != "studies"
        for target, line in runtime_imports(path, module) if top_package(target) == "studies"
    ]
    assert not problems, "only sparsehydro.studies may import sparsehydro.studies:\n" + "\n".join(problems)


#: __init__ files that may define functions/classes, with the reason.
INIT_EXCEPTIONS = {
    # Fallback stubs that raise a helpful ImportError when plotly is missing.
    "visualization/__init__.py",
}


@pytest.mark.parametrize("path", sorted(PACKAGE.rglob("__init__.py")), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_init_files_only_re_export(path):
    rel = str(path.relative_to(PACKAGE)).replace("\\", "/")
    if rel in INIT_EXCEPTIONS:
        return
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert not defs, f"{rel} defines {defs}; move the logic into a module and re-export it"


def test_importing_the_package_stays_light():
    code = ("import sys, sparsehydro, sparsehydro.models; "
            "print(sorted({m.split('.')[1] for m in sys.modules if m.startswith('sparsehydro.')}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=PACKAGE.parent, check=True).stdout
    loaded = set(ast.literal_eval(out.strip().splitlines()[-1]))
    heavy = loaded & {"calibration", "events", "filters", "metrics", "visualization", "viewer", "studies"}
    assert not heavy, f"import sparsehydro loads {sorted(heavy)}"
