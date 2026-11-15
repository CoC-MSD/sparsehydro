"""load_extensions: imports extension modules so their classes register."""

from __future__ import annotations

import importlib.metadata
import logging
import sys
import textwrap

import pytest

import sparsehydro
from sparsehydro.registry import EXTENSION_GROUP, ExtensionReport, load_extensions, registry

# ``sparsehydro.registry`` the attribute is the registry instance; the module is in sys.modules.
registry_module = sys.modules["sparsehydro.registry"]

EXTENSION = textwrap.dedent('''
    from sparsehydro.models.unithydrograph.models import GammaUH
    from sparsehydro.registry import registry


    @registry.register
    class ExtensionGammaUH(GammaUH):
        """A unit hydrograph defined outside sparsehydro."""
        model_name = "{name}"
''')


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Each test sees an empty report and unloaded entry points; registrations are undone."""
    monkeypatch.setattr(registry_module, "_report", ExtensionReport())
    monkeypatch.setattr(registry_module, "_entry_points_loaded", set())
    before = set(registry.names())
    modules = set(sys.modules)
    yield
    for name in set(registry.names()) - before:
        registry.unregister(name)
    for name in set(sys.modules) - modules:
        if name.startswith(("sparsehydro_extension_", "ext_pkg")):
            del sys.modules[name]


def _write(tmp_path, stem: str, model_name: str):
    path = tmp_path / f"{stem}.py"
    path.write_text(EXTENSION.format(name=model_name), encoding="utf-8")
    return path


class _EntryPoint:
    def __init__(self, name: str, value: str) -> None:
        self.name, self.value = name, value

    def load(self):
        return importlib.import_module(self.value)


def test_python_file_registers_its_models(tmp_path) -> None:
    path = _write(tmp_path, "my_uh", "ext-file-uh")
    report = load_extensions([path], entry_points=False)
    assert report.loaded == [str(path)] and not report.failed
    model = registry.create("ext-file-uh")
    model.initialize()
    assert "theta" in model.scalar_parameter_names


def test_module_name_and_entry_points(tmp_path, monkeypatch) -> None:
    pkg = tmp_path / "ext_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text(EXTENSION.format(name="ext-ep-uh"), encoding="utf-8")
    (pkg / "more.py").write_text(EXTENSION.format(name="ext-module-uh"), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []

    def fake_entry_points(*, group):
        seen.append(group)
        return [_EntryPoint("demo", "ext_pkg.models")]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    report = load_extensions(["ext_pkg.more"])
    assert seen == [EXTENSION_GROUP]
    assert report.loaded == ["demo (ext_pkg.models)", "ext_pkg.more"]
    assert {"ext-ep-uh", "ext-module-uh"} <= set(registry.names())
    # entry points and modules are imported once per process
    assert load_extensions(["ext_pkg.more"]).loaded == report.loaded and seen == [EXTENSION_GROUP]


def test_failures_are_reported_and_logged(tmp_path, caplog) -> None:
    bad = tmp_path / "broken.py"
    bad.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    good = _write(tmp_path, "fine", "ext-after-failure-uh")
    with caplog.at_level(logging.WARNING, logger="sparsehydro.registry"):
        report = load_extensions([bad, "no_such_extension_module", good], entry_points=False)
    failed = dict(report.failed)
    assert "RuntimeError: boom" == failed[str(bad)]
    assert failed["no_such_extension_module"].startswith("ModuleNotFoundError")
    assert report.loaded == [str(good)]  # one broken extension does not stop the others
    assert "boom" in caplog.text and "sparsehydro_extension_broken" not in sys.modules
    with pytest.raises(FileNotFoundError):
        load_extensions([tmp_path / "missing.py"], entry_points=False, strict=True)


def test_exported_at_top_level() -> None:
    assert sparsehydro.load_extensions is load_extensions
