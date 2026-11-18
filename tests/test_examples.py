"""Every script in examples/ runs to completion on the bundled sample."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((REPO / "examples").glob("*.py"))


def test_examples_exist():
    assert EXAMPLES, "examples/*.py not found"


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script, tmp_path):
    # Run against this checkout, not whatever sparsehydro happens to be installed.
    path = os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": path}
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip()
