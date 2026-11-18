"""Release checks run by .github/workflows/publish.yml (also runnable locally).

    python .github/scripts/check_release.py v1.0.0b1 [--prerelease true|false] [--dist dist]

Fails (exit 1) when:
- the tag is not "v" + the version in sparsehydro/__init__.py;
- the GitHub release's pre-release flag does not match the version
  (alpha/beta/rc versions must be marked pre-release, final versions not);
- CHANGELOG.md has no "## [<version>]" section;
- built files in --dist carry a different version.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]


def package_version() -> str:
    text = (ROOT / "sparsehydro" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        raise SystemExit("could not find __version__ in sparsehydro/__init__.py")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--prerelease", choices=["true", "false"])
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()

    version = package_version()
    problems = []
    if args.tag != f"v{version}":
        problems.append(f"tag {args.tag!r} does not match package version {version!r} (expected 'v{version}')")
    if args.prerelease is not None:
        is_pre = Version(version).is_prerelease
        if is_pre != (args.prerelease == "true"):
            want = "ticked" if is_pre else "unticked"
            problems.append(f"version {version} needs 'Set as a pre-release' {want} on the GitHub release")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## [{version}]" not in changelog:
        problems.append(f"CHANGELOG.md has no '## [{version}]' section")
    if args.dist is not None:
        files = sorted(args.dist.glob("sparsehydro-*"))
        if not files:
            problems.append(f"no built files in {args.dist}")
        for f in files:
            if not f.name.startswith(f"sparsehydro-{version}"):
                problems.append(f"{f.name} does not carry version {version}")

    for p in problems:
        print(f"::error::{p}")
    if not problems:
        print(f"release checks passed for {version}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
