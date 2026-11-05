#!/usr/bin/env python3
"""Install one dependency profile from the repository's requirements.txt."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def read_profiles(path: Path) -> dict[str, list[str]]:
    profiles: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"# \[profile:([a-z0-9-]+)\]", line)
        if match:
            current = match.group(1)
            if current in profiles:
                raise ValueError(f"Duplicate requirements profile: {current}")
            profiles[current] = []
        elif current is not None:
            if line.startswith("#| "):
                profiles[current].append(line[3:])
            elif line.strip() and not line.lstrip().startswith("#"):
                profiles[current].append(line)
    return profiles


def main() -> None:
    profiles = read_profiles(ROOT / "requirements.txt")
    if not profiles:
        raise SystemExit("requirements.txt has no dependency profiles. Update it together with the setup scripts.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(profiles), required=True)
    parser.add_argument("--python", default=sys.executable, help="Target environment's Python executable")
    parser.add_argument("--installer", choices=["pip", "uv"], default="pip")
    parser.add_argument("--pin", action="append", default=[], help="Additional requirement constraint, repeatable")
    parser.add_argument("--print", action="store_true", dest="print_only", help="Print requirements without installing")
    args = parser.parse_args()
    content = "\n".join(profiles[args.profile]) + "\n"
    if not profiles[args.profile]:
        parser.error(f"Empty dependency profile: {args.profile}")
    if args.print_only:
        print(content, end="")
        return
    if args.profile in {"annotation", "grounded", "robot"} and not (ROOT / "lerobot/pyproject.toml").is_file():
        parser.error("LeRobot submodule is missing; run git submodule update --init --recursive lerobot")
    with tempfile.TemporaryDirectory(prefix="robot101-requirements-") as directory:
        path = Path(directory) / "selected.txt"
        path.write_text(content, encoding="utf-8")
        if args.installer == "uv":
            command = ["uv", "pip", "install", "--python", args.python]
        else:
            command = [args.python, "-m", "pip", "install"]
        subprocess.run(command + ["-r", str(path), *args.pin], cwd=ROOT, check=True)
        if args.profile == "grounded":
            # This inference stack intentionally retains Transformers 4 / Hub <1.
            # Register the pinned LeRobot sources without resolving robot-training extras.
            subprocess.run(command + ["--no-deps", "-e", str(ROOT / "lerobot")], cwd=ROOT, check=True)
        # Make python -m robot101... entrypoints available in the target environment.
        subprocess.run(command + ["--no-deps", "-e", str(ROOT)], cwd=ROOT, check=True)
        # An editable install can succeed while discovering zero packages. Check
        # outside the checkout, without PYTHONPATH masking a packaging mistake.
        subprocess.run([args.python, "-I", "-c",
                        "from pathlib import Path; import robot101; "
                        f"assert Path(robot101.__file__).resolve() == Path({str(ROOT / 'src/robot101/__init__.py')!r}).resolve(), "
                        "'robot101 resolves to a different checkout'; "
                        "print('Installed robot101:', robot101.__file__)"], cwd=directory, check=True)


if __name__ == "__main__":
    main()
