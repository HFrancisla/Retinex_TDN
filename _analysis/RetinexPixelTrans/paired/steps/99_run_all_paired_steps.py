#!/usr/bin/env python3
"""Run all reusable RetinexPixelTrans paired analysis steps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STEP_ROOT = Path(__file__).resolve().parent
ROOT = STEP_ROOT.parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--force", action="store_true", help="force regeneration in step 01")
    parser.add_argument("--skip-details", action="store_true", help="skip step 01 if details are already prepared")
    default_python = ROOT / ".venv" / "bin" / "python3"
    parser.add_argument(
        "--python",
        default=str(default_python) if default_python.is_file() else sys.executable,
    )
    return parser.parse_args()


def run(python: str, script: str, args: list[str]) -> int:
    command = [python, str(STEP_ROOT / script), *args]
    print("")
    print(" ".join(command))
    proc = subprocess.run(command)
    return proc.returncode


def main() -> int:
    args = parse_args()
    common_args = ["--iteration", str(args.iteration)]
    steps = [
        ("00_inventory.py", common_args),
    ]
    if not args.skip_details:
        detail_args = [*common_args]
        if args.force:
            detail_args.append("--force")
        steps.append(("01_prepare_details.py", detail_args))
    steps.extend(
        [
            ("02_summarize_rank.py", common_args),
            ("03_training_dynamics.py", []),
            ("04_make_visual_grids.py", common_args),
            ("00_inventory.py", common_args),
        ]
    )
    for script, script_args in steps:
        status = run(args.python, script, script_args)
        if status != 0:
            print(f"[ERROR] {script} failed with status {status}", file=sys.stderr)
            return status
    print("")
    print(f"Done. Results: {STEP_ROOT / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
