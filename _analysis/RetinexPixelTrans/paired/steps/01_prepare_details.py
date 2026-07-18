#!/usr/bin/env python3
"""Step 01: ensure per-image decomposition details exist for every paired run."""

from __future__ import annotations

import argparse
import subprocess
import sys

from paired_steps_common import (
    COMPARE_ANALYZER,
    ROOT,
    RESULT_ROOT,
    details_path,
    discover_runs,
    ensure_output_dirs,
    relative,
    report_path,
    write_csv,
)

REQUIRED_DETAIL_COLUMNS = {
    "r_low_highref_psnr",
    "r_high_highref_psnr",
    "r_low_highref_mean_ratio",
    "r_low_highref_overbright_010",
    "l_low_input_gray_corr",
    "l_high_input_gray_corr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    default_python = ROOT / ".venv" / "bin" / "python3"
    parser.add_argument(
        "--python",
        default=str(default_python) if default_python.is_file() else sys.executable,
    )
    return parser.parse_args()


def details_have_required_columns(path) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().strip().split(",")
    return REQUIRED_DETAIL_COLUMNS.issubset(set(header))


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    rows = []
    failures = 0
    for run_dir in discover_runs():
        force_this_run = args.force or not details_have_required_columns(details_path(run_dir))
        command = [
            args.python,
            str(COMPARE_ANALYZER),
            "--iteration",
            str(args.iteration),
            "--details",
            str(run_dir),
        ]
        if force_this_run:
            command.insert(2, "--force")
        reason = "force/stale-schema" if force_this_run else "fresh-check"
        print(f"[details] {relative(run_dir)} ({reason})")
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ok = proc.returncode == 0
        if not ok:
            failures += 1
        rows.append(
            {
                "run": run_dir.name,
                "returncode": proc.returncode,
                "report": relative(report_path(run_dir)),
                "details": relative(details_path(run_dir)),
                "has_report": report_path(run_dir).is_file(),
                "has_details": details_path(run_dir).is_file(),
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )
        if proc.stdout.strip():
            print(proc.stdout.strip())
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
    output = RESULT_ROOT / "prepare_details_log.csv"
    write_csv(output, rows)
    print(f"saved: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
