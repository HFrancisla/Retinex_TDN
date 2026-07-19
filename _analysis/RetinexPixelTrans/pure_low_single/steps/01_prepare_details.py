#!/usr/bin/env python3
"""Step 01: ensure detailed per-image decomposition analysis exists."""

from __future__ import annotations

import argparse
import subprocess
import sys

from pure_single_steps_common import (
    COMPARE_ANALYZER,
    RESULT_ROOT,
    ROOT,
    add_image_set_args,
    analyzer_args_for_image_set,
    component_counts,
    details_path,
    discover_runs,
    ensure_output_dirs,
    read_csv,
    relative,
    report_path,
    resolve_image_set,
    selected_image_set,
    write_csv,
)


REQUIRED_COLUMNS = {
    "self_low_psnr",
    "r_low_tv_to_input",
    "l_low_input_gray_corr",
    "anchor_abs_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_image_set_args(parser)
    parser.add_argument("--force", action="store_true")
    default_python = ROOT / ".venv" / "bin" / "python3"
    parser.add_argument("--python", default=str(default_python) if default_python.is_file() else sys.executable)
    return parser.parse_args()


def has_required_schema(path, image_set: str) -> bool:
    if not path.is_file():
        return False
    try:
        rows = read_csv(path)
    except Exception:
        return False
    if not rows:
        return False
    if not REQUIRED_COLUMNS.issubset(set(rows[0].keys())):
        return False
    if "image_set" in rows[0]:
        return any(row.get("image_set") == image_set for row in rows)
    return image_set.isdigit()


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    image_set = selected_image_set(args)
    rows = []
    failures = 0
    for run_dir in discover_runs():
        resolved_image_set = resolve_image_set(run_dir, image_set)
        counts = component_counts(run_dir, resolved_image_set)
        if counts["R_low"] == 0 or counts["L_low"] != counts["R_low"]:
            rows.append(
                {
                    "run": run_dir.name,
                    "status": "skip_incomplete",
                    "requested_image_set": image_set,
                    "target_image_set": resolved_image_set,
                    "returncode": "",
                    "R_low_count": counts["R_low"],
                    "L_low_count": counts["L_low"],
                    "stdout": "",
                    "stderr": "",
                }
            )
            print(f"[skip incomplete] {relative(run_dir)}")
            continue
        force_this_run = args.force or not has_required_schema(details_path(run_dir), resolved_image_set)
        command = [
            args.python,
            str(COMPARE_ANALYZER),
            *analyzer_args_for_image_set(args, resolved_image_set),
            "--details",
            str(run_dir),
        ]
        if force_this_run:
            command.insert(2, "--force")
        reason = "force/stale-schema" if force_this_run else "fresh-check"
        print(f"[details] {relative(run_dir)} ({reason})")
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            failures += 1
        rows.append(
            {
                "run": run_dir.name,
                "status": "ok" if proc.returncode == 0 else "failed",
                "requested_image_set": image_set,
                "target_image_set": resolved_image_set,
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
