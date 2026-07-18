#!/usr/bin/env python3
"""Step 00: inventory pure-low-single runs and artifact completeness."""

from __future__ import annotations

import argparse

from pure_single_steps_common import (
    RESULT_ROOT,
    component_counts,
    config_fingerprint,
    details_path,
    discover_runs,
    ensure_output_dirs,
    full_validation_metrics,
    iteration_dirs,
    markdown_table,
    relative,
    report_path,
    run_config,
    run_label,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    rows = []
    for run_dir in discover_runs():
        config = run_config(run_dir)
        fingerprint = config_fingerprint(config, run_dir.name)
        iterations = iteration_dirs(run_dir)
        counts = component_counts(run_dir, args.iteration)
        n_expected = max(counts.values()) if counts else 0
        complete_components = n_expected > 0 and counts["R_low"] == n_expected and counts["L_low"] == n_expected
        synthesis_dir = run_dir / "synthesis" / str(args.iteration)
        row = {
            "run": run_dir.name,
            "label": run_label(run_dir, config),
            "path": relative(run_dir),
            **fingerprint,
            "available_iterations": ",".join(map(str, iterations)),
            "target_iteration": args.iteration,
            "has_target_iteration": args.iteration in iterations,
            "R_low_count": counts["R_low"],
            "L_low_count": counts["L_low"],
            "components_complete": complete_components,
            "has_config": (run_dir / "config.yaml").is_file(),
            "has_train_log": (run_dir / "train.log").is_file(),
            "has_synthesis": synthesis_dir.is_dir(),
            "has_compare_psnr": (run_dir / "synthesis_compare.txt").is_file(),
            "has_decomp_report": report_path(run_dir).is_file(),
            "has_decomp_details": details_path(run_dir).is_file(),
            "has_final_full_validation": (run_dir / "final_full_validation.yaml").is_file(),
            "has_best_model": (run_dir / "weights" / "best_model.pth").is_file(),
            "has_last_model": (run_dir / "weights" / "last_model.pth").is_file(),
            **full_validation_metrics(run_dir),
        }
        row["analysis_ready"] = bool(row["has_target_iteration"] and row["components_complete"])
        rows.append(row)

    write_csv(RESULT_ROOT / "inventory.csv", rows)
    incomplete = [row for row in rows if not row["analysis_ready"]]
    missing_details = [row for row in rows if row["analysis_ready"] and not row["has_decomp_details"]]

    md_lines = [
        "# Step 00 pure-low-single inventory",
        "",
        f"Target iteration: `{args.iteration}`",
        f"Runs discovered: `{len(rows)}`",
        "",
        "## Artifact summary",
        "",
        markdown_table(
            rows,
            [
                ("dataset", "dataset", ""),
                ("run", "run", ""),
                ("label", "label", ""),
                ("iter?", "has_target_iteration", ""),
                ("complete?", "components_complete", ""),
                ("R", "R_low_count", ""),
                ("L", "L_low_count", ""),
                ("synth", "has_synthesis", ""),
                ("details", "has_decomp_details", ""),
                ("full-val", "has_final_full_validation", ""),
            ],
        ),
        "",
        "## Required follow-up",
        "",
    ]
    if incomplete:
        md_lines.extend([
            "These runs are missing target iteration outputs or complete R/L components. Treat them as training/incomplete, not failed analysis:",
            "",
            markdown_table(
                incomplete,
                [
                    ("dataset", "dataset", ""),
                    ("run", "run", ""),
                    ("iter?", "has_target_iteration", ""),
                    ("R", "R_low_count", ""),
                    ("L", "L_low_count", ""),
                    ("available", "available_iterations", ""),
                ],
            ),
            "",
        ])
    if missing_details:
        md_lines.extend(["Run step 01 to generate missing details for:", ""])
        md_lines.extend(f"- `{row['run']}`" for row in missing_details)
        md_lines.append("")
    if not incomplete and not missing_details:
        md_lines.append("No blocking artifact gaps for the default analysis path.")
    (RESULT_ROOT / "inventory.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"saved: {RESULT_ROOT / 'inventory.csv'}")
    print(f"saved: {RESULT_ROOT / 'inventory.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
