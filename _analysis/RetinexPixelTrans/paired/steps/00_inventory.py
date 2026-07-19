#!/usr/bin/env python3
"""Step 00: inventory RetinexPixelTrans paired runs and artifact completeness."""

from __future__ import annotations

import argparse
from pathlib import Path

from paired_steps_common import (
    RESULT_ROOT,
    add_image_set_args,
    component_counts,
    config_fingerprint,
    details_path,
    discover_runs,
    ensure_output_dirs,
    iteration_dirs,
    markdown_table,
    relative,
    report_path,
    run_config,
    run_label,
    selected_image_set,
    synthesis_dir_for_image_set,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_image_set_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    image_set = selected_image_set(args)
    rows = []
    for run_dir in discover_runs():
        config = run_config(run_dir)
        fingerprint = config_fingerprint(config)
        iterations = iteration_dirs(run_dir)
        counts = component_counts(run_dir, image_set)
        n_expected = max(counts.values()) if counts else 0
        complete_components = (
            n_expected > 0
            and all(counts[name] == n_expected for name in ("R_low", "L_low", "R_high", "L_high"))
        )
        synthesis_dir = synthesis_dir_for_image_set(run_dir, image_set)
        row = {
            "run": run_dir.name,
            "label": run_label(run_dir, config),
            "path": relative(run_dir),
            **fingerprint,
            "available_image_sets": ",".join(map(str, iterations)),
            "target_image_set": image_set,
            "has_target_image_set": image_set in iterations,
            "R_low_count": counts["R_low"],
            "L_low_count": counts["L_low"],
            "R_high_count": counts["R_high"],
            "L_high_count": counts["L_high"],
            "components_complete": complete_components,
            "has_config": (run_dir / "config.yaml").is_file(),
            "has_train_log": (run_dir / "train.log").is_file(),
            "has_synthesis": synthesis_dir.is_dir(),
            "has_compare_psnr": (run_dir / "synthesis_compare.txt").is_file(),
            "has_decomp_report": report_path(run_dir).is_file(),
            "has_decomp_details": details_path(run_dir).is_file(),
            "has_best_model": (run_dir / "weights" / "best_model.pth").is_file(),
            "has_last_model": (run_dir / "weights" / "last_model.pth").is_file(),
        }
        rows.append(row)

    output_csv = RESULT_ROOT / "inventory.csv"
    write_csv(output_csv, rows)

    comparable_keys = [
        "model.name",
        "model.dim",
        "data.mode",
        "data.path",
        "data.batch_size",
        "data.crop_size",
        "training.seed",
        "training.max_iterations",
        "loss.mode",
        "loss.recon_weight_high",
        "loss.recon_weight_low",
    ]
    mismatches = []
    if rows:
        for key in comparable_keys:
            values = sorted({str(row.get(key, "")) for row in rows})
            if len(values) > 1:
                mismatches.append(f"- `{key}` differs: {', '.join(values)}")

    incomplete = [
        row for row in rows
        if not row["has_target_image_set"] or not row["components_complete"]
    ]
    missing_details = [row for row in rows if not row["has_decomp_details"]]

    shown_rows = rows
    md_lines = [
        "# Step 00 inventory",
        "",
        f"Target image set: `{image_set}`",
        f"Runs discovered: `{len(rows)}`",
        "",
        "## Comparable-field check",
        "",
        "\n".join(mismatches) if mismatches else "All core comparable fields match across discovered runs.",
        "",
        "## Artifact summary",
        "",
        markdown_table(
            shown_rows,
            [
                ("run", "run", ""),
                ("label", "label", ""),
                ("img set?", "has_target_image_set", ""),
                ("complete?", "components_complete", ""),
                ("Rlow", "R_low_count", ""),
                ("Rhigh", "R_high_count", ""),
                ("synth", "has_synthesis", ""),
                ("report", "has_decomp_report", ""),
                ("details", "has_decomp_details", ""),
            ],
        ),
        "",
        "## Required follow-up",
        "",
    ]
    if incomplete:
        md_lines.extend([
            "The following runs are missing target iteration outputs or complete R/L components:",
            "",
            markdown_table(
                incomplete,
                [
                    ("run", "run", ""),
                    ("img set?", "has_target_image_set", ""),
                    ("Rlow", "R_low_count", ""),
                    ("Llow", "L_low_count", ""),
                    ("Rhigh", "R_high_count", ""),
                    ("Lhigh", "L_high_count", ""),
                ],
            ),
            "",
        ])
    if missing_details:
        md_lines.extend([
            "`decomposition_analysis_details.csv` is missing for at least one run. Run step 01.",
            "",
        ])
    if not incomplete and not missing_details:
        md_lines.append("No blocking artifact gaps for the default analysis path.")
    (RESULT_ROOT / "inventory.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"saved: {output_csv}")
    print(f"saved: {RESULT_ROOT / 'inventory.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
