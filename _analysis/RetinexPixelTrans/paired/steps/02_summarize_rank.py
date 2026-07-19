#!/usr/bin/env python3
"""Step 02: corrected paired summary and ranking.

The ranking is deliberately high-reference aware.  R_low/R_high consistency is
reported, but it is not allowed to dominate cases where both R images are
jointly over-bright or poorly scaled.
"""

from __future__ import annotations

import argparse
import math

from paired_steps_common import (
    RESULT_ROOT,
    add_image_set_args,
    as_float,
    config_fingerprint,
    detail_rows_for_image_set,
    details_path,
    discover_runs,
    ensure_output_dirs,
    markdown_table,
    read_csv,
    run_config,
    run_label,
    selected_image_set,
    summarize_metric,
    write_csv,
)


METRICS = [
    "input_low_highref_psnr",
    "r_low_highref_l1",
    "r_low_highref_psnr",
    "r_low_highref_ssim",
    "r_low_highref_mean_ratio",
    "r_low_highref_overbright_010",
    "r_low_highref_chroma_l1",
    "r_low_highref_gray_corr",
    "r_low_highref_psnr_gain_vs_input",
    "r_high_highref_l1",
    "r_high_highref_psnr",
    "r_consistency_l1",
    "r_consistency_psnr",
    "r_consistency_ssim",
    "r_low_mean",
    "r_high_mean",
    "r_low_std",
    "r_high_std",
    "r_low_bright_095",
    "r_high_bright_095",
    "l_low_mean",
    "l_high_mean",
    "l_low_std",
    "l_high_std",
    "l_low_input_gray_corr",
    "l_high_input_gray_corr",
    "l_low_tv_to_input",
    "l_high_tv_to_input",
    "self_low_psnr",
    "self_high_psnr",
    "cross_low_psnr",
    "cross_high_psnr",
]

REQUIRED_COLUMNS = {
    "r_low_highref_psnr",
    "r_high_highref_psnr",
    "r_low_highref_mean_ratio",
    "r_low_highref_overbright_010",
    "r_consistency_psnr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_image_set_args(parser)
    return parser.parse_args()


def add_rank(rows: list[dict], key: str, higher_is_better: bool, rank_key: str) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            math.isfinite(float(row.get(key, math.nan))),
            float(row.get(key, math.nan)),
        ),
        reverse=higher_is_better,
    )
    if not higher_is_better:
        ordered = sorted(
            rows,
            key=lambda row: (
                not math.isfinite(float(row.get(key, math.nan))),
                float(row.get(key, math.inf)),
            ),
        )
    for index, row in enumerate(ordered, start=1):
        row[rank_key] = index


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    image_set = selected_image_set(args)
    summary_rows = []
    missing = []
    stale = []
    for run_dir in discover_runs():
        path = details_path(run_dir)
        if not path.is_file():
            missing.append(run_dir.name)
            continue
        all_rows = read_csv(path)
        available_columns = set(all_rows[0].keys()) if all_rows else set()
        if not REQUIRED_COLUMNS.issubset(available_columns):
            stale.append(run_dir.name)
            continue
        rows = detail_rows_for_image_set(all_rows, args)
        if not rows:
            missing.append(f"{run_dir.name} (no image set {image_set})")
            continue
        config = run_config(run_dir)
        summary = {
            "run": run_dir.name,
            "label": run_label(run_dir, config),
            "n_images": len(rows),
            **config_fingerprint(config),
        }
        for metric in METRICS:
            if any(metric in row and row[metric] != "" for row in rows):
                summary.update(summarize_metric(rows, metric))
        ratio = float(summary.get("r_low_highref_mean_ratio_mean", math.nan))
        summary["r_low_highref_mean_ratio_abs_err"] = (
            abs(ratio - 1.0) if math.isfinite(ratio) else math.nan
        )
        summary["n_best_r_low_highref_psnr"] = sum(
            1 for row in rows
            if as_float(row, "r_low_highref_psnr") >= max(
                as_float(peer, "r_low_highref_psnr") for peer in rows
            )
        )
        summary_rows.append(summary)

    if not summary_rows:
        raise SystemExit(
            "No usable details files found. Run 01_prepare_details.py --force first."
        )

    add_rank(summary_rows, "r_low_highref_psnr_mean", True, "rank_r_low_highref_psnr")
    add_rank(summary_rows, "r_high_highref_psnr_mean", True, "rank_r_high_highref_psnr")
    add_rank(summary_rows, "r_low_highref_overbright_010_mean", False, "rank_overbright")
    add_rank(summary_rows, "r_low_highref_mean_ratio_abs_err", False, "rank_mean_ratio")
    add_rank(summary_rows, "r_consistency_psnr_mean", True, "rank_consistency")

    for row in summary_rows:
        # Primary corrected score: absolute R fidelity dominates, consistency is secondary.
        row["corrected_rank_score"] = (
            4 * int(row["rank_r_low_highref_psnr"])
            + 2 * int(row["rank_r_high_highref_psnr"])
            + 2 * int(row["rank_overbright"])
            + 2 * int(row["rank_mean_ratio"])
            + int(row["rank_consistency"])
        )

    ranking = sorted(
        summary_rows,
        key=lambda row: (
            int(row["corrected_rank_score"]),
            -float(row.get("r_low_highref_psnr_mean", -math.inf)),
        ),
    )
    consistency_ranking = sorted(
        summary_rows,
        key=lambda row: -float(row.get("r_consistency_psnr_mean", -math.inf)),
    )

    summary_csv = RESULT_ROOT / "corrected_summary.csv"
    ranking_csv = RESULT_ROOT / "corrected_ranking.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(ranking_csv, ranking)

    top = ranking[0]
    consistency_top = consistency_ranking[0]
    md_lines = [
        "# Step 02 corrected paired analysis",
        "",
        f"Image set: `{image_set}`",
        "",
        "Primary rule: rank by how close `R_low` and `R_high` are to matched `I_high`, then use `R_low/R_high` consistency as a secondary criterion.",
        "",
        "This avoids the known failure mode where `R_low≈R_high≈over-bright` receives a high consistency score.",
        "",
        "## Corrected top runs",
        "",
        markdown_table(
            ranking[:10],
            [
                ("rank_score", "corrected_rank_score", ".0f"),
                ("run", "run", ""),
                ("label", "label", ""),
                ("Rlow→high PSNR", "r_low_highref_psnr_mean", ".2f"),
                ("Rhigh→high PSNR", "r_high_highref_psnr_mean", ".2f"),
                ("R/high", "r_low_highref_mean_ratio_mean", ".2f"),
                (">high+0.1", "r_low_highref_overbright_010_mean", ".3f"),
                ("Rlow/Rhigh PSNR", "r_consistency_psnr_mean", ".2f"),
                ("Lhigh mean", "l_high_mean_mean", ".3f"),
                ("corr(Llow,I)", "l_low_input_gray_corr_mean", ".3f"),
                ("corr(Lhigh,I)", "l_high_input_gray_corr_mean", ".3f"),
            ],
        ),
        "",
        "## Consistency-only top runs",
        "",
        markdown_table(
            consistency_ranking[:10],
            [
                ("run", "run", ""),
                ("label", "label", ""),
                ("Rlow/Rhigh PSNR", "r_consistency_psnr_mean", ".2f"),
                ("Rlow→high PSNR", "r_low_highref_psnr_mean", ".2f"),
                ("R/high", "r_low_highref_mean_ratio_mean", ".2f"),
                (">high+0.1", "r_low_highref_overbright_010_mean", ".3f"),
            ],
        ),
        "",
        "## Main verdict",
        "",
        f"- Corrected best run: `{top['run']}` (`{top['label']}`).",
        f"- Consistency-only best run: `{consistency_top['run']}` (`{consistency_top['label']}`).",
    ]
    if top["run"] != consistency_top["run"]:
        md_lines.append(
            "- The two best runs differ, so the old consistency-only ranking is not sufficient for this batch."
        )
    else:
        md_lines.append(
            "- The corrected and consistency-only best run match for this batch; still inspect high-reference and L-leakage columns before accepting it."
        )
    if missing:
        md_lines.extend(["", "## Missing details", "", *[f"- {item}" for item in missing]])
    if stale:
        md_lines.extend([
            "",
            "## Stale details schema",
            "",
            "These details files lack required high-reference columns. Run `01_prepare_details.py --force`.",
            "",
            *[f"- {item}" for item in stale],
        ])
    md_lines.append("")
    (RESULT_ROOT / "corrected_analysis.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"saved: {summary_csv}")
    print(f"saved: {ranking_csv}")
    print(f"saved: {RESULT_ROOT / 'corrected_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
