#!/usr/bin/env python3
"""Step 02: summarize and rank pure-low-single runs per dataset."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

from pure_single_steps_common import (
    RESULT_ROOT,
    as_float,
    config_fingerprint,
    details_path,
    discover_runs,
    ensure_output_dirs,
    full_validation_metrics,
    markdown_table,
    read_csv,
    run_config,
    run_label,
    summarize_metric,
    write_csv,
)


METRICS = [
    "input_low_mean",
    "input_low_gray_std",
    "r_low_mean",
    "r_low_std",
    "r_low_gray_std",
    "r_low_bright_095",
    "r_low_dark_005",
    "r_low_mean_gain_vs_input",
    "r_low_contrast_gain_vs_input",
    "r_low_tv_to_input",
    "r_low_input_gray_corr",
    "r_low_input_chroma_l1",
    "l_low_mean",
    "l_low_std",
    "l_low_bright_095",
    "l_low_tv_to_input",
    "l_low_input_gray_corr",
    "self_low_l1",
    "self_low_psnr",
    "anchor_target",
    "anchor_abs_error",
    "input_low_highref_psnr",
    "r_low_highref_l1",
    "r_low_highref_psnr",
    "r_low_highref_ssim",
    "r_low_highref_mean_ratio",
    "r_low_highref_overbright_010",
    "r_low_highref_chroma_l1",
    "r_low_highref_gray_corr",
    "r_low_highref_psnr_gain_vs_input",
]


REQUIRED_COLUMNS = {
    "self_low_psnr",
    "r_low_tv_to_input",
    "l_low_input_gray_corr",
    "anchor_abs_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, default=10000)
    return parser.parse_args()


def add_rank(rows: list[dict], key: str, higher_is_better: bool, rank_key: str) -> None:
    def value(row: dict) -> float:
        result = float(row.get(key, math.nan))
        if not math.isfinite(result):
            return -math.inf if higher_is_better else math.inf
        return result

    ordered = sorted(rows, key=value, reverse=higher_is_better)
    for index, row in enumerate(ordered, start=1):
        row[rank_key] = index


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    summaries = []
    missing = []
    stale = []
    incomplete = []
    for run_dir in discover_runs():
        path = details_path(run_dir)
        config = run_config(run_dir)
        if not path.is_file():
            missing.append(run_dir.name)
            continue
        all_rows = read_csv(path)
        if not all_rows:
            stale.append(run_dir.name)
            continue
        if not REQUIRED_COLUMNS.issubset(set(all_rows[0].keys())):
            stale.append(run_dir.name)
            continue
        rows = [row for row in all_rows if int(float(row.get("iteration", -1))) == args.iteration]
        if not rows:
            incomplete.append(run_dir.name)
            continue
        summary = {
            "run": run_dir.name,
            "label": run_label(run_dir, config),
            "n_images": len(rows),
            **config_fingerprint(config, run_dir.name),
            **full_validation_metrics(run_dir),
        }
        for metric in METRICS:
            if any(metric in row and row[metric] != "" for row in rows):
                summary.update(summarize_metric(rows, metric))
        ratio = float(summary.get("r_low_highref_mean_ratio_mean", math.nan))
        summary["r_low_highref_mean_ratio_abs_err"] = abs(ratio - 1.0) if math.isfinite(ratio) else math.nan
        summaries.append(summary)

    if not summaries:
        raise SystemExit("No usable details files found. Run 01_prepare_details.py --force first.")

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in summaries:
        by_dataset[str(row["dataset"])].append(row)

    ranking = []
    for dataset, rows in sorted(by_dataset.items()):
        add_rank(rows, "self_low_psnr_mean", True, "rank_self_recon")
        add_rank(rows, "r_low_tv_to_input_mean", False, "rank_r_noise")
        add_rank(rows, "l_low_input_gray_corr_mean", False, "rank_l_leakage_corr")
        # Anchor v1/v2 use different targets, so anchor_abs_error is useful as
        # an in-run diagnostic but should not directly rank across versions.
        add_rank(rows, "anchor_abs_error_mean", False, "rank_anchor_diagnostic")
        add_rank(rows, "r_low_bright_095_mean", False, "rank_r_bright")
        add_rank(rows, "full_recon_loss", False, "rank_full_recon")
        if any(math.isfinite(float(row.get("r_low_highref_psnr_mean", math.nan))) for row in rows):
            add_rank(rows, "r_low_highref_psnr_mean", True, "rank_highref")
            add_rank(rows, "r_low_highref_mean_ratio_abs_err", False, "rank_highref_scale")
            add_rank(rows, "r_low_highref_overbright_010_mean", False, "rank_highref_overbright")
        for row in rows:
            has_highref = math.isfinite(float(row.get("r_low_highref_psnr_mean", math.nan)))
            anchor_version = str(row.get("loss.anchor_version", ""))
            row["canonical_anchor_penalty"] = 12 if anchor_version and anchor_version != "v2" else 0
            row["pure_single_rank_score"] = (
                3 * int(row.get("rank_self_recon", len(rows)))
                + 3 * int(row.get("rank_r_noise", len(rows)))
                + 2 * int(row.get("rank_l_leakage_corr", len(rows)))
                + int(row.get("rank_r_bright", len(rows)))
                + (2 * int(row.get("rank_full_recon", len(rows))) if row.get("full_recon_loss", "") != "" else 0)
                + (3 * int(row.get("rank_highref", len(rows))) if has_highref else 0)
                + (2 * int(row.get("rank_highref_scale", len(rows))) if has_highref else 0)
                + (2 * int(row.get("rank_highref_overbright", len(rows))) if has_highref else 0)
                + int(row["canonical_anchor_penalty"])
            )
        ranking.extend(
            sorted(
                rows,
                key=lambda row: (
                    int(row["pure_single_rank_score"]),
                    -float(row.get("self_low_psnr_mean", -math.inf)),
                ),
            )
        )

    write_csv(RESULT_ROOT / "pure_single_summary.csv", summaries)
    write_csv(RESULT_ROOT / "pure_single_ranking.csv", ranking)

    md_lines = [
        "# Step 02 pure-low-single summary",
        "",
        f"Iteration: `{args.iteration}`",
        "",
        "Ranking is computed per dataset. LOLv2 high-reference metrics are diagnostic only; BDDnight is ranked without high-reference metrics.",
        "",
        "`anchor_abs_error` is reported but not used as a cross-version rank term because anchor v1/v2 have different targets. Non-v2 anchor runs get a small canonical-anchor penalty so old ablations do not outrank the current v2 baseline solely on a different anchor definition.",
        "",
    ]
    for dataset, rows in sorted(by_dataset.items()):
        ranked = [row for row in ranking if row["dataset"] == dataset]
        md_lines.extend([
            f"## {dataset} ranking",
            "",
            markdown_table(
                ranked,
                [
                    ("score", "pure_single_rank_score", ".0f"),
                    ("run", "run", ""),
                    ("label", "label", ""),
                    ("self PSNR", "self_low_psnr_mean", ".2f"),
                    ("R TV/input", "r_low_tv_to_input_mean", ".2f"),
                    ("corr(L,I)", "l_low_input_gray_corr_mean", ".3f"),
                    ("anchor err", "anchor_abs_error_mean", ".4f"),
                    ("R>0.95", "r_low_bright_095_mean", ".3f"),
                    ("anchor penalty", "canonical_anchor_penalty", ".0f"),
                    ("R→high PSNR", "r_low_highref_psnr_mean", ".2f"),
                    ("R/high", "r_low_highref_mean_ratio_mean", ".2f"),
                    ("full recon", "full_recon_loss", ".5f"),
                ],
            ),
            "",
        ])
        best = ranked[0]
        md_lines.append(f"Dataset verdict: current conservative top is `{best['run']}` (`{best['label']}`).")
        md_lines.append("")

    if missing:
        md_lines.extend(["## Missing details", "", *[f"- `{item}`" for item in missing], ""])
    if stale:
        md_lines.extend([
            "## Stale details schema",
            "",
            "These details lack required pure-low-single diagnostic columns. Run `01_prepare_details.py --force`.",
            "",
            *[f"- `{item}`" for item in stale],
            "",
        ])
    if incomplete:
        md_lines.extend(["## Missing target iteration rows", "", *[f"- `{item}`" for item in incomplete], ""])
    (RESULT_ROOT / "pure_single_analysis.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"saved: {RESULT_ROOT / 'pure_single_summary.csv'}")
    print(f"saved: {RESULT_ROOT / 'pure_single_ranking.csv'}")
    print(f"saved: {RESULT_ROOT / 'pure_single_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
