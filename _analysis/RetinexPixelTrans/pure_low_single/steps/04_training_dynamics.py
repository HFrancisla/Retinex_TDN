#!/usr/bin/env python3
"""Step 04: parse train.log eval curves and full-validation summaries."""

from __future__ import annotations

import math
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pure_single_steps_common import (
    FIG_ROOT,
    RESULT_ROOT,
    discover_runs,
    ensure_output_dirs,
    full_validation_metrics,
    run_config,
    run_label,
    write_csv,
)


EVAL_RE = re.compile(r"\[eval\s+step\s+(\d+)\]\s+Loss\(weighted\):\s*(.*)$")


def number_after(line: str, key: str) -> float:
    match = re.search(rf"{key}:\s*([-\d.eE+]+)", line)
    return float(match.group(1)) if match else math.nan


def parse_log(run_dir) -> list[dict]:
    path = run_dir / "train.log"
    if not path.is_file():
        return []
    config = run_config(run_dir)
    loss = config.get("loss", {}) or {}
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVAL_RE.search(line)
        if not match:
            continue
        tail = match.group(2)
        rows.append(
            {
                "run": run_dir.name,
                "label": run_label(run_dir, config),
                "dataset": str((config.get("data", {}) or {}).get("path", "")),
                "step": int(match.group(1)),
                "recon_weight": loss.get("recon_weight", ""),
                "anchor_weight": loss.get("anchor_weight", ""),
                "bdsp_weight": loss.get("bdsp_weight", ""),
                "smooth_weight": loss.get("smooth_weight", ""),
                "total_weighted": number_after(tail, "total"),
                "recon_weighted": number_after(tail, "recon"),
                "anchor_weighted": number_after(tail, "anchor"),
                "bdsp_weighted": number_after(tail, "bdsp"),
                "smooth_weighted": number_after(tail, "smooth"),
            }
        )
    return rows


def make_figure(rows: list[dict]) -> None:
    if not rows:
        return
    by_run: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_run[str(row["run"])].append(row)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=160)
    for run, run_rows in sorted(by_run.items()):
        run_rows = sorted(run_rows, key=lambda row: int(row["step"]))
        label = str(run_rows[0]["label"])
        steps = [int(row["step"]) for row in run_rows]
        axes[0].plot(steps, [float(row["total_weighted"]) for row in run_rows], linewidth=1.1, label=label)
        axes[1].plot(steps, [float(row["recon_weighted"]) for row in run_rows], linewidth=1.1, label=label)
    axes[0].set_title("eval total weighted loss")
    axes[1].set_title("eval recon weighted loss")
    for axis in axes:
        axis.set_xlabel("iteration")
        axis.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    figure.savefig(FIG_ROOT / "training_loss_curves.png", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    ensure_output_dirs()
    rows = []
    full_rows = []
    for run_dir in discover_runs():
        config = run_config(run_dir)
        full = full_validation_metrics(run_dir)
        if full:
            full_rows.append({"run": run_dir.name, "label": run_label(run_dir, config), **full})
        rows.extend(parse_log(run_dir))
    write_csv(RESULT_ROOT / "training_eval.csv", rows)
    write_csv(RESULT_ROOT / "final_full_validation_summary.csv", full_rows)
    make_figure(rows)

    flagged = []
    for row in full_rows:
        recon_loss = row.get("full_recon_loss", "")
        weighted = row.get("full_recon_weighted_loss", "")
        if recon_loss != "" and weighted != "" and float(recon_loss) > float(weighted) * 2:
            flagged.append(row)
    md_lines = [
        "# Step 04 training dynamics",
        "",
        f"Eval rows parsed: `{len(rows)}`",
        f"Full-validation summaries: `{len(full_rows)}`",
        "",
        "Use unweighted `full_recon_loss` when comparing runs with different `recon_weight`; weighted total loss is not directly comparable across recon weights.",
        "",
    ]
    if flagged:
        md_lines.extend(["## Recon-weight comparison warning", ""])
        for row in flagged:
            md_lines.append(
                f"- `{row['run']}`: full recon={float(row['full_recon_loss']):.5f}, "
                f"weighted recon={float(row['full_recon_weighted_loss']):.5f}"
            )
        md_lines.append("")
    md_lines.append(f"Figure: `{FIG_ROOT / 'training_loss_curves.png'}`")
    (RESULT_ROOT / "training_dynamics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"saved: {RESULT_ROOT / 'training_eval.csv'}")
    print(f"saved: {RESULT_ROOT / 'final_full_validation_summary.csv'}")
    print(f"saved: {RESULT_ROOT / 'training_dynamics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
