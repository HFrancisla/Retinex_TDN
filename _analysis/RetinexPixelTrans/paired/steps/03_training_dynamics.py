#!/usr/bin/env python3
"""Step 03: parse eval training dynamics from train.log."""

from __future__ import annotations

import argparse
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paired_steps_common import (
    FIG_ROOT,
    RESULT_ROOT,
    discover_runs,
    ensure_output_dirs,
    run_config,
    run_label,
    write_csv,
)


EVAL_RE = re.compile(r"\[eval\s+step\s+(\d+)\](.*)$")


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def number_after(line: str, patterns: list[str]) -> float:
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return float(match.group(1))
    return math.nan


def parse_log(run_dir) -> list[dict]:
    path = run_dir / "train.log"
    if not path.is_file():
        return []
    config = run_config(run_dir)
    label = run_label(run_dir, config)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVAL_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        tail = match.group(2)
        rows.append(
            {
                "run": run_dir.name,
                "label": label,
                "step": step,
                "total_weighted": number_after(tail, [r"total:\s*([-\d.eE+]+)"]),
                "recon_weighted": number_after(tail, [r"recon(?:\(weighted\))?:\s*([-\d.eE+]+)"]),
                "cross_recon_weighted": number_after(tail, [r"cross_recon(?:\(weighted\))?:\s*([-\d.eE+]+)"]),
                "smooth_weighted": number_after(tail, [r"smooth(?:\(weighted\))?:\s*([-\d.eE+]+)"]),
                "equal_r_weighted": number_after(tail, [r"equal_r(?:\(weighted\))?:\s*([-\d.eE+]+)"]),
                "r_psnr_proxy": number_after(tail, [r"PSNR\(proxy\):\s*([-\d.eE+]+)dB"]),
            }
        )
    return rows


def make_figure(rows: list[dict]) -> None:
    if not rows:
        return
    by_run: dict[str, list[dict]] = {}
    for row in rows:
        by_run.setdefault(str(row["run"]), []).append(row)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=160)
    for run, run_rows in sorted(by_run.items()):
        run_rows = sorted(run_rows, key=lambda item: int(item["step"]))
        label = str(run_rows[0]["label"])
        steps = [int(row["step"]) for row in run_rows]
        axes[0].plot(steps, [float(row["total_weighted"]) for row in run_rows], linewidth=1.2, label=label)
        axes[1].plot(steps, [float(row["r_psnr_proxy"]) for row in run_rows], linewidth=1.2, label=label)
    axes[0].set_title("eval total weighted loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[1].set_title("eval R low/high PSNR proxy")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("dB")
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    output = FIG_ROOT / "training_total_proxy.png"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parse_args()
    ensure_output_dirs()
    rows = []
    for run_dir in discover_runs():
        rows.extend(parse_log(run_dir))
    output = RESULT_ROOT / "training_eval.csv"
    write_csv(output, rows)
    make_figure(rows)

    by_run: dict[str, list[dict]] = {}
    for row in rows:
        by_run.setdefault(str(row["run"]), []).append(row)

    summary = []
    for run, run_rows in sorted(by_run.items()):
        run_rows = sorted(run_rows, key=lambda item: int(item["step"]))
        if not run_rows:
            continue
        start = run_rows[0]
        end = run_rows[-1]
        summary.append(
            {
                "run": run,
                "label": start["label"],
                "first_step": start["step"],
                "last_step": end["step"],
                "total_delta": float(end["total_weighted"]) - float(start["total_weighted"]),
                "r_proxy_delta": float(end["r_psnr_proxy"]) - float(start["r_psnr_proxy"]),
                "last_total": end["total_weighted"],
                "last_r_proxy": end["r_psnr_proxy"],
            }
        )
    collapse = [
        row for row in summary
        if float(row["total_delta"]) < 0 and float(row["r_proxy_delta"]) < -10
    ]
    md_lines = [
        "# Step 03 training dynamics",
        "",
        f"Eval rows parsed: `{len(rows)}`",
        "",
        "A run is flagged when eval total loss decreases while the R consistency proxy drops by more than 10 dB.",
        "",
    ]
    if collapse:
        md_lines.extend(["## Potential loss/R-consistency mismatch", ""])
        for row in collapse:
            md_lines.append(
                f"- `{row['run']}`: total Δ={float(row['total_delta']):.4f}, "
                f"R proxy Δ={float(row['r_proxy_delta']):.2f} dB"
            )
    else:
        md_lines.append("No run matched the simple collapse heuristic.")
    md_lines.extend(["", f"Figure: `{FIG_ROOT / 'training_total_proxy.png'}`", ""])
    (RESULT_ROOT / "training_dynamics.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"saved: {output}")
    print(f"saved: {RESULT_ROOT / 'training_dynamics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
