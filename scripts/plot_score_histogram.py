#!/usr/bin/env python3
"""Plot score bucket histogram from NavSim-style eval CSV (score column)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def score_bucket(s: float) -> str:
    if s == 0:
        return "0"
    if s <= 0.5:
        return "(0, 0.5]"
    if s <= 0.7:
        return "(0.5, 0.7]"
    if s <= 0.8:
        return "(0.7, 0.8]"
    if s <= 0.85:
        return "(0.8, 0.85]"
    if s <= 0.9:
        return "(0.85, 0.9]"
    if s <= 0.95:
        return "(0.9, 0.95]"
    if s < 1.0:
        return "(0.95, 1.0)"
    return "1.0"


BUCKET_ORDER = [
    "0",
    "(0, 0.5]",
    "(0.5, 0.7]",
    "(0.7, 0.8]",
    "(0.8, 0.85]",
    "(0.85, 0.9]",
    "(0.9, 0.95]",
    "(0.95, 1.0)",
    "1.0",
]

DEFAULT_CSV = Path(
    "/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive_hdmap/exp/recogdrive_qwen3_hdmap_ddim_IL_eval/2026.06.25.10.01.35/2026.06.25.10.12.38.csv"
)
DEFAULT_OUTPUT = Path("vis_result_statistics/eval/0625_hdmap/score_RL.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Histogram of score buckets from eval CSV.")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=DEFAULT_CSV,
        help=f"Path to CSV with a 'score' column. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Save figure to this path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI when saving.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "score" not in df.columns:
        raise SystemExit(f"Column 'score' not found in {args.csv}")

    n = len(df)
    buckets = df["score"].astype(float).map(score_bucket)
    counts = buckets.value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int)
    pct = 100.0 * counts / n

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(BUCKET_ORDER))
    ax.bar(x, counts.values, color="steelblue", edgecolor="white", linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(BUCKET_ORDER, rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.set_xlabel("score bucket")
    ax.set_title(f"Score distribution")

    ymax = counts.max()
    ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)

    for i, (c, p) in enumerate(zip(counts.values, pct.values)):
        if c == 0:
            continue
        ax.text(
            i,
            c + ymax * 0.015,
            f"{int(c)}\n({p:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

