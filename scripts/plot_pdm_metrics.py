#!/usr/bin/env python3
"""Summarize PDM metrics and plot bucket histograms from an eval CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CSV = Path(
    "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/exp/recogdrive_qwen3_RL_eval/2026.06.23.09.01.52/2026.06.23.09.14.16.csv"
)
DEFAULT_OUTPUT_DIR = Path("vis_output/pdm_stats/2026.06.23.09.01.52")

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

METRIC_COLUMNS = [
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "obstacle_clearance",
    "centerline_convergence",
    "start_anchor_comfort",
]

DISPLAY_NAMES = {
    "score": "score",
    "no_at_fault_collisions": "no_at_fault_collisions",
    "drivable_area_compliance": "drivable_area_compliance",
    "ego_progress": "ego_progress",
    "time_to_collision_within_bound": "time_to_collision_within_bound",
    "comfort": "comfort",
    "driving_direction_compliance": "driving_direction_compliance",
    "obstacle_clearance": "obstacle_clearance",
    "centerline_convergence": "centerline_convergence",
    "start_anchor_comfort": "start_anchor_comfort",
}


def score_bucket(score: float) -> str:
    if score == 0:
        return "0"
    if score <= 0.5:
        return "(0, 0.5]"
    if score <= 0.7:
        return "(0.5, 0.7]"
    if score <= 0.8:
        return "(0.7, 0.8]"
    if score <= 0.85:
        return "(0.8, 0.85]"
    if score <= 0.9:
        return "(0.85, 0.9]"
    if score <= 0.95:
        return "(0.9, 0.95]"
    if score < 1.0:
        return "(0.95, 1.0)"
    return "1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and plot PDM metrics.")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=DEFAULT_CSV,
        help=f"Eval CSV to analyze. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--scene-map-csv",
        type=Path,
        default=None,
        help="Optional token-to-scene CSV. When set, write scene-wise PDM summaries and plots.",
    )
    parser.add_argument(
        "--scene-column",
        default="scene_merge",
        help="Scene grouping column in --scene-map-csv. Default: scene_merge.",
    )
    parser.add_argument(
        "--token-column",
        default="token",
        help="Token column shared by eval CSV and scene map CSV. Default: token.",
    )
    parser.add_argument(
        "--missing-scene-label",
        default="lane_keep",
        help="Scene label assigned to eval tokens missing from --scene-map-csv. Default: lane_keep.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    return parser.parse_args()


def load_eval_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [column for column in METRIC_COLUMNS if column not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns {missing}: {csv_path}")

    if "token" in df.columns:
        df = df[df["token"].astype(str) != "average"].copy()
    return df


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "mean": df[METRIC_COLUMNS].mean(numeric_only=True),
            "std": df[METRIC_COLUMNS].std(numeric_only=True),
            "min": df[METRIC_COLUMNS].min(numeric_only=True),
            "max": df[METRIC_COLUMNS].max(numeric_only=True),
        }
    )
    summary.index.name = "metric"
    return summary


def attach_scene_labels(
    eval_df: pd.DataFrame,
    scene_df: pd.DataFrame,
    scene_column: str = "scene_merge",
    token_column: str = "token",
    missing_scene_label: str = "lane_keep",
) -> pd.DataFrame:
    if token_column not in eval_df.columns:
        raise ValueError(f"Eval CSV missing token column: {token_column}")
    missing_scene_columns = [column for column in [token_column, scene_column] if column not in scene_df.columns]
    if missing_scene_columns:
        raise ValueError(f"Scene map CSV missing columns: {missing_scene_columns}")

    scene_labels = scene_df[[token_column, scene_column]].drop_duplicates(subset=[token_column])
    merged = eval_df.merge(scene_labels, on=token_column, how="left")
    merged[scene_column] = merged[scene_column].fillna(missing_scene_label).astype(str)
    return merged


def summarize_metrics_by_scene(df: pd.DataFrame, scene_column: str = "scene_merge") -> pd.DataFrame:
    if scene_column not in df.columns:
        raise ValueError(f"DataFrame missing scene column: {scene_column}")

    rows = []
    group_order = df[scene_column].value_counts().index.tolist()
    for scene_name in group_order:
        group = df[df[scene_column] == scene_name]
        row = {scene_column: scene_name, "sample_count": int(len(group))}
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        rows.append(row)

    total_row = {scene_column: "all", "sample_count": int(len(df))}
    for metric in METRIC_COLUMNS:
        total_row[f"{metric}_mean"] = float(pd.to_numeric(df[metric], errors="coerce").mean())
    rows.append(total_row)

    return pd.DataFrame(rows)


def _get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for PNG plots. Install matplotlib or use the CSV/Markdown outputs."
        ) from exc

    return plt


def plot_bucket_histograms(df: pd.DataFrame, columns: Iterable[str], output_path: Path, dpi: int) -> None:
    plt = _get_pyplot()
    columns = list(columns)
    ncols = 2
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4.2 * nrows))
    axes = axes.flatten()
    total = len(df)

    for ax, column in zip(axes, columns):
        buckets = df[column].astype(float).map(score_bucket)
        counts = buckets.value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int)
        pct = 100.0 * counts / total

        x = range(len(BUCKET_ORDER))
        ax.bar(x, counts.values, color="steelblue", edgecolor="white", linewidth=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(BUCKET_ORDER, rotation=35, ha="right")
        ax.set_title(DISPLAY_NAMES[column])
        ax.set_xlabel("score bucket")
        ax.set_ylabel("Count")

        ymax = int(counts.max()) if len(counts) else 0
        ax.set_ylim(0, ymax * 1.16 if ymax > 0 else 1)
        for idx, (count, fraction) in enumerate(zip(counts.values, pct.values)):
            if count == 0:
                continue
            ax.text(
                idx,
                count + max(ymax * 0.015, 0.5),
                f"{int(count)}\n({fraction:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.suptitle("PDM metric bucket distributions", y=1.01, fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_score_histogram(df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    plt = _get_pyplot()
    series = df["score"].astype(float)
    buckets = series.map(score_bucket)
    counts = buckets.value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int)
    total = len(series)
    pct = 100.0 * counts / total

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(BUCKET_ORDER))
    ax.bar(x, counts.values, color="steelblue", edgecolor="white", linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(BUCKET_ORDER, rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.set_xlabel("score bucket")
    ax.set_title("Score distribution")

    ymax = int(counts.max()) if len(counts) else 0
    ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)
    for idx, (count, fraction) in enumerate(zip(counts.values, pct.values)):
        if count == 0:
            continue
        ax.text(
            idx,
            count + max(ymax * 0.015, 0.5),
            f"{int(count)}\n({fraction:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_scene_score_means(scene_summary: pd.DataFrame, scene_column: str, output_path: Path, dpi: int) -> None:
    plt = _get_pyplot()
    plot_df = scene_summary[scene_summary[scene_column] != "all"].copy()
    plot_df = plot_df.sort_values("score_mean", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(plot_df))
    ax.bar(x, plot_df["score_mean"].values, color="steelblue", edgecolor="white", linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df[scene_column].tolist(), rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean score")
    ax.set_xlabel(scene_column)
    ax.set_title("Mean PDM score by scene")

    for idx, row in enumerate(plot_df.itertuples(index=False)):
        score_mean = getattr(row, "score_mean")
        sample_count = getattr(row, "sample_count")
        ax.text(idx, score_mean + 0.015, f"{score_mean:.3f}\nn={sample_count}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_scene_metric_heatmap(scene_summary: pd.DataFrame, scene_column: str, output_path: Path, dpi: int) -> None:
    plt = _get_pyplot()
    plot_df = scene_summary[scene_summary[scene_column] != "all"].copy()
    plot_df = plot_df.set_index(scene_column)
    values = plot_df[[f"{metric}_mean" for metric in METRIC_COLUMNS]]
    labels = [column.removesuffix("_mean") for column in values.columns]

    fig_width = max(12, 0.8 * len(labels))
    fig_height = max(4, 0.6 * len(values.index) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(values.index)))
    ax.set_yticklabels(values.index.tolist())
    ax.set_title("Mean PDM metrics by scene")

    for y_idx, scene_name in enumerate(values.index):
        for x_idx, metric in enumerate(labels):
            value = values.iloc[y_idx, x_idx]
            ax.text(x_idx, y_idx, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)

    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Mean value")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary(summary: pd.DataFrame, output_dir: Path, sample_count: int) -> None:
    summary_csv = output_dir / "pdm_metric_summary.csv"
    summary_md = output_dir / "pdm_metric_summary.md"

    summary.to_csv(summary_csv, float_format="%.6f")
    with summary_md.open("w", encoding="utf-8") as f:
        f.write(f"# PDM metric summary\n\nSamples: {sample_count}\n\n")
        f.write("| metric | mean | std | min | max |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for metric, row in summary.round(6).iterrows():
            f.write(
                f"| {metric} | {row['mean']:.6f} | {row['std']:.6f} | "
                f"{row['min']:.6f} | {row['max']:.6f} |\n"
            )

    print(summary.round(6).to_string())
    print()
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_md}")


def write_scene_summary(scene_summary: pd.DataFrame, output_dir: Path, scene_column: str) -> None:
    summary_csv = output_dir / "pdm_scene_summary.csv"
    summary_md = output_dir / "pdm_scene_summary.md"

    scene_summary.to_csv(summary_csv, index=False, float_format="%.6f")
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# PDM scene summary\n\n")
        f.write(f"Scene column: `{scene_column}`\n\n")
        columns = [scene_column, "sample_count"] + [f"{metric}_mean" for metric in METRIC_COLUMNS]
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |\n")
        for _, row in scene_summary.round(6).iterrows():
            values = [str(row[scene_column]), str(int(row["sample_count"]))]
            values.extend(f"{row[f'{metric}_mean']:.6f}" for metric in METRIC_COLUMNS)
            f.write("| " + " | ".join(values) + " |\n")

    print()
    print(scene_summary.round(6).to_string(index=False))
    print()
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_md}")


def try_plot(description: str, plot_func, *args, **kwargs) -> bool:
    try:
        plot_func(*args, **kwargs)
    except RuntimeError as exc:
        print(f"skipped {description}: {exc}")
        return False
    return True


def main() -> None:
    args = parse_args()
    df = load_eval_df(args.csv)
    summary = summarize_metrics(df)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_summary(summary, output_dir, sample_count=len(df))
    if try_plot("score_distribution.png", plot_score_histogram, df, output_dir / "score_distribution.png", args.dpi):
        print(f"wrote {output_dir / 'score_distribution.png'}")
    if try_plot(
        "pdm_metric_distributions.png",
        plot_bucket_histograms,
        df,
        METRIC_COLUMNS,
        output_dir / "pdm_metric_distributions.png",
        args.dpi,
    ):
        print(f"wrote {output_dir / 'pdm_metric_distributions.png'}")

    if args.scene_map_csv is not None:
        scene_df = pd.read_csv(args.scene_map_csv)
        scene_eval_df = attach_scene_labels(
            df,
            scene_df,
            scene_column=args.scene_column,
            token_column=args.token_column,
            missing_scene_label=args.missing_scene_label,
        )
        scene_eval_path = output_dir / "pdm_eval_with_scene.csv"
        scene_eval_df.to_csv(scene_eval_path, index=False)
        scene_summary = summarize_metrics_by_scene(scene_eval_df, scene_column=args.scene_column)
        write_scene_summary(scene_summary, output_dir, scene_column=args.scene_column)
        wrote_scene_score_plot = try_plot(
            "pdm_score_by_scene.png",
            plot_scene_score_means,
            scene_summary,
            args.scene_column,
            output_dir / "pdm_score_by_scene.png",
            args.dpi,
        )
        wrote_scene_heatmap = try_plot(
            "pdm_metrics_by_scene.png",
            plot_scene_metric_heatmap,
            scene_summary,
            args.scene_column,
            output_dir / "pdm_metrics_by_scene.png",
            args.dpi,
        )
        print(f"wrote {scene_eval_path}")
        if wrote_scene_score_plot:
            print(f"wrote {output_dir / 'pdm_score_by_scene.png'}")
        if wrote_scene_heatmap:
            print(f"wrote {output_dir / 'pdm_metrics_by_scene.png'}")


if __name__ == "__main__":
    main()
