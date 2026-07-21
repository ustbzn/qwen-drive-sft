#!/usr/bin/env python3
"""Run per-city MOGO evaluation from a manifest and merge the resulting CSV files."""
from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
# Default manifest is overridable via `export MOGO_MANIFEST=<path>`; worldsim stays the fallback.
DEFAULT_MANIFEST = Path(
    os.environ.get(
        "MOGO_MANIFEST",
        str(REPO_ROOT / "navsim/planning/script/config/common/mogo_dataset/vla_multi.yaml"),
    )
)
DEFAULT_EVAL_SCRIPT = REPO_ROOT / "scripts/evaluation/run_recogdrive_qwen3_agent_pdm_score_evaluation_2b.sh"
# Fallback merged-CSV root when the manifest has no `merged_eval_root`.
FALLBACK_MERGED_EVAL_ROOT = Path("/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/mogo_multi") / "merged_eval"
MERGED_CSV_FILENAME = "merged_results.csv"
SUMMARY_FILENAME = "evaluation_summary.md"


def load_datasets(manifest: Path, names: list[str] | None) -> list[Any]:
    cfg = OmegaConf.load(manifest)
    datasets = list(cfg.datasets)
    if names:
        wanted = set(names)
        datasets = [dataset for dataset in datasets if str(dataset.name) in wanted]
        missing = wanted - {str(dataset.name) for dataset in datasets}
        if missing:
            raise SystemExit(f"Unknown dataset name(s): {sorted(missing)}")
    return datasets


def load_merged_eval_root(manifest: Path) -> str | None:
    """Read optional top-level `merged_eval_root` from the manifest; None if absent."""
    cfg = OmegaConf.load(manifest)
    root = cfg.get("merged_eval_root", None)
    return str(root) if root else None


def latest_csv(data_root: Path, experiment_name: str, previous: set[Path]) -> Path | None:
    exp_dir = data_root / "exp" / experiment_name
    csv_files = set(exp_dir.glob("*/*.csv")) if exp_dir.is_dir() else set()
    new_files = sorted(csv_files - previous, key=lambda path: path.stat().st_mtime)
    return new_files[-1] if new_files else None


def _normalized_checkpoint(checkpoint: Any) -> str:
    return str(checkpoint).strip().strip("\"'")


def find_reusable_csv(data_root: Path, experiment_name: str, checkpoint: str) -> Path | None:
    exp_dir = data_root / "exp" / experiment_name
    csv_files = sorted(exp_dir.glob("*/*.csv"), key=lambda path: path.stat().st_mtime)
    if not csv_files:
        return None

    csv_path = csv_files[-1]
    try:
        tokens = pd.read_csv(csv_path, usecols=["token"])["token"].astype(str)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Cannot resume with invalid CSV {csv_path}: {error}") from error
    if not tokens.ne("average").any():
        raise SystemExit(f"Cannot resume with CSV containing no scenario results: {csv_path}")

    if checkpoint:
        config_candidates = [
            csv_path.parent / "code" / "hydra" / "config.yaml",
            csv_path.parent / ".hydra" / "config.yaml",
        ]
        config_path = next((path for path in config_candidates if path.is_file()), None)
        if config_path is None:
            raise SystemExit(f"Cannot verify checkpoint for reusable CSV: {csv_path}")
        cfg = OmegaConf.load(config_path)
        existing_checkpoint = OmegaConf.select(cfg, "agent.checkpoint_path")
        if existing_checkpoint is None:
            raise SystemExit(f"Reusable CSV Hydra config has no agent.checkpoint_path: {config_path}")
        if _normalized_checkpoint(existing_checkpoint) != _normalized_checkpoint(checkpoint):
            raise SystemExit(
                f"Checkpoint mismatch for reusable CSV {csv_path}: "
                f"expected {checkpoint}, found {existing_checkpoint}"
            )

    return csv_path


def run_dataset(dataset: Any, args: argparse.Namespace, index: int) -> Path | None:
    env = os.environ.copy()
    data_root = Path(str(dataset.data_root))
    experiment_name = f"{args.experiment_name}_{dataset.name}"
    exp_dir = data_root / "exp" / experiment_name
    if args.resume:
        reusable_csv = find_reusable_csv(data_root, experiment_name, args.checkpoint)
        if reusable_csv is not None:
            print(f"[{dataset.name}] reusing_csv={reusable_csv}")
            return reusable_csv

    previous_csvs = set(exp_dir.glob("*/*.csv")) if exp_dir.is_dir() else set()
    env.update(
        {
            "DATA_ROOT": str(data_root),
            "METRIC_CACHE_PATH": str(dataset.metric_cache_path),
            "SQLITE_PATH": str(dataset.sqlite_path),
            "MOGO_UTM_ZONE": str(dataset.utm_zone),
            "MOGO_UTM_SOUTH": str(bool(dataset.get("utm_south", False))).lower(),
            "EXPERIMENT_NAME": experiment_name,
            "NAVSIM_DEVKIT_ROOT": str(REPO_ROOT),
            "OPENSCENE_DATA_ROOT": str(data_root / "dataset"),
            "NUPLAN_MAPS_ROOT": str(data_root / "dataset" / "maps"),
            "NAVSIM_EXP_ROOT": str(data_root / "exp"),
            "PYTHONPATH": f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}",
            "MASTER_PORT": str(args.master_port_start + index),
        }
    )
    if args.gpus:
        env["GPUS"] = str(args.gpus)
    if args.checkpoint:
        env["CHECKPOINT"] = args.checkpoint
    if args.qwen3_vl_path:
        env["QWEN3_VL_PATH"] = args.qwen3_vl_path

    cmd = ["bash", str(args.eval_script), *args.overrides]
    print(f"[{dataset.name}] DATA_ROOT={env['DATA_ROOT']}")
    print(f"[{dataset.name}] METRIC_CACHE_PATH={env['METRIC_CACHE_PATH']}")
    print(f"[{dataset.name}] EXPERIMENT_NAME={experiment_name}")
    print(" ".join(cmd))
    if args.dry_run:
        return None

    subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)
    csv_path = latest_csv(data_root, experiment_name, previous_csvs)
    if csv_path is None:
        raise SystemExit(f"No new CSV found under {exp_dir}")
    print(f"[{dataset.name}] csv={csv_path}")
    return csv_path


def merge_csvs(csv_by_dataset: list[tuple[Any, Path]], output_path: Path) -> pd.DataFrame:
    frames = []
    for dataset, csv_path in csv_by_dataset:
        df = pd.read_csv(csv_path)
        # Older evaluation CSVs include the pandas index as an unnamed column.
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")
        df = df[df["token"].astype(str) != "average"].copy()
        df["dataset_name"] = str(dataset.name)
        df["city"] = str(dataset.city)
        df["source_csv"] = str(csv_path)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    average_drop_columns = [
        column
        for column in [
            "token",
            "valid",
            "rank",
            "trajectory_path",
            "trajectory_source",
            "progress_reference_source",
            "dataset_name",
            "city",
            "source_csv",
        ]
        if column in merged.columns
    ]
    average_row = merged.drop(columns=average_drop_columns).mean(numeric_only=True, skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = merged["valid"].all() if "valid" in merged.columns else True
    if "rank" in merged.columns:
        average_row["rank"] = 0
    for column in ["trajectory_path", "trajectory_source", "progress_reference_source"]:
        if column in merged.columns:
            average_row[column] = ""
    average_row["dataset_name"] = "all"
    average_row["city"] = "all"
    average_row["source_csv"] = ""
    merged.loc[len(merged)] = average_row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"merged_csv={output_path}")
    return merged


def default_output_paths(merged_root: Path, experiment_name: str, timestamp: str) -> tuple[Path, Path]:
    run_dir = merged_root / f"{experiment_name}_{timestamp}"
    return run_dir / MERGED_CSV_FILENAME, run_dir / SUMMARY_FILENAME


def _successful_count(frame: pd.DataFrame) -> int:
    if "valid" not in frame.columns:
        return len(frame)
    valid = frame["valid"]
    if pd.api.types.is_bool_dtype(valid):
        return int(valid.fillna(False).sum())
    return int(valid.astype(str).str.lower().eq("true").sum())


def _mean_score(frame: pd.DataFrame) -> float:
    if "score" not in frame.columns:
        return float("nan")
    return float(pd.to_numeric(frame["score"], errors="coerce").mean())


def _format_score(score: float) -> str:
    return "n/a" if pd.isna(score) else f"{score:.12f}"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_evaluation_summary(
    merged: pd.DataFrame,
    summary_path: Path,
    merged_csv_path: Path,
    experiment_name: str,
    checkpoint: str,
    manifest: Path,
    generated_at: datetime,
) -> None:
    details = merged[merged["token"].astype(str) != "average"].copy()
    successful = _successful_count(details)
    failed = len(details) - successful

    lines = [
        "# MOGO multi-dataset evaluation summary",
        "",
        f"- Experiment: `{experiment_name}`",
        f"- Generated at: `{generated_at.astimezone().isoformat(timespec='seconds')}`",
        f"- Checkpoint: `{checkpoint or '(evaluation script default)'}`",
        f"- Manifest: `{manifest}`",
        f"- Merged CSV: `{merged_csv_path}`",
        f"- Total scenarios: {len(details)}",
        f"- Successful scenarios: {successful}",
        f"- Failed scenarios: {failed}",
        f"- Overall average score: `{_format_score(_mean_score(details))}`",
        "",
        "## Dataset results",
        "",
        "| Dataset | City | Scenarios | Successful | Failed | Average score | Source CSV |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for dataset_name, frame in details.groupby("dataset_name", sort=False):
        dataset_successful = _successful_count(frame)
        source_csv = frame["source_csv"].iloc[0] if "source_csv" in frame.columns else ""
        city = frame["city"].iloc[0] if "city" in frame.columns else ""
        lines.append(
            "| "
            f"{_markdown_cell(dataset_name)} | {_markdown_cell(city)} | {len(frame)} | "
            f"{dataset_successful} | {len(frame) - dataset_successful} | "
            f"{_format_score(_mean_score(frame))} | {_markdown_cell(source_csv)} |"
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"evaluation_summary={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--name", action="append", dest="names", help="Dataset name to run; repeatable")
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument("--experiment-name", default="mogo_multi_eval")
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=None,
        help="Explicit merged CSV path; the evaluation summary is written alongside it",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--qwen3-vl-path", default="")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--master-port-start", type=int, default=9675)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the latest complete, checkpoint-matching CSV and evaluate only missing datasets",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Extra Hydra overrides forwarded to the eval script")
    args = parser.parse_args()

    csv_by_dataset: list[tuple[Any, Path]] = []
    for index, dataset in enumerate(load_datasets(args.manifest, args.names)):
        csv_path = run_dataset(dataset, args, index)
        if csv_path is not None:
            csv_by_dataset.append((dataset, csv_path))

    if args.dry_run or not csv_by_dataset:
        return

    generated_at = datetime.now()
    output_path = args.merged_csv
    if output_path is None:
        timestamp = generated_at.strftime("%Y.%m.%d.%H.%M.%S")
        merged_root = load_merged_eval_root(args.manifest) or str(FALLBACK_MERGED_EVAL_ROOT)
        output_path, summary_path = default_output_paths(Path(merged_root), args.experiment_name, timestamp)
    else:
        summary_path = output_path.parent / SUMMARY_FILENAME

    merged = merge_csvs(csv_by_dataset, output_path)
    write_evaluation_summary(
        merged=merged,
        summary_path=summary_path,
        merged_csv_path=output_path,
        experiment_name=args.experiment_name,
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        generated_at=generated_at,
    )


if __name__ == "__main__":
    main()
