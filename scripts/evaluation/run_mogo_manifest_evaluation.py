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
        str(REPO_ROOT / "navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml"),
    )
)
DEFAULT_EVAL_SCRIPT = REPO_ROOT / "scripts/evaluation/worldsim_pdms_evaluation_2b.sh"
# Fallback merged-CSV root when the manifest has no `merged_eval_root`.
FALLBACK_MERGED_EVAL_ROOT = Path("/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim") / "merged_eval"


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


def run_dataset(dataset: Any, args: argparse.Namespace, index: int) -> Path | None:
    env = os.environ.copy()
    data_root = Path(str(dataset.data_root))
    experiment_name = f"{args.experiment_name}_{dataset.name}"
    exp_dir = data_root / "exp" / experiment_name
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


def merge_csvs(csv_by_dataset: list[tuple[Any, Path]], output_path: Path) -> None:
    frames = []
    for dataset, csv_path in csv_by_dataset:
        df = pd.read_csv(csv_path)
        df["dataset_name"] = str(dataset.name)
        df["city"] = str(dataset.city)
        df["source_csv"] = str(csv_path)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["token"].astype(str) != "average"].copy()
    average_drop_columns = [
        column
        for column in ["token", "valid", "rank", "trajectory_path", "dataset_name", "city", "source_csv"]
        if column in merged.columns
    ]
    average_row = merged.drop(columns=average_drop_columns).mean(numeric_only=True, skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = merged["valid"].all() if "valid" in merged.columns else True
    if "rank" in merged.columns:
        average_row["rank"] = "0"
    if "trajectory_path" in merged.columns:
        average_row["trajectory_path"] = ""
    average_row["dataset_name"] = "all"
    average_row["city"] = "all"
    average_row["source_csv"] = ""
    merged.loc[len(merged)] = average_row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"merged_csv={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--name", action="append", dest="names", help="Dataset name to run; repeatable")
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument("--experiment-name", default="mogo_multi_eval")
    parser.add_argument("--merged-csv", type=Path, default=None)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--qwen3-vl-path", default="")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--master-port-start", type=int, default=9675)
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

    output_path = args.merged_csv
    if output_path is None:
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        merged_root = load_merged_eval_root(args.manifest) or str(FALLBACK_MERGED_EVAL_ROOT)
        output_path = Path(merged_root) / f"{args.experiment_name}_{timestamp}.csv"
    merge_csvs(csv_by_dataset, output_path)


if __name__ == "__main__":
    main()
