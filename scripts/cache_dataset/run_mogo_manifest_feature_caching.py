#!/usr/bin/env python3
"""Run ReCogDrive feature caching for each dataset in a MOGO manifest."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
# Default manifest is overridable via `export MOGO_MANIFEST=<path>`; worldsim stays the fallback.
DEFAULT_MANIFEST = Path(
    os.environ.get(
        "MOGO_MANIFEST",
        str(REPO_ROOT / "navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml"),
    )
)
DEFAULT_CACHE_SCRIPT = REPO_ROOT / "scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh"


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


def run_dataset(dataset: Any, args: argparse.Namespace, index: int) -> None:
    env = os.environ.copy()
    data_root = Path(str(dataset.data_root))
    env.update(
        {
            "DATA_ROOT": str(data_root),
            "CACHE_PATH": str(dataset.cache_path),
            "SQLITE_PATH": str(dataset.sqlite_path),
            "MOGO_UTM_ZONE": str(dataset.utm_zone),
            "MOGO_UTM_SOUTH": str(bool(dataset.get("utm_south", False))).lower(),
            "NAVSIM_DEVKIT_ROOT": str(REPO_ROOT),
            "OPENSCENE_DATA_ROOT": str(data_root / "dataset"),
            "NUPLAN_MAPS_ROOT": str(data_root / "dataset" / "maps"),
            "NAVSIM_EXP_ROOT": str(data_root / "exp"),
            "PYTHONPATH": f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}",
            "TRAIN_TEST_SPLIT": args.train_test_split,
            "MASTER_PORT": str(args.master_port_start + index),
        }
    )
    if args.qwen3_vl_path:
        env["QWEN3_VL_PATH"] = args.qwen3_vl_path
    if args.gpus:
        env["GPUS"] = str(args.gpus)

    cmd = ["bash", str(args.cache_script), *args.overrides]
    print(f"[{dataset.name}] DATA_ROOT={env['DATA_ROOT']}")
    print(f"[{dataset.name}] CACHE_PATH={env['CACHE_PATH']}")
    print(f"[{dataset.name}] SQLITE_PATH={env['SQLITE_PATH']} UTM={env['MOGO_UTM_ZONE']}")
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--name", action="append", dest="names", help="Dataset name to run; repeatable")
    parser.add_argument("--cache-script", type=Path, default=DEFAULT_CACHE_SCRIPT)
    parser.add_argument("--train-test-split", default="mogo_trainval")
    parser.add_argument("--qwen3-vl-path", default="")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--master-port-start", type=int, default=9675)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Extra Hydra overrides forwarded to the cache script")
    args = parser.parse_args()

    for index, dataset in enumerate(load_datasets(args.manifest, args.names)):
        run_dataset(dataset, args, index)


if __name__ == "__main__":
    main()
