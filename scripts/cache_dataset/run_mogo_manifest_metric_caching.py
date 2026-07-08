#!/usr/bin/env python3
"""Run MOGO metric caching for each dataset in a MOGO manifest."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "navsim/planning/script/config/common/mogo_dataset/worldsim.yaml"
METRIC_SCRIPT = REPO_ROOT / "navsim/planning/script/run_mogo_metric_caching.py"


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


def run_dataset(dataset: Any, args: argparse.Namespace) -> None:
    env = os.environ.copy()
    data_root = Path(str(dataset.data_root))
    env.update(
        {
            "NAVSIM_DEVKIT_ROOT": str(REPO_ROOT),
            "OPENSCENE_DATA_ROOT": str(data_root / "dataset"),
            "NUPLAN_MAPS_ROOT": str(data_root / "dataset" / "maps"),
            "NAVSIM_EXP_ROOT": str(data_root / "exp"),
            "PYTHONPATH": f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}",
        }
    )
    cmd = [
        sys.executable,
        str(METRIC_SCRIPT),
        f"train_test_split={args.train_test_split}",
        f"cache.cache_path={dataset.metric_cache_path}",
        f"mogo_map.sqlite_path={dataset.sqlite_path}",
        f"mogo_map.utm_zone={dataset.utm_zone}",
        f"mogo_map.utm_south={str(bool(dataset.get('utm_south', False))).lower()}",
        *args.overrides,
    ]
    print(f"[{dataset.name}] metric_cache_path={dataset.metric_cache_path}")
    print(" ".join(str(part) for part in cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--name", action="append", dest="names", help="Dataset name to run; repeatable")
    parser.add_argument("--train-test-split", default="trainval")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Extra Hydra overrides forwarded to metric caching")
    args = parser.parse_args()

    for dataset in load_datasets(args.manifest, args.names):
        run_dataset(dataset, args)


if __name__ == "__main__":
    main()
