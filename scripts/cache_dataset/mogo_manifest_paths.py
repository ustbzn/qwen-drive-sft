#!/usr/bin/env python3
"""Print manifest-derived paths for the IL/RL training shells.

Single source of truth: given a MOGO manifest, emit either the colon-joined list of a
per-dataset field (e.g. cache_path / metric_cache_path) across all (or --names filtered)
datasets, or a single top-level scalar (e.g. train_exp_root). The training shells call
this to auto-populate CACHE_PATHS / METRIC_CACHE_PATHS / NAVSIM_EXP_ROOT so multi-city
runs need only `export MOGO_MANIFEST=<file>`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(
    os.environ.get(
        "MOGO_MANIFEST",
        str(REPO_ROOT / "navsim/planning/script/config/common/mogo_dataset/worldsim.yaml"),
    )
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--field", default="cache_path", help="per-dataset field, or top-level key with --top-level")
    parser.add_argument("--names", default="", help="comma-separated dataset names to include (empty = all)")
    parser.add_argument("--sep", default=os.pathsep, help="join separator for the per-dataset list (default os.pathsep=':')")
    parser.add_argument("--top-level", action="store_true", help="read a single top-level scalar field instead of a per-dataset list")
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    cfg = OmegaConf.load(args.manifest)

    if args.top_level:
        value = cfg.get(args.field, None)
        sys.stdout.write("" if value is None else str(value))
        return

    datasets = list(cfg.datasets)
    wanted = [name for name in args.names.split(",") if name]
    if wanted:
        datasets = [dataset for dataset in datasets if str(dataset.name) in set(wanted)]
        missing = set(wanted) - {str(dataset.name) for dataset in datasets}
        if missing:
            raise SystemExit(f"Unknown dataset name(s): {sorted(missing)}")

    values = [str(dataset.get(args.field)) for dataset in datasets if dataset.get(args.field) is not None]
    sys.stdout.write(args.sep.join(values))


if __name__ == "__main__":
    main()
