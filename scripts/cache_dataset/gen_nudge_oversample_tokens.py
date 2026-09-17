#!/usr/bin/env python3
"""Collect road-nudge scene tokens from derived-window datasets into one token list.

训练侧的过采样按 token 而不是按 log 名匹配，原因是两批 nudge 数据的 token 归属
不对称：

- 珠海 nudge 的源数据集（0629-0701 / 20260624 / 20260709-20260714）在 manifest 里
  是注释状态，所以它的 380 个 token 只存在于 nudge 自己的 cache root。
- 北京 nudge 的 111 个 token 全部已存在于 active 的源数据集缓存
  （0509_0626-0627 占 64 个，0714-0716 占 47 个），因此 gen_mogo_multi_log_split.py
  会把北京 nudge 整体剔除以避免重复训练与跨 split 泄漏。这些 token 仍留在训练集
  里，只是通过源数据集进入，而源 log 名不带 road-nudge 后缀。

所以只有 token 粒度能同时覆盖两批数据。本脚本把两批 nudge 缓存的 token 求并集
写成一行一个 token 的 txt（与 local_config/data_selection/ 既有格式一致），供
run_training_recogdrive*.py 的 train_oversample_tokens_path 使用。

用法：
    python scripts/cache_dataset/gen_nudge_oversample_tokens.py \
        --manifest navsim/planning/script/config/common/mogo_dataset/nav_station.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "navsim/local_config/data_selection"
DEFAULT_MANIFEST = Path(
    os.environ.get(
        "MOGO_MANIFEST",
        str(REPO_ROOT / "navsim/planning/script/config/common/mogo_dataset/nav_station.yaml"),
    )
)


def cache_tokens(cache_root: Path) -> set[str]:
    """读取 cache root 下所有 token 名。"""
    if not cache_root.is_dir():
        return set()
    tokens: set[str] = set()
    for log_path in cache_root.iterdir():
        if not log_path.is_dir():
            continue
        tokens |= {p.name for p in log_path.iterdir() if p.is_dir()}
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--match",
        default="nudge",
        help="only datasets whose name contains this substring are collected (default: nudge)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output txt; default derives from --match")
    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_OUT_DIR / f"{args.match}_oversample_tokens_v1.txt"

    cfg = OmegaConf.load(args.manifest)
    tokens: set[str] = set()
    matched = 0
    for dataset in cfg.datasets:
        name = str(dataset.name)
        if args.match not in name:
            continue
        matched += 1
        current = cache_tokens(Path(str(dataset.cache_path)))
        if not current:
            print(f"WARNING: {name}: cache not found or empty, contributed 0 token(s)")
        else:
            print(f"{name}: {len(current)} token(s)")
        tokens |= current

    if not matched:
        raise SystemExit(f"No dataset name in {args.manifest} contains {args.match!r}")
    if not tokens:
        raise SystemExit("Collected 0 token(s); refusing to write an empty oversample list")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        for token in sorted(tokens):
            fp.write(f"{token}\n")
    print(f"\ntotal {len(tokens)} unique token(s) from {matched} dataset(s) -> {args.out}")


if __name__ == "__main__":
    main()
