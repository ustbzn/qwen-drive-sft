#!/usr/bin/env python3
"""Generate one train/val/test log split from a MOGO dataset manifest."""
from __future__ import annotations

import argparse
import math
import os
import random
import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_SPLIT_DIR = REPO_ROOT / "navsim/planning/script/config/training/log_split"

# 派生窗口 log 名规则：<源 log 名>__<场景标签>_<起始帧号>，如
# SA010522_0629-0701__20260702085916_131__road-nudge_000015。
# 这类 pkl 不是独立采集 log，而是从别的数据集按固定 stride 切出的重叠窗口
# （见 nudge/selection_manifest.json 的 rule 字段：66 帧窗口、每 5 帧起一个、
# 中心=start+15）。相邻窗口共享 61/66 帧，因此必须按源 log 整组分到 train 或
# val，否则同一路段的近邻窗口会横跨两侧，val loss 被污染。
#
# 标签段必须含至少一个小写字母，用于区分普通采集 log 的纯数字尾段
# （JLBJA20073D_0714-0716__20260714151402_132 的 20260714151402_132 不是标签）。
DERIVED_WINDOW_PATTERN = re.compile(
    r"^(?P<source>.+)__(?P<label>[a-z0-9-]*[a-z][a-z0-9-]*)_(?P<start>\d+)$"
)
# Default manifest = MOGO_MANIFEST env, else the project's active multi-city manifest (mogo_multi).
# Override with `export MOGO_MANIFEST=<path>` or `--manifest <path>`.
DEFAULT_MANIFEST = Path(
    os.environ.get(
        "MOGO_MANIFEST",
        str(REPO_ROOT / "navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml"),
    )
)


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


def log_names_in(split_dir: Path) -> list[str]:
    if not split_dir.is_dir():
        return []
    return sorted(path.stem for path in split_dir.glob("*.pkl"))


def split_train_val(log_names: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise SystemExit(f"--val-ratio must be in (0, 1), got {val_ratio}")
    if len(log_names) < 2:
        raise SystemExit("Need at least 2 train logs to create train/val split")

    val_count = max(1, math.ceil(len(log_names) * val_ratio))
    val_count = min(val_count, len(log_names) - 1)
    shuffled = list(log_names)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[val_count:]), sorted(shuffled[:val_count])


def source_log_of(log_name: str) -> str | None:
    """派生窗口返回其源 log 名；普通采集 log 返回 None。"""
    match = DERIVED_WINDOW_PATTERN.match(log_name)
    return match.group("source") if match else None


def is_derived_dataset(log_names: list[str]) -> bool:
    """整个数据集的 log 是否都是派生窗口。"""
    return bool(log_names) and all(source_log_of(name) is not None for name in log_names)


def split_train_val_grouped(
    log_names: list[str], val_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    """按源 log 分组切分派生窗口，同源窗口不跨 train/val。

    val_ratio 作用在“源 log 组”上而不是窗口上，因此实际窗口比例会随各源
    log 的窗口数波动，这是保证不泄漏的必要代价。
    """
    groups: dict[str, list[str]] = {}
    for name in log_names:
        source = source_log_of(name)
        assert source is not None, f"Not a derived window log name: {name}"
        groups.setdefault(source, []).append(name)

    sources = sorted(groups)
    if len(sources) < 2:
        # 只有一个源 log，无法在不泄漏的前提下切出 val。
        return sorted(log_names), []

    val_count = max(1, math.ceil(len(sources) * val_ratio))
    val_count = min(val_count, len(sources) - 1)
    shuffled = list(sources)
    random.Random(seed).shuffle(shuffled)
    val_sources = set(shuffled[:val_count])

    train_logs = [name for source in sources if source not in val_sources for name in groups[source]]
    val_logs = [name for source in sources if source in val_sources for name in groups[source]]
    return sorted(train_logs), sorted(val_logs)


def cache_tokens_by_log(cache_root: Path) -> dict[str, set[str]]:
    """读取 cache root 下 log_name -> token 集合。cache 不存在时返回空。"""
    if not cache_root.is_dir():
        return {}
    tokens: dict[str, set[str]] = {}
    for log_path in cache_root.iterdir():
        if not log_path.is_dir():
            continue
        tokens[log_path.name] = {
            token_path.name for token_path in log_path.iterdir() if token_path.is_dir()
        }
    return tokens


def drop_tokens_owned_by_others(
    derived_name: str,
    derived_cache: dict[str, set[str]],
    other_tokens: set[str],
) -> tuple[set[str], int, int]:
    """剔除派生数据集中 token 已被其他数据集覆盖的 log。

    派生窗口的 token 是其中心帧 token，与源数据集缓存里的 token 完全相同。若源
    数据集也 active，同一帧会被两个 cache root 各算一份样本：train 内被重复训练，
    且两侧 split 分配不同时会造成 train/val 泄漏。保留源数据集那份、剔除派生侧。
    """
    dropped_logs: set[str] = set()
    dropped_tokens = 0
    for log_name, tokens in derived_cache.items():
        overlap = tokens & other_tokens
        if not overlap:
            continue
        dropped_tokens += len(overlap)
        if overlap == tokens:
            dropped_logs.add(log_name)
        else:
            # 当前 nudge 数据集是一窗口一 token，不会走到这里。真出现部分重叠
            # 时无法用 log_names 粒度表达“只排除部分 token”，直接失败而不是静默
            # 训练重复样本。
            raise SystemExit(
                f"{derived_name}: log {log_name} has partial token overlap "
                f"({len(overlap)}/{len(tokens)}); log_names cannot express a partial exclusion"
            )
    return dropped_logs, dropped_tokens, len(derived_cache)


def write_yaml(out: Path, train_logs: list[str], val_logs: list[str], test_logs: list[str], args: argparse.Namespace) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        fp.write("# Auto-generated by gen_mogo_multi_log_split.py. Do not edit by hand.\n")
        fp.write(f"# manifest={args.manifest} train_subdir={args.train_subdir} test_subdir={args.test_subdir} ")
        fp.write(f"val_ratio={args.val_ratio} seed={args.seed}")
        fp.write(f" derived_grouping={not args.no_derived_grouping}")
        fp.write(f" global_split={args.global_split}\n")
        fp.write("# Derived-window datasets (e.g. *__road-nudge_<start>) are split by source log and\n")
        fp.write("# deduplicated against non-derived datasets' cache tokens.\n")
        fp.write("train_logs:\n")
        for name in train_logs:
            fp.write(f"  - {name}\n")
        fp.write("\nval_logs:\n")
        for name in val_logs:
            fp.write(f"  - {name}\n")
        fp.write("\ntest_logs:\n")
        for name in test_logs:
            fp.write(f"  - {name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--name", action="append", dest="names", help="Dataset name to include; repeatable")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output yaml; default log_split/<manifest stem>.yaml (matches training LOG_SPLIT=<stem>)",
    )
    parser.add_argument("--train-subdir", default="train")
    parser.add_argument("--test-subdir", default="val")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-split", action="store_true", help="Split after merging all dataset logs")
    parser.add_argument(
        "--no-derived-grouping",
        action="store_true",
        help="Disable source-log grouping and cross-dataset token dedup for derived-window datasets",
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = LOG_SPLIT_DIR / f"{args.manifest.stem}.yaml"

    datasets = load_datasets(args.manifest, args.names)

    # 先收集全部数据集的原始 log 名，再决定派生/普通处理方式。
    collected: list[dict[str, Any]] = []
    for dataset in datasets:
        logs_root = Path(str(dataset.data_root)) / "dataset" / "navsim_logs"
        current_train_logs = log_names_in(logs_root / args.train_subdir)
        current_test_logs = log_names_in(logs_root / args.test_subdir)
        derived = not args.no_derived_grouping and is_derived_dataset(current_train_logs)
        collected.append(
            {
                "name": str(dataset.name),
                "cache_path": Path(str(dataset.cache_path)),
                "train": current_train_logs,
                "test": current_test_logs,
                "derived": derived,
            }
        )
        print(
            f"{dataset.name}: train_source={len(current_train_logs)} "
            f"test={len(current_test_logs)}{' [derived-window]' if derived else ''}"
        )

    # 派生数据集的 token 与其源数据集缓存完全相同。源数据集同时 active 时，
    # 同一帧会被两个 cache root 各算一份样本，需在 log_names 层面剔除派生侧。
    if any(entry["derived"] for entry in collected):
        plain_tokens: set[str] = set()
        for entry in collected:
            if entry["derived"]:
                continue
            for tokens in cache_tokens_by_log(entry["cache_path"]).values():
                plain_tokens |= tokens
        for entry in collected:
            if not entry["derived"]:
                continue
            derived_cache = cache_tokens_by_log(entry["cache_path"])
            if not derived_cache:
                print(f"  {entry['name']}: cache not found, skipping token dedup")
                continue
            dropped_logs, dropped_tokens, total_logs = drop_tokens_owned_by_others(
                entry["name"], derived_cache, plain_tokens
            )
            if dropped_logs:
                entry["train"] = [name for name in entry["train"] if name not in dropped_logs]
                entry["test"] = [name for name in entry["test"] if name not in dropped_logs]
                print(
                    f"  {entry['name']}: dropped {len(dropped_logs)}/{total_logs} log(s) "
                    f"({dropped_tokens} token(s)) already covered by non-derived datasets"
                )

    train_source_logs: list[str] = []
    train_logs: list[str] = []
    val_logs: list[str] = []
    test_logs: list[str] = []
    for entry in collected:
        current_train_logs = entry["train"]
        train_source_logs.extend(current_train_logs)
        test_logs.extend(entry["test"])
        if not args.global_split:
            if len(current_train_logs) < 2:
                train_logs.extend(current_train_logs)
            elif entry["derived"]:
                current_train, current_val = split_train_val_grouped(
                    current_train_logs, args.val_ratio, args.seed
                )
                train_logs.extend(current_train)
                val_logs.extend(current_val)
            else:
                current_train, current_val = split_train_val(current_train_logs, args.val_ratio, args.seed)
                train_logs.extend(current_train)
                val_logs.extend(current_val)

    duplicate_train_logs = sorted({name for name in train_source_logs if train_source_logs.count(name) > 1})
    if duplicate_train_logs:
        raise SystemExit(
            "Duplicate train log name(s) across datasets; log_names cannot disambiguate them: "
            f"{duplicate_train_logs[:10]}"
        )

    if args.global_split:
        if any(entry["derived"] for entry in collected):
            print(
                "WARNING: --global-split splits at window level and ignores source-log grouping; "
                "derived-window datasets will leak overlapping windows across train/val"
            )
        train_logs, val_logs = split_train_val(sorted(train_source_logs), args.val_ratio, args.seed)
    else:
        train_logs = sorted(train_logs)
        val_logs = sorted(val_logs)
    overlap = (set(train_logs) | set(val_logs)) & set(test_logs)
    if overlap:
        print(f"WARNING: {len(overlap)} log(s) appear in both train/val and test, e.g. {sorted(overlap)[:3]}")

    write_yaml(args.out, train_logs, val_logs, sorted(set(test_logs)), args)
    print(f"train_logs: {len(train_logs)}  val_logs: {len(val_logs)}  test_logs: {len(set(test_logs))}  -> {args.out}")


if __name__ == "__main__":
    main()
