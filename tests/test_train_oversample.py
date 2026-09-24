"""稀有场景过采样与派生窗口 split 的单元测试（无需 GPU / 真实缓存）。"""
from __future__ import annotations

import gzip
import pickle
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts/cache_dataset"))

from navsim.planning.training.dataset import (  # noqa: E402
    CacheOnlyDataset,
    MultiCacheOnlyDataset,
    apply_train_oversample,
    oversample_qwen_train_dataset,
    load_oversample_tokens,
    oversample_train_dataset,
)


class _Builder:
    """最小 feature/target builder，只提供 get_unique_name。"""

    def __init__(self, name: str):
        self._name = name

    def get_unique_name(self) -> str:
        return self._name


def _write_cache(root: Path, layout: dict[str, list[str]], names: list[str]) -> None:
    """按 root/log/token/<name>.gz 写出最小缓存。"""
    for log_name, tokens in layout.items():
        for token in tokens:
            token_dir = root / log_name / token
            token_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                with gzip.open(token_dir / f"{name}.gz", "wb") as fp:
                    pickle.dump({name: torch.zeros(1)}, fp)


@pytest.fixture()
def builders() -> tuple[list[_Builder], list[_Builder]]:
    return [_Builder("feat")], [_Builder("targ")]


def test_multi_cache_oversample_duplicates_only_target_tokens(tmp_path, builders):
    feature_builders, target_builders = builders
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _write_cache(root_a, {"log1": ["t1", "t2"]}, ["feat", "targ"])
    _write_cache(root_b, {"log2": ["t3"]}, ["feat", "targ"])

    dataset = MultiCacheOnlyDataset(
        cache_paths=[str(root_a), str(root_b)],
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    assert len(dataset) == 3

    added = oversample_train_dataset(dataset, ["t1", "t3"], factor=5)

    assert added == 8  # 2 个命中 token × (5-1)
    assert len(dataset) == 11
    assert len(dataset.tokens) == len(dataset.samples)
    counts = {t: dataset.tokens.count(t) for t in ("t1", "t2", "t3")}
    assert counts == {"t1": 5, "t2": 1, "t3": 5}


def test_multi_cache_oversampled_samples_remain_loadable(tmp_path, builders):
    feature_builders, target_builders = builders
    root = tmp_path / "a"
    _write_cache(root, {"log1": ["t1", "t2"]}, ["feat", "targ"])

    dataset = MultiCacheOnlyDataset(
        cache_paths=[str(root), str(root)],
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    oversample_train_dataset(dataset, ["t1"], factor=3)

    # 每个下标都必须能取出样本，复制项与原项内容一致。
    loaded = [dataset[i] for i in range(len(dataset))]
    assert all("feat" in features for features, _, _ in loaded)
    assert all("targ" in targets for _, targets, _ in loaded)


def test_single_cache_oversample_duplicates_tokens(tmp_path, builders):
    feature_builders, target_builders = builders
    root = tmp_path / "a"
    _write_cache(root, {"log1": ["t1", "t2", "t3"]}, ["feat", "targ"])

    dataset = CacheOnlyDataset(
        cache_path=str(root),
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    assert len(dataset) == 3

    added = oversample_train_dataset(dataset, ["t2"], factor=4)

    assert added == 3
    assert len(dataset) == 6
    assert dataset.tokens.count("t2") == 4
    # 复制的 token 仍可正常加载（__getitem__ 按 idx 取 token 再查 path）。
    assert all("feat" in dataset[i][0] for i in range(len(dataset)))


@pytest.mark.parametrize("factor", [0, 1])
def test_oversample_is_noop_for_factor_le_one(tmp_path, builders, factor):
    feature_builders, target_builders = builders
    root = tmp_path / "a"
    _write_cache(root, {"log1": ["t1"]}, ["feat", "targ"])
    dataset = CacheOnlyDataset(
        cache_path=str(root),
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    assert oversample_train_dataset(dataset, ["t1"], factor=factor) == 0
    assert len(dataset) == 1


def test_oversample_is_noop_for_empty_tokens(tmp_path, builders):
    feature_builders, target_builders = builders
    root = tmp_path / "a"
    _write_cache(root, {"log1": ["t1"]}, ["feat", "targ"])
    dataset = CacheOnlyDataset(
        cache_path=str(root),
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    assert oversample_train_dataset(dataset, [], factor=5) == 0
    assert oversample_train_dataset(dataset, None, factor=5) == 0
    assert len(dataset) == 1


def test_oversample_unknown_token_adds_nothing(tmp_path, builders):
    feature_builders, target_builders = builders
    root = tmp_path / "a"
    _write_cache(root, {"log1": ["t1"]}, ["feat", "targ"])
    dataset = CacheOnlyDataset(
        cache_path=str(root),
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    assert oversample_train_dataset(dataset, ["not-in-cache"], factor=5) == 0
    assert len(dataset) == 1


def test_apply_train_oversample_reads_token_file(tmp_path, builders):
    feature_builders, target_builders = builders
    root = tmp_path / "a"
    _write_cache(root, {"log1": ["t1", "t2"]}, ["feat", "targ"])
    tokens_file = tmp_path / "tokens.txt"
    tokens_file.write_text("t1\n\n  t2  \n", encoding="utf-8")

    dataset = CacheOnlyDataset(
        cache_path=str(root),
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    added = apply_train_oversample(dataset, str(tokens_file), factor=3)

    assert load_oversample_tokens(str(tokens_file)) == ["t1", "t2"]
    assert added == 4
    assert len(dataset) == 6


def test_apply_train_oversample_missing_file_raises(tmp_path, builders):
    feature_builders, target_builders = builders
    root = tmp_path / "a"
    _write_cache(root, {"log1": ["t1"]}, ["feat", "targ"])
    dataset = CacheOnlyDataset(
        cache_path=str(root),
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    with pytest.raises(FileNotFoundError):
        apply_train_oversample(dataset, str(tmp_path / "missing.txt"), factor=3)


def test_apply_train_oversample_empty_path_is_noop(tmp_path, builders):
    feature_builders, target_builders = builders
    root = tmp_path / "a"
    _write_cache(root, {"log1": ["t1"]}, ["feat", "targ"])
    dataset = CacheOnlyDataset(
        cache_path=str(root),
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    assert apply_train_oversample(dataset, "", factor=5) == 0
    assert len(dataset) == 1


class _SceneBackedDataset(torch.utils.data.Dataset):
    """Small stand-in for Qwen's raw-scene Dataset and its token ordering."""

    def __init__(self, tokens):
        self._scene_loader = SimpleNamespace(tokens=list(tokens))

    def __len__(self):
        return len(self._scene_loader.tokens)

    def __getitem__(self, index):
        return self._scene_loader.tokens[index]


def test_qwen_online_dataset_oversamples_matching_tokens():
    dataset = _SceneBackedDataset(["ordinary-a", "nudge-a", "ordinary-b"])

    oversampled, added = oversample_qwen_train_dataset(dataset, ["nudge-a"], factor=4)

    assert added == 3
    assert len(oversampled) == 6
    values = [oversampled[index] for index in range(len(oversampled))]
    assert values.count("nudge-a") == 4
    assert values.count("ordinary-a") == values.count("ordinary-b") == 1


def test_qwen_concat_dataset_oversamples_tokens_across_sources():
    dataset = torch.utils.data.ConcatDataset([
        _SceneBackedDataset(["ordinary-a", "nudge-a"]),
        _SceneBackedDataset(["ordinary-b", "nudge-b"]),
    ])

    oversampled, added = oversample_qwen_train_dataset(
        dataset, ["nudge-b"], factor=3,
    )

    assert added == 2
    values = [oversampled[index] for index in range(len(oversampled))]
    assert values.count("nudge-b") == 3
    assert values.count("ordinary-a") == values.count("nudge-a") == values.count("ordinary-b") == 1


def test_qwen_oversampling_with_no_matching_token_is_noop():
    dataset = _SceneBackedDataset(["ordinary-a"])

    oversampled, added = oversample_qwen_train_dataset(dataset, ["missing"], factor=5)

    assert oversampled is dataset
    assert added == 0
    assert len(oversampled) == 1


# --- gen_mogo_multi_log_split.py 的派生窗口处理 ---------------------------------


@pytest.mark.parametrize(
    "log_name,expected",
    [
        # 普通采集 log：尾段是纯数字时间戳，不能被当成派生窗口。
        ("JLBJA20073D_0714-0716__20260714151402_132", None),
        ("JLBJA20073D_0731-0802__20260801095847_132", None),
        ("JLBJA20073D_0509__2026-05-09-14-24-06__20260509142406", None),
        # 派生窗口：标签段含小写字母。
        (
            "SA010522_0629-0701__20260702085916_131__road-nudge_000015",
            "SA010522_0629-0701__20260702085916_131",
        ),
        (
            "JLBJA20073D_0626-0627__20260626152720__road-nudge_000190",
            "JLBJA20073D_0626-0627__20260626152720",
        ),
    ],
)
def test_source_log_of(log_name, expected):
    from gen_mogo_multi_log_split import source_log_of

    assert source_log_of(log_name) == expected


def test_is_derived_dataset():
    from gen_mogo_multi_log_split import is_derived_dataset

    assert is_derived_dataset(
        [
            "src_a__road-nudge_000015",
            "src_a__road-nudge_000020",
        ]
    )
    assert not is_derived_dataset(["JLBJA20073D_0714-0716__20260714151402_132"])
    # 混合数据集不按派生处理，避免把普通 log 误分组。
    assert not is_derived_dataset(
        ["src_a__road-nudge_000015", "JLBJA20073D_0714-0716__20260714151402_132"]
    )
    assert not is_derived_dataset([])


def test_split_train_val_grouped_never_splits_a_source_log():
    from gen_mogo_multi_log_split import source_log_of, split_train_val_grouped

    logs = [f"src_{i}__road-nudge_{j:06d}" for i in range(10) for j in range(15, 15 + 5 * 4, 5)]
    train, val = split_train_val_grouped(logs, val_ratio=0.1, seed=0)

    assert sorted(train + val) == sorted(logs)
    train_sources = {source_log_of(n) for n in train}
    val_sources = {source_log_of(n) for n in val}
    assert not (train_sources & val_sources)
    assert val_sources  # 10 个源 log、ratio 0.1 -> 至少 1 组进 val


def test_split_train_val_grouped_single_source_yields_no_val():
    from gen_mogo_multi_log_split import split_train_val_grouped

    logs = ["src_a__road-nudge_000015", "src_a__road-nudge_000020"]
    train, val = split_train_val_grouped(logs, val_ratio=0.1, seed=0)
    assert train == sorted(logs)
    assert val == []


def test_drop_tokens_owned_by_others_drops_fully_covered_logs():
    from gen_mogo_multi_log_split import drop_tokens_owned_by_others

    derived_cache = {
        "src_a__road-nudge_000015": {"tok1"},
        "src_a__road-nudge_000020": {"tok2"},
        "src_b__road-nudge_000015": {"tok3"},
    }
    dropped, dropped_tokens, total = drop_tokens_owned_by_others(
        "beijing_road_nudge", derived_cache, other_tokens={"tok1", "tok3"}
    )
    assert dropped == {"src_a__road-nudge_000015", "src_b__road-nudge_000015"}
    assert dropped_tokens == 2
    assert total == 3


def test_drop_tokens_owned_by_others_no_overlap_keeps_all():
    from gen_mogo_multi_log_split import drop_tokens_owned_by_others

    derived_cache = {"src_a__road-nudge_000015": {"tok1"}}
    dropped, dropped_tokens, total = drop_tokens_owned_by_others(
        "zhuhai_road_nudge", derived_cache, other_tokens={"other"}
    )
    assert dropped == set()
    assert dropped_tokens == 0
    assert total == 1


def test_drop_tokens_owned_by_others_rejects_partial_overlap():
    from gen_mogo_multi_log_split import drop_tokens_owned_by_others

    # 一个 log 内只有部分 token 被覆盖时，log_names 粒度无法表达，必须失败而不是
    # 静默训练重复样本。
    derived_cache = {"src_a__road-nudge_000015": {"tok1", "tok2"}}
    with pytest.raises(SystemExit):
        drop_tokens_owned_by_others(
            "beijing_road_nudge", derived_cache, other_tokens={"tok1"}
        )
