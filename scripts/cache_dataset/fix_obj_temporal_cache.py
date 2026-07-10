#!/usr/bin/env python
"""清洗 object_temporal_feature_v1.gz 缓存里的异常 obj_diff_features 坏样本。

背景：极少数样本因历史帧 ego_pose 异常，坐标变换未抵消 UTM 大偏移，导致 obj_diff_features
出现 1e5~1e6 量级坏值，写进了缓存 gz。这些值在 fp16 训练下经 TemporalObjectEncoder 的
Conv1d tokenizer 溢出成 inf/nan，污染整个 batch 的 loss（表现为 train/loss_step=nan）。

模型侧防御(planner sanitize_obj_diff)与数据生成侧修复(_build_diff_features 异常帧置零)已落地；
本脚本负责第三层：清洗**已经生成**的缓存文件，把坏 slot 的 diff 就地置零，使无需重跑整批缓存
即可得到干净数据。置零后的 slot 等价于"该 track 无有效时序运动"（motion 分支输出 0），abs 几何
分支与 obj_valid_mask 不受影响，语义安全。

默认 dry-run，只报告不改动。--apply 才写回（写回前对每个文件做 .bak 备份，除非 --no-backup）。

用法：
  # 只扫描报告（默认，安全）
  python scripts/cache_dataset/fix_obj_temporal_cache.py --roots /path/a /path/b
  # 用 manifest 自动取所有批次 cache_path 扫描
  python scripts/cache_dataset/fix_obj_temporal_cache.py --manifest navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml
  # 确认后就地修复
  python scripts/cache_dataset/fix_obj_temporal_cache.py --manifest ... --apply
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

import torch

FEATURE_NAME = "object_temporal_feature_v1.gz"
DIFF_KEY = "obj_diff_features"
# 与 planner sanitize / 数据侧 _MAX_DELTA_POS 对齐的判据：归一化后正常 diff 全量实测 max≈6，
# 坏值 1e4~1e6。阈值 50（归一化空间）远离正常、牢牢罩住坏值。
BAD_THRESHOLD = 50.0


def _iter_feature_files(root: str):
    """yield 每个 log 目录下的所有 object_temporal_feature_v1.gz 路径。"""
    try:
        logs = [e.path for e in os.scandir(root) if e.is_dir()]
    except FileNotFoundError:
        return
    for logdir in logs:
        try:
            toks = [e.path for e in os.scandir(logdir) if e.is_dir()]
        except FileNotFoundError:
            continue
        for tok in toks:
            p = os.path.join(tok, FEATURE_NAME)
            if os.path.exists(p):
                yield p


def _inspect_and_maybe_fix(args: Tuple[str, bool, bool, float]) -> Tuple[str, float, int, bool]:
    """返回 (path, worst_abs, n_bad_slots, changed)。apply=False 时 changed 恒 False。"""
    path, apply, backup, threshold = args
    try:
        with gzip.open(path, "rb") as fh:
            data = pickle.load(fh)
    except Exception:
        return (path, -1.0, 0, False)
    if not isinstance(data, dict) or DIFF_KEY not in data:
        return (path, 0.0, 0, False)
    diff = data[DIFF_KEY]
    if not torch.is_tensor(diff):
        return (path, 0.0, 0, False)
    clean = torch.nan_to_num(diff.float(), nan=0.0, posinf=0.0, neginf=0.0)
    worst = float(clean.abs().max()) if clean.numel() else 0.0
    has_naninf = bool(torch.isnan(diff).any() or torch.isinf(diff).any())
    # 坏 slot: 该 slot 任意 delta 的任意维超过阈值，或含 nan/inf
    per_slot_max = clean.abs().amax(dim=tuple(range(1, clean.dim())))  # [num_slots]
    bad_mask = (per_slot_max > threshold)
    if diff.dim() >= 2:
        naninf_slot = torch.isnan(diff).any(dim=tuple(range(1, diff.dim()))) | torch.isinf(diff).any(dim=tuple(range(1, diff.dim())))
        bad_mask = bad_mask | naninf_slot
    n_bad = int(bad_mask.sum())
    if n_bad == 0 and not has_naninf:
        return (path, worst, 0, False)
    if not apply:
        return (path, worst, n_bad, False)
    # 就地修复：坏 slot 的 diff 整体置零（等价该 track 无有效时序运动）。
    fixed = diff.clone()
    fixed[bad_mask] = 0.0
    fixed = torch.nan_to_num(fixed, nan=0.0, posinf=0.0, neginf=0.0)
    data[DIFF_KEY] = fixed
    if backup:
        shutil.copy2(path, path + ".bak")
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb") as fh:
        pickle.dump(data, fh)
    os.replace(tmp, path)
    return (path, worst, n_bad, True)


def _resolve_roots(cli_roots: Optional[List[str]], manifest: Optional[str]) -> List[str]:
    roots: List[str] = list(cli_roots or [])
    if manifest:
        # 复用仓库既有 helper 解析 manifest 的 cache_path 字段
        here = os.path.dirname(os.path.abspath(__file__))
        helper = os.path.join(here, "mogo_manifest_paths.py")
        import subprocess

        out = subprocess.check_output(
            [sys.executable, helper, "--manifest", manifest, "--field", "cache_path"],
            text=True,
        ).strip()
        roots.extend([r for r in out.split(":") if r])
    # 去重保序
    seen = set()
    uniq = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", default=None, help="cache_path 根目录（可多个）")
    ap.add_argument("--manifest", default=None, help="mogo manifest yaml，自动取所有批次 cache_path")
    ap.add_argument("--apply", action="store_true", help="就地写回修复（默认 dry-run 只报告）")
    ap.add_argument("--no-backup", action="store_true", help="apply 时不保留 .bak 备份")
    ap.add_argument("--workers", type=int, default=6, help="并行进程数")
    ap.add_argument("--threshold", type=float, default=BAD_THRESHOLD, help="坏值判定阈值（归一化空间）")
    args = ap.parse_args()

    threshold = float(args.threshold)

    roots = _resolve_roots(args.roots, args.manifest)
    if not roots:
        print("ERROR: 未提供 --roots 或 --manifest", file=sys.stderr)
        return 2
    print(f"扫描 {len(roots)} 个 cache 根目录，mode={'APPLY' if args.apply else 'DRY-RUN'}，阈值={threshold}")
    for r in roots:
        print(f"  - {r}")

    total = 0
    bad_files = 0
    fixed_files = 0
    worst_overall = 0.0
    bad_list: List[Tuple[str, float, int]] = []

    for root in roots:
        files = list(_iter_feature_files(root))
        tasks = [(f, args.apply, not args.no_backup, threshold) for f in files]
        n = 0
        rbad = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for (path, worst, nbad, changed) in ex.map(_inspect_and_maybe_fix, tasks, chunksize=8):
                n += 1
                if worst > worst_overall:
                    worst_overall = worst
                if nbad > 0:
                    rbad += 1
                    bad_files += 1
                    if len(bad_list) < 100:
                        bad_list.append((path, worst, nbad))
                if changed:
                    fixed_files += 1
        total += n
        print(f"[{os.path.basename(os.path.dirname(os.path.dirname(root)))}] scanned={n} bad_files={rbad}")

    print("\n===== 汇总 =====")
    print(f"总扫描文件: {total}")
    print(f"坏文件数  : {bad_files}")
    print(f"最坏 |diff|: {worst_overall:.4e}")
    if args.apply:
        print(f"已修复文件: {fixed_files}" + ("" if args.no_backup else "（含 .bak 备份）"))
    else:
        print("DRY-RUN：未改动任何文件。确认后加 --apply 就地修复。")
    if bad_list:
        print("\n坏文件样例（最多 100 个）:")
        for path, worst, nbad in sorted(bad_list, key=lambda x: -x[1])[:100]:
            print(f"  worst={worst:.3e} bad_slots={nbad} :: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
