#!/usr/bin/env python
"""
就地修正 Mogo NAVSIM pkl 中障碍物坐标系不一致的 bug。

背景（详见生成脚本 parseRosBag-2O-RL-RecogDrive.py 的修复）：
  gt_boxes 的 center (x,y) 已是 ego 局部系，但 heading（第6维）与
  gt_velocity_3d 是【全局系】。下游 gt_boxes_oriented_box / rotate_vector
  会再加一次 ego_yaw，导致障碍物朝向、速度方向二次偏转 ->
  BEV 显示"横着"、PDM-Closed 满速规划出失真参考轨迹。

修正（对每帧每个 box，幂等性由备份+一次性运行保证）：
  ego_yaw = yaw(ego2global_rotation)          # [w,x,y,z] 绕 z
  heading_new = wrap(heading_old - ego_yaw)
  [vx,vy]_new = R(-ego_yaw) · [vx,vy]_old
  center / 尺寸 / 其它字段不变。

用法：
  python scripts/cache_dataset/fix_mogo_obstacle_frame.py \
      --logs-dir /.../navsim_logs/train --logs-dir /.../navsim_logs/val
  # 修完后用 --rebuild-trainval 从 train+val 重新生成 trainval
"""
import argparse
import math
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


def remove_existing_path(path: Path) -> None:
    """删除已有文件、软链或目录。"""
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def symlink_target(src: Path, dst: Path) -> Path:
    """返回从 dst 所在目录指向 src 的相对软链目标。"""
    return Path(os.path.relpath(src, start=dst.parent))


def quat_to_yaw(q):
    """q = [w, x, y, z]，返回绕 z 轴的 yaw。"""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def fix_frame(frame) -> int:
    """就地修正单帧的障碍物 heading 与 velocity。返回修正的 box 数。"""
    anns = frame.get("anns")
    if not anns:
        return 0
    boxes = anns.get("gt_boxes")
    if boxes is None or len(boxes) == 0:
        return 0

    ego_yaw = quat_to_yaw(frame["ego2global_rotation"])
    cos_e, sin_e = math.cos(ego_yaw), math.sin(ego_yaw)

    boxes = np.asarray(boxes, dtype=np.float32)
    # heading: 全局 -> ego 系
    boxes[:, 6] = np.arctan2(
        np.sin(boxes[:, 6] - ego_yaw), np.cos(boxes[:, 6] - ego_yaw)
    ).astype(np.float32)
    anns["gt_boxes"] = boxes

    vel = anns.get("gt_velocity_3d")
    if vel is not None and len(vel) > 0:
        vel = np.asarray(vel, dtype=np.float32)
        vx, vy = vel[:, 0].copy(), vel[:, 1].copy()
        # R(-ego_yaw): 全局 -> ego 系
        vel[:, 0] = (cos_e * vx + sin_e * vy).astype(np.float32)
        vel[:, 1] = (-sin_e * vx + cos_e * vy).astype(np.float32)
        anns["gt_velocity_3d"] = vel

    return int(len(boxes))


def fix_pkl(path: Path, marker_key: str, dry_run: bool) -> tuple[int, int]:
    """修正单个 pkl。返回 (帧数, 修正 box 数)。已修正过的（带 marker）跳过。"""
    with path.open("rb") as f:
        data = pickle.load(f)

    # 幂等保护：在第一帧打标记，已修正则跳过
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if data[0].get(marker_key):
            return (0, -1)  # -1 表示已修正、跳过

    n_box = 0
    for frame in data:
        n_box += fix_frame(frame)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data[0][marker_key] = True

    if not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(data, f)
        tmp.replace(path)
    return (len(data), n_box)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", action="append", default=[], type=Path,
                    help="要修正的 navsim_logs 子目录（可多次指定，如 train、val）。")
    ap.add_argument("--backup-suffix", default=".orig",
                    help="备份目录后缀；<dir><suffix> 不存在时整目录复制一份备份。设为空字符串可跳过备份。")
    ap.add_argument("--marker-key", default="_obstacle_frame_fixed",
                    help="写入首帧的幂等标记键名。")
    ap.add_argument("--rebuild-trainval", type=Path, default=None,
                    help="可选：从已修正的 --logs-dir 重新生成该 trainval 目录。")
    ap.add_argument("--rebuild-trainval-as-symlink", action="store_true",
                    help="重建 trainval 时创建指向 train/val pkl 的软链，而不是复制文件。")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘。")
    args = ap.parse_args()

    if not args.logs_dir:
        ap.error("至少指定一个 --logs-dir")

    total_files = total_frames = total_boxes = skipped = 0
    for d in args.logs_dir:
        if not d.is_dir():
            print(f"[WARN] 目录不存在，跳过: {d}")
            continue
        # 整目录备份（仅一次）
        if args.backup_suffix and not args.dry_run:
            bak = Path(str(d) + args.backup_suffix)
            if not bak.exists():
                print(f"[INFO] 备份 {d} -> {bak}")
                shutil.copytree(d, bak)
            else:
                print(f"[INFO] 备份已存在，跳过备份: {bak}")

        pkls = sorted(d.glob("*.pkl"))
        print(f"[INFO] {d}: {len(pkls)} 个 pkl")
        for p in pkls:
            nf, nb = fix_pkl(p, args.marker_key, args.dry_run)
            if nb == -1:
                skipped += 1
                continue
            total_files += 1
            total_frames += nf
            total_boxes += nb
        print(f"[INFO] {d} 完成。")

    print(f"\n[SUMMARY] 修正文件={total_files} 跳过(已修正)={skipped} "
          f"帧={total_frames} box={total_boxes} dry_run={args.dry_run}")

    if args.rebuild_trainval and not args.dry_run:
        tv = args.rebuild_trainval
        print(f"\n[INFO] 重建 trainval: {tv}")
        if tv.exists() or tv.is_symlink():
            bak = Path(str(tv) + args.backup_suffix)
            if args.backup_suffix and not bak.exists():
                print(f"[INFO] 备份旧 trainval -> {bak}")
                shutil.move(str(tv), str(bak))
            else:
                remove_existing_path(tv)
        tv.mkdir(parents=True, exist_ok=True)
        n = 0
        for d in args.logs_dir:
            for p in sorted(d.glob("*.pkl")):
                dst = tv / p.name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if args.rebuild_trainval_as_symlink:
                    dst.symlink_to(symlink_target(p, dst))
                else:
                    shutil.copy2(p, dst)
                n += 1
        action = "软链" if args.rebuild_trainval_as_symlink else "拷入"
        print(f"[INFO] trainval 重建完成，{action} {n} 个 pkl -> {tv}")


if __name__ == "__main__":
    sys.exit(main())
