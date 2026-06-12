#!/usr/bin/env python
"""
轻量验证：直接从 Mogo pkl 渲染 BEV，检查障碍物朝向/速度是否已修正到 ego 系。

不依赖模型/地图/sensor。BEV 惯例：x 朝上（车头方向）、y 朝左。
- 灰色矩形：障碍物 box，按 (x, y, heading) 绘制
- 红色短线：box 朝向（heading）
- 蓝色短线：速度方向（velocity，仅画运动目标）
- 黑色矩形：ego，固定朝上（在原点）

修正正确的判据：直道场景里多数车辆矩形长边竖直（同向/对向），
朝向箭头多指上/下而非左右横着。

用法：
  python scripts/cache_dataset/vis_check_mogo_pkl.py \
      --pkl /.../navsim_logs/trainval/<name>.pkl \
      --frame-idx -1 --out vis_output/check_fixed.png
  # 或给目录，默认取目录下第一个 pkl
  python scripts/cache_dataset/vis_check_mogo_pkl.py \
      --logs-dir /.../navsim_logs/trainval --out vis_output/check_fixed.png
"""
import argparse
import glob
import math
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


def box_corners(x, y, heading, length, width):
    """返回 box 四角 (4,2) in ego frame。length 沿 heading 方向。"""
    dx, dy = length / 2.0, width / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    c, s = math.cos(heading), math.sin(heading)
    R = np.array([[c, -s], [s, c]])
    return (local @ R.T) + np.array([x, y])


def draw_frame(ax, frame, max_range=64.0, draw_velocity=True):
    anns = frame["anns"]
    boxes = np.asarray(anns["gt_boxes"], dtype=np.float64)
    names = list(anns.get("gt_names", []))
    vel = np.asarray(anns.get("gt_velocity_3d", np.zeros((len(boxes), 3))), dtype=np.float64)

    n_box = 0
    for i, b in enumerate(boxes):
        x, y, _z, length, width = b[0], b[1], b[2], b[3], b[4]
        heading = b[6]
        # BEV: 横轴=y(左为正), 纵轴=x(前为正) -> plot(col=y, row=x)
        corners = box_corners(x, y, heading, length, width)
        poly = MplPolygon(
            np.column_stack([corners[:, 1], corners[:, 0]]),
            closed=True, fill=True, facecolor="0.7", edgecolor="0.2",
            linewidth=0.8, zorder=2,
        )
        ax.add_patch(poly)
        # 朝向箭头（红）：从中心沿 heading 伸出 length/2+1
        hx = x + math.cos(heading) * (length / 2 + 1.0)
        hy = y + math.sin(heading) * (length / 2 + 1.0)
        ax.plot([y, hy], [x, hx], color="red", linewidth=1.5, zorder=3)
        # 速度箭头（蓝）：仅运动目标
        if draw_velocity and i < len(vel):
            spd = math.hypot(vel[i, 0], vel[i, 1])
            if spd > 1.0:
                vx_end = x + vel[i, 0] * 0.8
                vy_end = y + vel[i, 1] * 0.8
                ax.plot([y, vy_end], [x, vx_end], color="blue",
                        linewidth=1.2, linestyle="--", zorder=4)
        n_box += 1

    # ego：固定朝上，画在原点
    ego = box_corners(0, 0, 0, 4.6, 2.0)
    ax.add_patch(MplPolygon(
        np.column_stack([ego[:, 1], ego[:, 0]]),
        closed=True, fill=True, facecolor="black", edgecolor="black", zorder=5))
    ax.plot([0, 0], [0, 3.0], color="lime", linewidth=2.0, zorder=6)  # ego 朝向

    ax.set_xlim(max_range, -max_range)   # y 轴：左正，翻转使左在左
    ax.set_ylim(-max_range, max_range)   # x 轴：前正，向上
    ax.set_aspect("equal")
    ax.set_xlabel("ego-left  y (m)")
    ax.set_ylabel("ego-front  x (m)")
    ax.grid(True, alpha=0.3)
    return n_box


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pkl", type=Path, help="单个 pkl 文件")
    g.add_argument("--logs-dir", type=Path, help="目录（默认取第一个 pkl）")
    ap.add_argument("--frame-idx", type=int, default=-1, help="帧索引，默认最后一帧")
    ap.add_argument("--n-frames", type=int, default=1, help="渲染连续多帧（横向拼图）")
    ap.add_argument("--max-range", type=float, default=64.0)
    ap.add_argument("--out", type=Path, default=Path("vis_output/check_fixed.png"))
    args = ap.parse_args()

    pkl = args.pkl
    if pkl is None:
        cands = sorted(glob.glob(str(args.logs_dir / "*.pkl")))
        if not cands:
            ap.error(f"目录下无 pkl: {args.logs_dir}")
        pkl = Path(cands[0])

    with pkl.open("rb") as f:
        data = pickle.load(f)
    n = len(data)
    fi0 = args.frame_idx if args.frame_idx >= 0 else n + args.frame_idx
    fis = [min(max(fi0 + k, 0), n - 1) for k in range(args.n_frames)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(fis), figsize=(7 * len(fis), 7), squeeze=False)
    for ax, fi in zip(axes[0], fis):
        nb = draw_frame(ax, data[fi], max_range=args.max_range)
        ego_yaw = math.degrees(math.atan2(
            2 * (data[fi]["ego2global_rotation"][0] * data[fi]["ego2global_rotation"][3]),
            1 - 2 * data[fi]["ego2global_rotation"][3] ** 2))
        ax.set_title(f"{pkl.name}\nframe {fi}/{n}  boxes={nb}  ego_yaw≈{ego_yaw:.0f}°", fontsize=9)

    fig.suptitle("红=朝向(heading)  蓝虚=速度  黑=ego(朝上)  | 直道应多数竖直", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[OK] 已保存 -> {args.out}  (帧 {fis}, pkl={pkl})")


if __name__ == "__main__":
    main()
