#!/usr/bin/env python3
"""检查 metric cache 中的道路边界信息。

用于验证边界类型(boundary_type)信息是否已正确保存在 metric cache 中，
以及分析边界约束对于特定场景的作用。

Usage:
    python scripts/check_boundary_in_metric_cache.py \\
        --cache-root /path/to/metric_cache \\
        --token TOKEN_ID
"""

import argparse
import lzma
import pickle
import sys
from collections import Counter
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def find_metric_cache(cache_root: Path, token: str) -> Path:
    """在 metric cache 根目录下查找指定 token 的 metric_cache.pkl 文件。"""
    cache_files = list(cache_root.rglob(f"*/{token}/metric_cache.pkl"))
    if not cache_files:
        raise FileNotFoundError(
            f"未找到 token={token} 的 metric cache 文件（在 {cache_root} 下）"
        )
    if len(cache_files) > 1:
        print(f"警告：找到多个匹配文件，使用第一个：{cache_files[0]}")
    return cache_files[0]


def load_metric_cache(cache_path: Path):
    """加载并解压 metric cache 文件。"""
    with open(cache_path, 'rb') as f:
        decompressed = lzma.decompress(f.read())
        return pickle.loads(decompressed)


def analyze_boundary_info(cache_data):
    """分析 centerline 的边界信息。"""
    centerline = cache_data.centerline
    print("=" * 80)
    print("中心线信息")
    print("=" * 80)
    print(f"类型: {type(centerline).__name__}")
    print(f"长度: {centerline.length:.2f} m")
    print(f"离散点数: {len(centerline.discrete_path)}")
    print(f"包含边界轮廓: {centerline.has_boundary_profile()}")
    print()

    if not centerline.has_boundary_profile():
        print("⚠️  该 centerline 没有边界轮廓信息！")
        return

    boundary_profile = centerline.boundary_profile
    left_boundary = boundary_profile.left_boundary
    right_boundary = boundary_profile.right_boundary

    print("=" * 80)
    print("边界点统计")
    print("=" * 80)
    print(f"左边界点数: {len(left_boundary)}")
    print(f"右边界点数: {len(right_boundary)}")
    print()

    # 分析边界类型分布
    left_types = Counter([p.line_type for p in left_boundary])
    right_types = Counter([p.line_type for p in right_boundary])

    BOUNDARY_TYPE_NAMES = {
        0: "DEFAULT (默认/未知)",
        1: "CROSSABLE (可压线，如虚线)",
        2: "NON_CROSSABLE (不可压线，如实线/路沿/护栏)",
    }

    print("=" * 80)
    print("左边界类型分布")
    print("=" * 80)
    for line_type, count in sorted(left_types.items()):
        percentage = 100.0 * count / len(left_boundary)
        print(f"  {line_type} = {BOUNDARY_TYPE_NAMES.get(line_type, '未知')}")
        print(f"      数量: {count} ({percentage:.1f}%)")
    print()

    print("=" * 80)
    print("右边界类型分布")
    print("=" * 80)
    for line_type, count in sorted(right_types.items()):
        percentage = 100.0 * count / len(right_boundary)
        print(f"  {line_type} = {BOUNDARY_TYPE_NAMES.get(line_type, '未知')}")
        print(f"      数量: {count} ({percentage:.1f}%)")
    print()

    # 采样边界点显示
    print("=" * 80)
    print("边界点采样（前5个 + 中间1个 + 后5个）")
    print("=" * 80)

    def sample_boundary(boundary_points, name):
        print(f"\n{name}:")
        indices = list(range(min(5, len(boundary_points))))
        if len(boundary_points) > 10:
            indices.append(len(boundary_points) // 2)
        indices.extend(range(max(0, len(boundary_points) - 5), len(boundary_points)))
        for i in indices:
            p = boundary_points[i]
            type_name = BOUNDARY_TYPE_NAMES.get(p.line_type, f"未知类型({p.line_type})")
            marker = "⚠️ " if p.line_type == 2 else "  "
            print(f"{marker}[{i:3d}] s={p.s:7.2f}m, l={p.l:+6.2f}m, type={type_name}")

    sample_boundary(left_boundary, "左边界")
    sample_boundary(right_boundary, "右边界")
    print()

    # 统计 NON_CROSSABLE 边界覆盖的路程
    left_non_crossable_s = [p.s for p in left_boundary if p.line_type == 2]
    right_non_crossable_s = [p.s for p in right_boundary if p.line_type == 2]

    print("=" * 80)
    print("NON_CROSSABLE 边界覆盖范围")
    print("=" * 80)
    if left_non_crossable_s:
        print(f"左边界: {min(left_non_crossable_s):.2f}m ~ {max(left_non_crossable_s):.2f}m "
              f"(共 {len(left_non_crossable_s)} 个点)")
    else:
        print("左边界: 无 NON_CROSSABLE 边界")

    if right_non_crossable_s:
        print(f"右边界: {min(right_non_crossable_s):.2f}m ~ {max(right_non_crossable_s):.2f}m "
              f"(共 {len(right_non_crossable_s)} 个点)")
    else:
        print("右边界: 无 NON_CROSSABLE 边界")
    print()

    # 边界约束强度建议
    print("=" * 80)
    print("边界约束建议")
    print("=" * 80)
    left_constraint_ratio = left_types.get(2, 0) / len(left_boundary)
    right_constraint_ratio = right_types.get(2, 0) / len(right_boundary)

    if right_constraint_ratio > 0.8:
        print("✓ 右边界强约束（>80% NON_CROSSABLE）：")
        print("  适合约束右转/直行轨迹不越过道路边缘")
    elif right_constraint_ratio > 0.3:
        print("○ 右边界中等约束（30%-80% NON_CROSSABLE）：")
        print("  部分路段有物理边界限制")
    else:
        print("  右边界弱约束（<30% NON_CROSSABLE）")

    if left_constraint_ratio > 0.5:
        print("✓ 左边界强约束（>50% NON_CROSSABLE）：")
        print("  限制左转/变道行为")
    elif left_constraint_ratio > 0.2:
        print("○ 左边界中等约束（20%-50% NON_CROSSABLE）")
    else:
        print("  左边界弱约束（<20% NON_CROSSABLE，多为可压虚线）")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="metric cache 根目录，例如 /path/to/metric_cache_mogo_v1"
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="要检查的场景 token ID"
    )
    args = parser.parse_args()

    try:
        cache_path = find_metric_cache(args.cache_root, args.token)
        print(f"找到 metric cache: {cache_path}")
        print()

        cache_data = load_metric_cache(cache_path)
        analyze_boundary_info(cache_data)

        print("=" * 80)
        print("检查完成")
        print("=" * 80)
        print("如果显示边界信息完整，说明 metric cache 已正确保存边界类型。")
        print("可以通过设置 agent.encode_boundary_constraint=true 来启用边界约束编码。")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
