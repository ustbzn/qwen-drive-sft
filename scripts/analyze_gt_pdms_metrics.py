from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable


DEFAULT_EXPERIMENT_DIR = Path(
    "exp/gt_pdm_score_rl_grpo_n4000_selected_tokens_v1/2026.06.29.08.53.16"
)
DEFAULT_CSV = DEFAULT_EXPERIMENT_DIR / "2026.06.29.08.54.23.csv"

MULTIPLICATIVE_METRICS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
]

WEIGHTED_METRICS = OrderedDict(
    [
        ("ego_progress", 5.0),
        ("time_to_collision_within_bound", 5.0),
        ("comfort", 2.0),
        ("driving_direction_compliance", 0.0),
        ("obstacle_clearance", 2.0),
        ("following_safety", 2.0),
        ("centerline_convergence", 1.0),
        ("start_anchor_comfort", 0.0),
    ]
)

SCORE_METRICS = MULTIPLICATIVE_METRICS + list(WEIGHTED_METRICS.keys()) + ["score"]
ACTIVE_WEIGHTED_METRICS = [metric for metric, weight in WEIGHTED_METRICS.items() if weight > 0]
ACTIVE_WEIGHT_SUM = sum(weight for weight in WEIGHTED_METRICS.values() if weight > 0)

METRIC_LABELS = {
    "no_at_fault_collisions": "无责碰撞 NC",
    "drivable_area_compliance": "可行驶区域 DAC",
    "ego_progress": "自车进度 EP",
    "time_to_collision_within_bound": "碰撞时间 TTC",
    "comfort": "舒适性",
    "driving_direction_compliance": "行驶方向",
    "obstacle_clearance": "障碍物间距",
    "following_safety": "跟车安全",
    "centerline_convergence": "中心线收敛",
    "start_anchor_comfort": "起步锚点舒适性",
    "score": "PDMS",
    "multiplicative_gate": "乘法门控",
    "weighted_average": "加权均值",
}

CAUSE_LABELS = {
    "no_at_fault_collisions": "无责碰撞门控",
    "drivable_area_compliance": "可行驶区域门控",
    "ego_progress": "自车进度",
    "time_to_collision_within_bound": "碰撞时间 TTC",
    "comfort": "舒适性",
    "obstacle_clearance": "障碍物间距",
    "following_safety": "跟车安全",
    "centerline_convergence": "中心线收敛",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GT PDMS metric CSV and write a markdown report."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input PDMS CSV.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the input CSV directory.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.8,
        help="Threshold used for low-score attribution.",
    )
    parser.add_argument(
        "--detail-limit",
        type=int,
        default=100,
        help="Number of lowest-score rows to include in markdown.",
    )
    return parser.parse_args()


def parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_float(value: str | None) -> float:
    if value is None or str(value).strip() == "":
        return math.nan
    return float(value)


def format_float(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def format_percent(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{100.0 * value:.{digits}f}%"


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def cause_label(cause: str) -> str:
    return CAUSE_LABELS.get(cause, cause)


def read_rows(csv_path: Path) -> tuple[list[dict[str, float | str | bool]], int, int]:
    rows: list[dict[str, float | str | bool]] = []
    invalid_count = 0
    source_count = 0

    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for raw in reader:
            token = str(raw.get("token", "")).strip()
            if token == "average":
                continue
            source_count += 1

            valid = parse_bool(raw.get("valid"))
            if not valid:
                invalid_count += 1
                continue

            row: dict[str, float | str | bool] = {"token": token, "valid": valid}
            for metric in SCORE_METRICS:
                row[metric] = parse_float(raw.get(metric))
            gt_frames = raw.get("gt_num_future_frames")
            if gt_frames is not None and str(gt_frames).strip():
                row["gt_num_future_frames"] = parse_float(gt_frames)
            rows.append(row)

    if not rows:
        raise ValueError(f"No valid metric rows found in {csv_path}")
    return rows, source_count, invalid_count


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def finite_values(rows: Iterable[dict[str, float | str | bool]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def describe_metric(rows: list[dict[str, float | str | bool]], metric: str) -> dict[str, float | int]:
    values = finite_values(rows, metric)
    count = len(values)
    mean = sum(values) / count if count else math.nan
    variance = sum((value - mean) ** 2 for value in values) / count if count else math.nan
    return {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance) if math.isfinite(variance) else math.nan,
        "min": min(values) if values else math.nan,
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else math.nan,
        "lt_1": sum(value < 1.0 - 1e-12 for value in values),
        "lt_095": sum(value < 0.95 for value in values),
        "lt_08": sum(value < 0.8 for value in values),
        "lt_05": sum(value < 0.5 for value in values),
        "eq_0": sum(abs(value) <= 1e-12 for value in values),
    }


def compute_contributions(row: dict[str, float | str | bool]) -> OrderedDict[str, float]:
    nc = float(row["no_at_fault_collisions"])
    dac = float(row["drivable_area_compliance"])
    gate = nc * dac

    contributions: OrderedDict[str, float] = OrderedDict()
    contributions["no_at_fault_collisions"] = max(0.0, 1.0 - nc)
    contributions["drivable_area_compliance"] = max(0.0, nc * (1.0 - dac))
    for metric, weight in WEIGHTED_METRICS.items():
        if weight <= 0:
            continue
        value = float(row[metric])
        contributions[metric] = max(0.0, gate * (weight / ACTIVE_WEIGHT_SUM) * (1.0 - value))
    return contributions


def enrich_rows(rows: list[dict[str, float | str | bool]]) -> list[dict[str, float | str | bool]]:
    enriched: list[dict[str, float | str | bool]] = []
    for row in rows:
        item = dict(row)
        nc = float(item["no_at_fault_collisions"])
        dac = float(item["drivable_area_compliance"])
        gate = nc * dac
        weighted_average = sum(float(item[metric]) * weight for metric, weight in WEIGHTED_METRICS.items()) / ACTIVE_WEIGHT_SUM
        recomputed_score = gate * weighted_average
        contributions = compute_contributions(item)
        sorted_causes = sorted(contributions.items(), key=lambda pair: pair[1], reverse=True)
        primary_cause, primary_loss = sorted_causes[0]

        item["multiplicative_gate"] = gate
        item["weighted_average"] = weighted_average
        item["recomputed_score"] = recomputed_score
        item["score_recompute_error"] = float(item["score"]) - recomputed_score
        item["total_score_deficit"] = 1.0 - float(item["score"])
        item["primary_cause"] = primary_cause if primary_loss > 1e-12 else "none"
        item["primary_cause_loss"] = primary_loss
        item["top_causes"] = "; ".join(
            f"{cause}:{loss:.6f}" for cause, loss in sorted_causes[:3] if loss > 1e-12
        )
        for cause, loss in contributions.items():
            item[f"loss_{cause}"] = loss
        enriched.append(item)
    return enriched


def score_bucket_counts(rows: list[dict[str, float | str | bool]]) -> list[tuple[str, int, float]]:
    buckets = [
        (0.0, 0.1),
        (0.1, 0.3),
        (0.3, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 0.95),
        (0.95, 1.0000001),
    ]
    total = len(rows)
    result: list[tuple[str, int, float]] = []
    for left, right in buckets:
        count = sum(left <= float(row["score"]) < right for row in rows)
        label = f"[{left:.2f}, {right:.2f})" if right < 1.0 else "[0.95, 1.00]"
        result.append((label, count, count / total if total else math.nan))
    return result


def mean_by_metric(rows: list[dict[str, float | str | bool]], metrics: list[str]) -> dict[str, float]:
    return {metric: describe_metric(rows, metric)["mean"] for metric in metrics}


def read_log_context(experiment_dir: Path) -> dict[str, int]:
    context: dict[str, int] = {}
    log_path = experiment_dir / "log.txt"
    if not log_path.is_file():
        return context

    text = log_path.read_text(errors="replace")
    selected_missing = re.search(r"Selected tokens missing from scene split: (\d+)", text)
    if selected_missing:
        context["missing_scene_tokens"] = int(selected_missing.group(1))
    start_line = re.search(r"Starting GT PDM scoring for (\d+) / (\d+) selected tokens", text)
    if start_line:
        context["evaluated_tokens_from_log"] = int(start_line.group(1))
        context["selected_tokens_from_log"] = int(start_line.group(2))
    failed = re.search(r"Number of failed scenarios: (\d+)", text)
    if failed:
        context["failed_scenarios_from_log"] = int(failed.group(1))
    return context


def write_csv(path: Path, rows: list[dict[str, float | str | bool]]) -> None:
    if not rows:
        return
    ordered_fields = [
        "token",
        "score",
        "multiplicative_gate",
        "weighted_average",
        "primary_cause",
        "primary_cause_loss",
        "top_causes",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "driving_direction_compliance",
        "obstacle_clearance",
        "following_safety",
        "centerline_convergence",
        "start_anchor_comfort",
        "gt_num_future_frames",
        "recomputed_score",
        "score_recompute_error",
        "total_score_deficit",
    ]
    loss_fields = [f"loss_{metric}" for metric in CAUSE_LABELS]
    fields = ordered_fields + [field for field in loss_fields if field in rows[0]]

    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def table_row(values: Iterable[str]) -> str:
    return "| " + " | ".join(values) + " |"


def build_report(
    csv_path: Path,
    out_dir: Path,
    rows: list[dict[str, float | str | bool]],
    enriched_rows: list[dict[str, float | str | bool]],
    source_count: int,
    invalid_count: int,
    score_threshold: float,
    detail_limit: int,
) -> str:
    experiment_dir = csv_path.parent
    log_context = read_log_context(experiment_dir)
    low_rows = [row for row in enriched_rows if float(row["score"]) < score_threshold]
    high_rows = [row for row in enriched_rows if float(row["score"]) >= score_threshold]
    sorted_low = sorted(low_rows, key=lambda row: float(row["score"]))
    sorted_all = sorted(enriched_rows, key=lambda row: float(row["score"]))

    metric_stats = {metric: describe_metric(enriched_rows, metric) for metric in SCORE_METRICS}
    aux_stats = {
        metric: describe_metric(enriched_rows, metric)
        for metric in ["multiplicative_gate", "weighted_average"]
    }

    total_deficit = sum(1.0 - float(row["score"]) for row in enriched_rows) / len(enriched_rows)
    contribution_sums: Counter[str] = Counter()
    for row in enriched_rows:
        for cause in CAUSE_LABELS:
            contribution_sums[cause] += float(row.get(f"loss_{cause}", 0.0))
    contribution_rows = [
        (cause, contribution_sums[cause] / len(enriched_rows))
        for cause in CAUSE_LABELS
        if contribution_sums[cause] > 0
    ]
    contribution_rows.sort(key=lambda pair: pair[1], reverse=True)

    low_contribution_sums: Counter[str] = Counter()
    for row in low_rows:
        for cause in CAUSE_LABELS:
            low_contribution_sums[cause] += float(row.get(f"loss_{cause}", 0.0))
    low_contribution_rows = [
        (cause, low_contribution_sums[cause] / len(low_rows))
        for cause in CAUSE_LABELS
        if low_rows and low_contribution_sums[cause] > 0
    ]
    low_contribution_rows.sort(key=lambda pair: pair[1], reverse=True)

    primary_counts = Counter(str(row["primary_cause"]) for row in low_rows)
    all_means = mean_by_metric(enriched_rows, SCORE_METRICS + ["multiplicative_gate", "weighted_average"])
    low_means = mean_by_metric(low_rows, SCORE_METRICS + ["multiplicative_gate", "weighted_average"]) if low_rows else {}
    high_means = mean_by_metric(high_rows, SCORE_METRICS + ["multiplicative_gate", "weighted_average"]) if high_rows else {}

    lines: list[str] = []
    lines.append("# GT PDMS 指标分析报告")
    lines.append("")
    lines.append("## 输入与样本")
    lines.append("")
    lines.append(f"- 实验目录：`{experiment_dir}`")
    lines.append(f"- 输入 CSV：`{csv_path}`")
    lines.append(f"- 有效评测样本：{len(enriched_rows)}")
    lines.append(f"- CSV 非 average 数据行：{source_count}")
    lines.append(f"- CSV 无效行：{invalid_count}")
    if log_context:
        if "selected_tokens_from_log" in log_context:
            lines.append(f"- 白名单 token：{log_context['selected_tokens_from_log']}")
        if "evaluated_tokens_from_log" in log_context:
            lines.append(f"- 日志记录参与评测 token：{log_context['evaluated_tokens_from_log']}")
        if "missing_scene_tokens" in log_context:
            lines.append(f"- scene split 缺失 token：{log_context['missing_scene_tokens']}")
        if "failed_scenarios_from_log" in log_context:
            lines.append(f"- 日志记录失败场景：{log_context['failed_scenarios_from_log']}")
    lines.append(f"- 低分归因阈值：`score < {score_threshold:g}`，低分样本 {len(low_rows)} / {len(enriched_rows)} ({format_percent(len(low_rows) / len(enriched_rows))})")
    lines.append(f"- 完整逐 token 明细：`{out_dir / 'gt_pdms_enriched_metrics.csv'}`")
    lines.append(f"- 低分逐 token 明细：`{out_dir / 'gt_pdms_low_score_details.csv'}`")
    lines.append("")

    lines.append("## 计分公式")
    lines.append("")
    lines.append("本次 CSV 的 `score` 可由以下公式复算，最大绝对误差见后文一致性检查：")
    lines.append("")
    lines.append("```text")
    lines.append("gate = no_at_fault_collisions * drivable_area_compliance")
    lines.append(
        "weighted_average = (5 * ego_progress + 5 * time_to_collision_within_bound + "
        "2 * comfort + 2 * obstacle_clearance + 2 * following_safety + "
        "1 * centerline_convergence) / 17"
    )
    lines.append("score = gate * weighted_average")
    lines.append("```")
    lines.append("")
    lines.append("`driving_direction_compliance` 和 `start_anchor_comfort` 本次有效权重为 0，只作为诊断指标输出，不直接影响 `score`。")
    lines.append("")

    max_recompute_error = max(abs(float(row["score_recompute_error"])) for row in enriched_rows)
    lines.append("## 总体结论")
    lines.append("")
    lines.append(f"- 平均 PDMS：{format_float(all_means['score'])}")
    lines.append(f"- 平均乘法门控：{format_float(all_means['multiplicative_gate'])}")
    lines.append(f"- 平均加权均值：{format_float(all_means['weighted_average'])}")
    lines.append(f"- 复算最大绝对误差：{max_recompute_error:.12g}")
    if low_rows:
        lines.append(
            f"- `score < {score_threshold:g}` 的低分组平均 PDMS 为 {format_float(low_means['score'])}，"
            f"高分组平均 PDMS 为 {format_float(high_means['score']) if high_rows else 'NA'}。"
        )
        if low_contribution_rows:
            top_low = ", ".join(
                f"{cause_label(cause)} {format_float(loss)}"
                for cause, loss in low_contribution_rows[:4]
            )
            lines.append(f"- 低分组平均损失最大的指标依次为：{top_low}。")
    if contribution_rows:
        top_all = ", ".join(
            f"{cause_label(cause)} {format_float(loss)}"
            for cause, loss in contribution_rows[:4]
        )
        lines.append(f"- 全量平均损失最大的指标依次为：{top_all}。")
    lines.append("")

    lines.append("## 指标统计")
    lines.append("")
    lines.append(table_row(["指标", "均值", "标准差", "min", "p05", "p50", "p95", "max", "<0.95", "<0.8", "=0"]))
    lines.append(table_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    for metric in SCORE_METRICS:
        stats = metric_stats[metric]
        lines.append(
            table_row(
                [
                    metric_label(metric),
                    format_float(float(stats["mean"])),
                    format_float(float(stats["std"])),
                    format_float(float(stats["min"])),
                    format_float(float(stats["p05"])),
                    format_float(float(stats["p50"])),
                    format_float(float(stats["p95"])),
                    format_float(float(stats["max"])),
                    str(stats["lt_095"]),
                    str(stats["lt_08"]),
                    str(stats["eq_0"]),
                ]
            )
        )
    for metric in ["multiplicative_gate", "weighted_average"]:
        stats = aux_stats[metric]
        lines.append(
            table_row(
                [
                    metric_label(metric),
                    format_float(float(stats["mean"])),
                    format_float(float(stats["std"])),
                    format_float(float(stats["min"])),
                    format_float(float(stats["p05"])),
                    format_float(float(stats["p50"])),
                    format_float(float(stats["p95"])),
                    format_float(float(stats["max"])),
                    str(stats["lt_095"]),
                    str(stats["lt_08"]),
                    str(stats["eq_0"]),
                ]
            )
        )
    lines.append("")

    lines.append("## Score 分布")
    lines.append("")
    lines.append(table_row(["score 区间", "样本数", "占比"]))
    lines.append(table_row(["---", "---:", "---:"]))
    for label, count, ratio in score_bucket_counts(enriched_rows):
        lines.append(table_row([label, str(count), format_percent(ratio)]))
    lines.append("")

    lines.append("## 全量低分成因拆解")
    lines.append("")
    lines.append("下表将 `1 - score` 拆成乘法门控损失和加权指标损失。数值越大，说明该指标对平均 PDMS 降低的贡献越大。")
    lines.append("")
    lines.append(table_row(["成因", "平均损失贡献", "占总损失比例"]))
    lines.append(table_row(["---", "---:", "---:"]))
    for cause, loss in contribution_rows:
        share = loss / total_deficit if total_deficit > 0 else math.nan
        lines.append(table_row([cause_label(cause), format_float(loss), format_percent(share)]))
    lines.append(table_row(["总损失 `1 - mean(score)`", format_float(total_deficit), "100.00%"]))
    lines.append("")

    lines.append(f"## 低分组成因拆解：score < {score_threshold:g}")
    lines.append("")
    if not low_rows:
        lines.append("没有低分样本。")
    else:
        low_deficit = sum(1.0 - float(row["score"]) for row in low_rows) / len(low_rows)
        lines.append(table_row(["成因", "低分组平均损失贡献", "占低分组总损失比例"]))
        lines.append(table_row(["---", "---:", "---:"]))
        for cause, loss in low_contribution_rows:
            share = loss / low_deficit if low_deficit > 0 else math.nan
            lines.append(table_row([cause_label(cause), format_float(loss), format_percent(share)]))
        lines.append(table_row(["总损失 `1 - mean(score)`", format_float(low_deficit), "100.00%"]))
        lines.append("")
        lines.append("低分组按单样本最大损失来源统计：")
        lines.append("")
        lines.append(table_row(["主因", "样本数", "低分组占比"]))
        lines.append(table_row(["---", "---:", "---:"]))
        for cause, count in primary_counts.most_common():
            lines.append(table_row([cause_label(cause), str(count), format_percent(count / len(low_rows))]))
    lines.append("")

    lines.append("## 低分组与高分组均值对比")
    lines.append("")
    lines.append(table_row(["指标", "全量均值", f"低分组均值 <{score_threshold:g}", f"高分组均值 >={score_threshold:g}", "低分组差值"]))
    lines.append(table_row(["---", "---:", "---:", "---:", "---:"]))
    for metric in SCORE_METRICS + ["multiplicative_gate", "weighted_average"]:
        low_mean = low_means.get(metric, math.nan)
        high_mean = high_means.get(metric, math.nan)
        delta = low_mean - all_means[metric] if math.isfinite(low_mean) else math.nan
        lines.append(
            table_row(
                [
                    metric_label(metric),
                    format_float(all_means[metric]),
                    format_float(low_mean),
                    format_float(high_mean),
                    format_float(delta),
                ]
            )
        )
    lines.append("")

    lines.append(f"## 最低分样本明细 Top {min(detail_limit, len(sorted_all))}")
    lines.append("")
    lines.append(table_row(["rank", "token", "score", "主因", "gate", "w_avg", "NC", "DAC", "EP", "TTC", "舒适", "障碍间距", "跟车", "中心线"]))
    lines.append(table_row(["---:", "---", "---:", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    for idx, row in enumerate(sorted_all[:detail_limit], start=1):
        lines.append(
            table_row(
                [
                    str(idx),
                    f"`{row['token']}`",
                    format_float(float(row["score"])),
                    cause_label(str(row["primary_cause"])),
                    format_float(float(row["multiplicative_gate"])),
                    format_float(float(row["weighted_average"])),
                    format_float(float(row["no_at_fault_collisions"])),
                    format_float(float(row["drivable_area_compliance"])),
                    format_float(float(row["ego_progress"])),
                    format_float(float(row["time_to_collision_within_bound"])),
                    format_float(float(row["comfort"])),
                    format_float(float(row["obstacle_clearance"])),
                    format_float(float(row["following_safety"])),
                    format_float(float(row["centerline_convergence"])),
                ]
            )
        )
    lines.append("")

    if sorted_low:
        lines.append(f"## 低分样本完整性说明")
        lines.append("")
        lines.append(
            f"Markdown 仅内嵌最低分 Top {min(detail_limit, len(sorted_all))}，"
            f"`gt_pdms_low_score_details.csv` 包含全部 {len(sorted_low)} 个 `score < {score_threshold:g}` 样本，"
            "`gt_pdms_enriched_metrics.csv` 包含全部有效样本及每个指标损失贡献。"
        )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    csv_path = args.csv
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    out_dir = args.out_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, source_count, invalid_count = read_rows(csv_path)
    enriched_rows = enrich_rows(rows)
    low_rows = sorted(
        [row for row in enriched_rows if float(row["score"]) < args.score_threshold],
        key=lambda row: float(row["score"]),
    )
    enriched_rows_sorted = sorted(enriched_rows, key=lambda row: float(row["score"]))

    enriched_csv = out_dir / "gt_pdms_enriched_metrics.csv"
    low_csv = out_dir / "gt_pdms_low_score_details.csv"
    report_path = out_dir / "gt_pdms_metrics_analysis.md"

    write_csv(enriched_csv, enriched_rows_sorted)
    write_csv(low_csv, low_rows)

    report = build_report(
        csv_path=csv_path,
        out_dir=out_dir,
        rows=rows,
        enriched_rows=enriched_rows,
        source_count=source_count,
        invalid_count=invalid_count,
        score_threshold=args.score_threshold,
        detail_limit=args.detail_limit,
    )
    report_path.write_text(report + "\n")

    print(f"Wrote report: {report_path}")
    print(f"Wrote full details: {enriched_csv}")
    print(f"Wrote low-score details: {low_csv}")


if __name__ == "__main__":
    main()
