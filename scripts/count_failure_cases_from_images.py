#!/usr/bin/env python3
"""Count collision and lane-boundary cases for PNG tokens listed in an eval CSV."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CSV = Path(
    "exp/recogdrive_qwen3_hdmap_ddim_IL_eval/2026.06.25.10.01.35/2026.06.25.10.12.38.csv"
)
DEFAULT_TOKEN_REPORT = "failure_case_token_report.txt"
DEFAULT_OPENSCENE_DATA_ROOT = Path(
    "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/dataset"
)
DEFAULT_DATA_SPLIT = "trainval"
TOKEN_IMAGE_RE = re.compile(
    r"^(?P<token>[0-9a-fA-F]{32})(?:_(?:score|csvscore)[^/]*)?\.png$"
)
COLLISION_FIELD = "no_at_fault_collisions"
LANE_FIELD = "drivable_area_compliance"
TRAJECTORY_PATH_FIELD = "trajectory_path"


@dataclass(frozen=True)
class MetricRow:
    token: str
    no_at_fault_collisions: float
    drivable_area_compliance: float


@dataclass(frozen=True)
class CountSummary:
    image_count: int
    unique_token_count: int
    matched_token_count: int
    missing_csv_count: int
    duplicate_image_token_count: int
    collision_count: int
    lane_boundary_count: int
    collision_and_lane_boundary_count: int


@dataclass(frozen=True)
class CaseGroups:
    collision_tokens: list[str]
    lane_boundary_tokens: list[str]
    collision_and_lane_boundary_tokens: list[str]
    collision_only_tokens: list[str]
    lane_boundary_only_tokens: list[str]
    unclassified_tokens: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count collision and lane-boundary/off-drivable-area cases by matching "
            "PNG filename tokens against an eval CSV."
        )
    )
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=DEFAULT_CSV,
        help=f"Eval CSV with token and metric columns. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("vis_output/eval/2026.06.25.10.01.35_qwen3_hdmap_RL_low_score"),
        help="Directory containing token-named PNG files. Default: current directory.",
    )
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=1.0,
        help=(
            "Count collision when no_at_fault_collisions is below this value. "
            "Default: 1.0"
        ),
    )
    parser.add_argument(
        "--lane-threshold",
        type=float,
        default=1.0,
        help=(
            "Count lane-boundary case when drivable_area_compliance is below this "
            "value. Use 0.050000001 for hard off-road only. Default: 1.0"
        ),
    )
    parser.add_argument(
        "--write-cases",
        type=Path,
        help="Optional path to write matched per-token case labels as CSV.",
    )
    parser.add_argument(
        "--token-report",
        type=Path,
        help=(
            "Path to write token groups as text. Default: "
            f"<image-dir>/{DEFAULT_TOKEN_REPORT}"
        ),
    )
    parser.add_argument(
        "--navsim-logs-dir",
        "--pkl-dir",
        dest="navsim_logs_dir",
        type=Path,
        help=(
            "Directory containing NAVSIM log .pkl files. Default: "
            "OPENSCENE_DATA_ROOT/navsim_logs/trainval, matching "
            "scripts/vis_scenario_video.py."
        ),
    )
    parser.add_argument(
        "--no-token-report",
        action="store_true",
        help="Do not write the token-group text report.",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Print PNG tokens that are missing from the CSV.",
    )
    return parser.parse_args()


def parse_metric(value: str, field: str, token: str) -> float:
    try:
        metric = float(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid {field} for token {token}: {value!r}") from exc
    if not math.isfinite(metric):
        raise SystemExit(f"Non-finite {field} for token {token}: {value!r}")
    return metric


def load_metric_rows(csv_path: Path) -> dict[str, MetricRow]:
    rows: dict[str, MetricRow] = {}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV is empty: {csv_path}")

        required = {"token", COLLISION_FIELD, LANE_FIELD}
        missing = required - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"CSV missing columns {sorted(missing)}: {csv_path}")

        for row in reader:
            token = row["token"]
            rows[token] = MetricRow(
                token=token,
                no_at_fault_collisions=parse_metric(
                    row[COLLISION_FIELD], COLLISION_FIELD, token
                ),
                drivable_area_compliance=parse_metric(row[LANE_FIELD], LANE_FIELD, token),
            )

    return rows


def token_from_png(path: Path) -> str | None:
    match = TOKEN_IMAGE_RE.match(path.name)
    if match is None:
        return None
    return match.group("token")


def load_png_tokens(image_dir: Path) -> tuple[list[str], int]:
    tokens: list[str] = []
    seen: set[str] = set()
    duplicate_count = 0

    for path in sorted(image_dir.glob("*.png")):
        token = token_from_png(path)
        if token is None:
            continue
        tokens.append(token)
        if token in seen:
            duplicate_count += 1
        else:
            seen.add(token)

    return tokens, duplicate_count


def infer_data_root_from_csv(csv_path: Path) -> Path | None:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or TRAJECTORY_PATH_FIELD not in reader.fieldnames:
            return None

        for row in reader:
            trajectory_path = row.get(TRAJECTORY_PATH_FIELD, "")
            if not trajectory_path:
                continue

            parts = Path(trajectory_path).parts
            if "exp" not in parts:
                continue

            exp_index = parts.index("exp")
            if exp_index == 0:
                continue
            return Path(*parts[:exp_index])

    return None


def default_navsim_logs_dir(csv_path: Path) -> Path:
    inferred_data_root = infer_data_root_from_csv(csv_path)
    if inferred_data_root is not None:
        return inferred_data_root / "dataset" / "navsim_logs" / DEFAULT_DATA_SPLIT

    openscene_root = Path(
        os.environ.get("OPENSCENE_DATA_ROOT", str(DEFAULT_OPENSCENE_DATA_ROOT))
    )
    return openscene_root / "navsim_logs" / DEFAULT_DATA_SPLIT


def resolve_navsim_logs_dir(navsim_logs_dir: Path | None, csv_path: Path) -> Path:
    if navsim_logs_dir is not None:
        return navsim_logs_dir

    return default_navsim_logs_dir(csv_path)


def load_token_pkl_names(
    navsim_logs_dir: Path,
    tokens: set[str],
) -> dict[str, str]:
    token_to_pkl: dict[str, str] = {}
    remaining_tokens = set(tokens)

    for pkl_path in sorted(navsim_logs_dir.glob("*.pkl")):
        if not remaining_tokens:
            break

        with pkl_path.open("rb") as f:
            frames = pickle.load(f)

        if not isinstance(frames, list):
            continue

        for frame in frames:
            if not isinstance(frame, dict):
                continue

            token = frame.get("token")
            if token in remaining_tokens:
                token_to_pkl[token] = pkl_path.name
                remaining_tokens.remove(token)

    return token_to_pkl


def is_below(metric: float, threshold: float) -> bool:
    return metric < threshold


def summarize_cases(
    tokens: list[str],
    duplicate_image_token_count: int,
    rows_by_token: dict[str, MetricRow],
    collision_threshold: float,
    lane_threshold: float,
) -> tuple[CountSummary, CaseGroups, list[dict[str, str]]]:
    unique_tokens = sorted(set(tokens))
    matched_rows: list[MetricRow] = []
    case_rows: list[dict[str, str]] = []
    collision_tokens: list[str] = []
    lane_boundary_tokens: list[str] = []
    collision_and_lane_boundary_tokens: list[str] = []
    collision_only_tokens: list[str] = []
    lane_boundary_only_tokens: list[str] = []
    unclassified_tokens: list[str] = []

    missing_csv_count = 0
    for token in unique_tokens:
        row = rows_by_token.get(token)
        if row is None:
            missing_csv_count += 1
            continue
        matched_rows.append(row)

        has_collision = is_below(row.no_at_fault_collisions, collision_threshold)
        has_lane_boundary = is_below(row.drivable_area_compliance, lane_threshold)
        if has_collision:
            collision_tokens.append(row.token)
        if has_lane_boundary:
            lane_boundary_tokens.append(row.token)
        if has_collision and has_lane_boundary:
            collision_and_lane_boundary_tokens.append(row.token)
        elif has_collision:
            collision_only_tokens.append(row.token)
        elif has_lane_boundary:
            lane_boundary_only_tokens.append(row.token)
        else:
            unclassified_tokens.append(row.token)

        case_rows.append(
            {
                "token": row.token,
                COLLISION_FIELD: f"{row.no_at_fault_collisions:.12g}",
                LANE_FIELD: f"{row.drivable_area_compliance:.12g}",
                "collision_case": str(has_collision),
                "lane_boundary_case": str(has_lane_boundary),
            }
        )

    collision_count = len(collision_tokens)
    lane_boundary_count = len(lane_boundary_tokens)
    both_count = len(collision_and_lane_boundary_tokens)

    return (
        CountSummary(
            image_count=len(tokens),
            unique_token_count=len(unique_tokens),
            matched_token_count=len(matched_rows),
            missing_csv_count=missing_csv_count,
            duplicate_image_token_count=duplicate_image_token_count,
            collision_count=collision_count,
            lane_boundary_count=lane_boundary_count,
            collision_and_lane_boundary_count=both_count,
        ),
        CaseGroups(
            collision_tokens=collision_tokens,
            lane_boundary_tokens=lane_boundary_tokens,
            collision_and_lane_boundary_tokens=collision_and_lane_boundary_tokens,
            collision_only_tokens=collision_only_tokens,
            lane_boundary_only_tokens=lane_boundary_only_tokens,
            unclassified_tokens=unclassified_tokens,
        ),
        case_rows,
    )


def format_tokens_by_pkl(tokens: list[str], token_to_pkl: dict[str, str]) -> list[str]:
    tokens_by_pkl: dict[str, list[str]] = {}
    missing_pkl_tokens: list[str] = []

    for token in tokens:
        pkl_name = token_to_pkl.get(token)
        if pkl_name is None:
            missing_pkl_tokens.append(token)
            continue
        tokens_by_pkl.setdefault(pkl_name, []).append(token)

    lines: list[str] = []
    for pkl_name in sorted(tokens_by_pkl):
        lines.append(pkl_name)
        lines.extend(f"  {token}" for token in tokens_by_pkl[pkl_name])

    if missing_pkl_tokens:
        lines.append("Unknown pkl")
        lines.extend(f"  {token}" for token in missing_pkl_tokens)

    return lines


def format_token_group(
    title: str,
    tokens: list[str],
    token_to_pkl: dict[str, str] | None = None,
) -> str:
    lines = [f"{title}: {len(tokens)}"]
    if token_to_pkl is None:
        lines.extend(tokens)
    else:
        lines.extend(format_tokens_by_pkl(tokens, token_to_pkl))
    return "\n".join(lines)


def write_token_report(
    path: Path,
    summary: CountSummary,
    groups: CaseGroups,
    collision_threshold: float,
    lane_threshold: float,
    token_to_pkl: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exclusive_total = (
        len(groups.collision_only_tokens)
        + len(groups.lane_boundary_only_tokens)
        + len(groups.collision_and_lane_boundary_tokens)
        + len(groups.unclassified_tokens)
    )
    content = "\n\n".join(
        [
            "Failure case token report",
            (
                f"Matched tokens: {summary.matched_token_count}\n"
                f"Exclusive category total: {exclusive_total}\n"
                f"Collision threshold: {COLLISION_FIELD} < {collision_threshold:g}\n"
                f"Lane-boundary threshold: {LANE_FIELD} < {lane_threshold:g}"
            ),
            "Exclusive categories",
            format_token_group(
                "Collision only",
                groups.collision_only_tokens,
                token_to_pkl,
            ),
            format_token_group(
                "Lane-boundary only",
                groups.lane_boundary_only_tokens,
                token_to_pkl,
            ),
            format_token_group(
                "Collision + lane-boundary",
                groups.collision_and_lane_boundary_tokens,
                token_to_pkl,
            ),
            format_token_group("Unclassified", groups.unclassified_tokens, token_to_pkl),
        ]
    )
    path.write_text(f"{content}\n", encoding="utf-8")


def write_case_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token",
                COLLISION_FIELD,
                LANE_FIELD,
                "collision_case",
                "lane_boundary_case",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.collision_threshold <= 0:
        raise SystemExit("--collision-threshold must be positive.")
    if args.lane_threshold <= 0:
        raise SystemExit("--lane-threshold must be positive.")
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")
    if not args.image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {args.image_dir}")

    rows_by_token = load_metric_rows(args.csv)
    tokens, duplicate_count = load_png_tokens(args.image_dir)
    summary, groups, case_rows = summarize_cases(
        tokens=tokens,
        duplicate_image_token_count=duplicate_count,
        rows_by_token=rows_by_token,
        collision_threshold=args.collision_threshold,
        lane_threshold=args.lane_threshold,
    )

    print(f"CSV: {args.csv}")
    print(f"Image dir: {args.image_dir}")
    print(
        f"PNG images: {summary.image_count}, unique tokens: {summary.unique_token_count}, "
        f"matched CSV tokens: {summary.matched_token_count}"
    )
    print(
        f"Collision cases ({COLLISION_FIELD} < {args.collision_threshold:g}): "
        f"{summary.collision_count}"
    )
    print(
        f"Lane-boundary cases ({LANE_FIELD} < {args.lane_threshold:g}): "
        f"{summary.lane_boundary_count}"
    )
    print(f"Collision + lane-boundary overlap: {summary.collision_and_lane_boundary_count}")
    print(
        "Exclusive categories: "
        f"collision only {len(groups.collision_only_tokens)}, "
        f"lane-boundary only {len(groups.lane_boundary_only_tokens)}, "
        f"both {len(groups.collision_and_lane_boundary_tokens)}, "
        f"unclassified {len(groups.unclassified_tokens)}"
    )
    print(
        f"Missing CSV tokens: {summary.missing_csv_count}, "
        f"duplicate image tokens: {summary.duplicate_image_token_count}"
    )

    if args.show_missing and summary.missing_csv_count:
        missing_tokens = sorted(set(tokens) - rows_by_token.keys())
        print("Missing tokens:")
        for token in missing_tokens:
            print(token)

    if args.write_cases is not None:
        write_case_rows(args.write_cases, case_rows)
        print(f"Wrote {args.write_cases}")

    token_to_pkl: dict[str, str] = {}
    report_tokens = set(groups.collision_only_tokens)
    report_tokens.update(groups.lane_boundary_only_tokens)
    report_tokens.update(groups.collision_and_lane_boundary_tokens)
    report_tokens.update(groups.unclassified_tokens)

    navsim_logs_dir = resolve_navsim_logs_dir(args.navsim_logs_dir, args.csv)
    if navsim_logs_dir.is_dir():
        token_to_pkl = load_token_pkl_names(navsim_logs_dir, report_tokens)
        print(
            f"Matched token pkls: {len(token_to_pkl)}/{len(report_tokens)} "
            f"from {navsim_logs_dir}"
        )
    elif args.navsim_logs_dir is not None:
        raise SystemExit(f"NAVSIM logs dir not found: {navsim_logs_dir}")
    else:
        print(f"NAVSIM logs dir not found, pkl names will be Unknown: {navsim_logs_dir}")

    if not args.no_token_report:
        token_report = args.token_report or args.image_dir / DEFAULT_TOKEN_REPORT
        write_token_report(
            token_report,
            summary,
            groups,
            args.collision_threshold,
            args.lane_threshold,
            token_to_pkl,
        )
        print(f"Wrote {token_report}")


if __name__ == "__main__":
    main()
