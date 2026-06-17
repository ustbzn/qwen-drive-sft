#!/usr/bin/env python3
"""Compare PDMS score bucket membership between IL and RL eval CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IL_CSV = Path(
    "exp/recogdrive_qwen3_IL_eval/2026.06.17.04.46.01/2026.06.17.04.49.13.csv"
)
DEFAULT_RL_CSV = Path(
    "exp/recogdrive_qwen3_RL_eval/2026.06.16.10.10.50/2026.06.16.10.14.05.csv"
)
DEFAULT_OUTPUT = Path("vis_result_statistics/eval/pdms_score_bucket_diff_report.txt")
DEFAULT_OPENSCENE_DATA_ROOT = Path(
    "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset"
)
DEFAULT_DATA_SPLIT = "trainval"
TOKEN_FIELD = "token"
SCORE_FIELD = "score"
TRAJECTORY_PATH_FIELD = "trajectory_path"
UNKNOWN_PKL = "Unknown pkl"


BUCKET_ORDER = [
    "0",
    "(0, 0.5]",
    "(0.5, 0.7]",
    "(0.7, 0.8]",
    "(0.8, 0.85]",
    "(0.85, 0.9]",
    "(0.9, 0.95]",
    "(0.95, 1.0)",
    "1.0",
]


@dataclass(frozen=True)
class ScoreRow:
    token: str
    score: float
    bucket: str


def score_bucket(score: float) -> str:
    if score == 0:
        return "0"
    if score <= 0.5:
        return "(0, 0.5]"
    if score <= 0.7:
        return "(0.5, 0.7]"
    if score <= 0.8:
        return "(0.7, 0.8]"
    if score <= 0.85:
        return "(0.8, 0.85]"
    if score <= 0.9:
        return "(0.85, 0.9]"
    if score <= 0.95:
        return "(0.9, 0.95]"
    if score < 1.0:
        return "(0.95, 1.0)"
    return "1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PDMS score bucket differences between IL and RL CSVs."
    )
    parser.add_argument(
        "--il-csv",
        type=Path,
        default=DEFAULT_IL_CSV,
        help=f"IL eval CSV. Default: {DEFAULT_IL_CSV}",
    )
    parser.add_argument(
        "--rl-csv",
        type=Path,
        default=DEFAULT_RL_CSV,
        help=f"RL eval CSV. Default: {DEFAULT_RL_CSV}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Text report output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--navsim-logs-dir",
        "--pkl-dir",
        dest="navsim_logs_dirs",
        type=Path,
        action="append",
        default=[],
        help=(
            "Directory containing NAVSIM log .pkl files. Can be repeated. "
            "Default: infer from trajectory_path, then OPENSCENE_DATA_ROOT."
        ),
    )
    return parser.parse_args()


def parse_score(value: str, csv_path: Path, token: str) -> float:
    try:
        score = float(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid score for token {token} in {csv_path}: {value!r}") from exc
    if not math.isfinite(score):
        raise SystemExit(f"Non-finite score for token {token} in {csv_path}: {value!r}")
    return score


def load_score_rows(csv_path: Path) -> dict[str, ScoreRow]:
    rows: dict[str, ScoreRow] = {}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV is empty: {csv_path}")

        required = {TOKEN_FIELD, SCORE_FIELD}
        missing = required - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"CSV missing columns {sorted(missing)}: {csv_path}")

        for row in reader:
            token = row[TOKEN_FIELD]
            if token in rows:
                raise SystemExit(f"Duplicate token {token} in {csv_path}")

            score = parse_score(row[SCORE_FIELD], csv_path, token)
            rows[token] = ScoreRow(token=token, score=score, bucket=score_bucket(score))

    return rows


def infer_data_roots_from_csv(csv_path: Path) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or TRAJECTORY_PATH_FIELD not in reader.fieldnames:
            return roots

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

            root = Path(*parts[:exp_index])
            if root not in seen:
                roots.append(root)
                seen.add(root)

    return roots


def unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        unique.append(path)
        seen.add(path)
    return unique


def resolve_navsim_logs_dirs(
    explicit_dirs: list[Path],
    csv_paths: list[Path],
) -> tuple[list[Path], bool]:
    if explicit_dirs:
        return unique_paths(explicit_dirs), True

    inferred_dirs: list[Path] = []
    for csv_path in csv_paths:
        for data_root in infer_data_roots_from_csv(csv_path):
            inferred_dirs.append(data_root / "dataset" / "navsim_logs" / DEFAULT_DATA_SPLIT)

    if inferred_dirs:
        return unique_paths(inferred_dirs), False

    openscene_root = Path(
        os.environ.get("OPENSCENE_DATA_ROOT", str(DEFAULT_OPENSCENE_DATA_ROOT))
    )
    return [openscene_root / "navsim_logs" / DEFAULT_DATA_SPLIT], False


def load_token_pkl_names(
    navsim_logs_dirs: list[Path],
    tokens: set[str],
) -> dict[str, str]:
    token_to_pkl: dict[str, str] = {}
    remaining_tokens = set(tokens)

    for navsim_logs_dir in navsim_logs_dirs:
        for pkl_path in sorted(navsim_logs_dir.glob("*.pkl")):
            if not remaining_tokens:
                return token_to_pkl

            with pkl_path.open("rb") as f:
                frames = pickle.load(f)

            if not isinstance(frames, list):
                continue

            for frame in frames:
                if not isinstance(frame, dict):
                    continue

                token = frame.get(TOKEN_FIELD)
                if token in remaining_tokens:
                    token_to_pkl[token] = pkl_path.name
                    remaining_tokens.remove(token)

    return token_to_pkl


def bucket_tokens(rows_by_token: dict[str, ScoreRow], bucket: str) -> set[str]:
    return {token for token, row in rows_by_token.items() if row.bucket == bucket}


def describe_model_token(rows_by_token: dict[str, ScoreRow], token: str) -> str:
    row = rows_by_token.get(token)
    if row is None:
        return "missing"
    return f"{row.bucket}, score={row.score:.12g}"


def group_tokens_by_pkl(
    tokens: list[str],
    token_to_pkl: dict[str, str],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for token in tokens:
        grouped.setdefault(token_to_pkl.get(token, UNKNOWN_PKL), []).append(token)
    return grouped


def ordered_pkl_names(grouped: dict[str, list[str]]) -> list[str]:
    names = sorted(name for name in grouped if name != UNKNOWN_PKL)
    if UNKNOWN_PKL in grouped:
        names.append(UNKNOWN_PKL)
    return names


def format_token_section(
    title: str,
    tokens: list[str],
    token_to_pkl: dict[str, str],
    il_rows: dict[str, ScoreRow],
    rl_rows: dict[str, ScoreRow],
) -> list[str]:
    lines = [f"{title}: {len(tokens)}"]
    if not tokens:
        lines.append("  (none)")
        return lines

    grouped = group_tokens_by_pkl(tokens, token_to_pkl)
    for pkl_name in ordered_pkl_names(grouped):
        lines.append(pkl_name)
        for token in sorted(grouped[pkl_name]):
            il_desc = describe_model_token(il_rows, token)
            rl_desc = describe_model_token(rl_rows, token)
            lines.append(f"  {token} | IL: {il_desc} | RL: {rl_desc}")

    return lines


def format_bucket_summary(
    il_rows: dict[str, ScoreRow],
    rl_rows: dict[str, ScoreRow],
) -> tuple[list[str], dict[str, tuple[list[str], list[str]]]]:
    lines = [
        "Bucket summary",
        "bucket,il_count,rl_count,rl_minus_il,rl_only_in_bucket,il_only_in_bucket",
    ]
    bucket_diffs: dict[str, tuple[list[str], list[str]]] = {}

    for bucket in BUCKET_ORDER:
        il_tokens = bucket_tokens(il_rows, bucket)
        rl_tokens = bucket_tokens(rl_rows, bucket)
        rl_only = sorted(rl_tokens - il_tokens)
        il_only = sorted(il_tokens - rl_tokens)
        bucket_diffs[bucket] = (rl_only, il_only)

        lines.append(
            ",".join(
                [
                    bucket,
                    str(len(il_tokens)),
                    str(len(rl_tokens)),
                    str(len(rl_tokens) - len(il_tokens)),
                    str(len(rl_only)),
                    str(len(il_only)),
                ]
            )
        )

    return lines, bucket_diffs


def build_report(
    il_csv: Path,
    rl_csv: Path,
    il_rows: dict[str, ScoreRow],
    rl_rows: dict[str, ScoreRow],
    navsim_logs_dirs: list[Path],
    token_to_pkl: dict[str, str],
    bucket_diffs: dict[str, tuple[list[str], list[str]]],
    explicit_pkl_dirs: bool,
) -> str:
    il_tokens = set(il_rows)
    rl_tokens = set(rl_rows)
    diff_tokens = set()
    for rl_only, il_only in bucket_diffs.values():
        diff_tokens.update(rl_only)
        diff_tokens.update(il_only)

    lines = [
        "PDMS score bucket diff report",
        "",
        f"IL CSV: {il_csv}",
        f"RL CSV: {rl_csv}",
        f"IL tokens: {len(il_tokens)}",
        f"RL tokens: {len(rl_tokens)}",
        f"Common tokens: {len(il_tokens & rl_tokens)}",
        f"IL-only tokens: {len(il_tokens - rl_tokens)}",
        f"RL-only tokens: {len(rl_tokens - il_tokens)}",
        "",
        "PKL search dirs:",
    ]
    lines.extend(f"- {path}" for path in navsim_logs_dirs)
    lines.extend(
        [
            f"PKL dirs explicit: {explicit_pkl_dirs}",
            f"Diff tokens with pkl: {len(token_to_pkl)}/{len(diff_tokens)}",
            "",
        ]
    )

    summary_lines, _ = format_bucket_summary(il_rows, rl_rows)
    lines.extend(summary_lines)

    lines.extend(["", "Bucket details"])
    for bucket in BUCKET_ORDER:
        rl_only, il_only = bucket_diffs[bucket]
        lines.extend(["", f"Bucket {bucket}"])
        lines.extend(
            format_token_section(
                "RL-only in bucket",
                rl_only,
                token_to_pkl,
                il_rows,
                rl_rows,
            )
        )
        lines.extend(
            format_token_section(
                "IL-only in bucket",
                il_only,
                token_to_pkl,
                il_rows,
                rl_rows,
            )
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    if not args.il_csv.is_file():
        raise SystemExit(f"IL CSV not found: {args.il_csv}")
    if not args.rl_csv.is_file():
        raise SystemExit(f"RL CSV not found: {args.rl_csv}")

    il_rows = load_score_rows(args.il_csv)
    rl_rows = load_score_rows(args.rl_csv)
    summary_lines, bucket_diffs = format_bucket_summary(il_rows, rl_rows)

    diff_tokens: set[str] = set()
    for rl_only, il_only in bucket_diffs.values():
        diff_tokens.update(rl_only)
        diff_tokens.update(il_only)

    navsim_logs_dirs, explicit_pkl_dirs = resolve_navsim_logs_dirs(
        args.navsim_logs_dirs,
        [args.il_csv, args.rl_csv],
    )
    valid_dirs = [path for path in navsim_logs_dirs if path.is_dir()]
    missing_dirs = [path for path in navsim_logs_dirs if not path.is_dir()]

    if explicit_pkl_dirs and missing_dirs:
        raise SystemExit(f"NAVSIM logs dir not found: {missing_dirs[0]}")
    if missing_dirs:
        for path in missing_dirs:
            print(f"NAVSIM logs dir not found, pkl names may be Unknown: {path}")

    token_to_pkl = load_token_pkl_names(valid_dirs, diff_tokens) if valid_dirs else {}

    report = build_report(
        args.il_csv,
        args.rl_csv,
        il_rows,
        rl_rows,
        navsim_logs_dirs,
        token_to_pkl,
        bucket_diffs,
        explicit_pkl_dirs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Diff tokens with pkl: {len(token_to_pkl)}/{len(diff_tokens)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
