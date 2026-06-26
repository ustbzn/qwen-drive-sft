#!/usr/bin/env python3
"""Copy low-score visualization images selected from an eval CSV."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CSV = Path(
    "exp/recogdrive_qwen3_hdmap_ddim_IL_eval/2026.06.25.10.01.35/2026.06.25.10.12.38.csv"
)
DEFAULT_IMAGE_DIR = Path("vis_output/eval/2026.06.25.10.01.35_qwen3_hdmap_RL")
DEFAULT_OUTPUT_DIR = Path("vis_output/eval/2026.06.25.10.01.35_qwen3_hdmap_RL_low_score")
DEFAULT_MANIFEST = "manifest.csv"
IMAGE_SCORE_RE = re.compile(
    r"_score(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+|nan|NaN|NA))\.png$"
)


@dataclass(frozen=True)
class LowScoreRow:
    token: str
    score: float


@dataclass(frozen=True)
class SourceImage:
    path: Path
    score: float | None


@dataclass(frozen=True)
class CopySummary:
    copied: int = 0
    skipped: int = 0
    missing: int = 0
    source_score_out_of_range: int = 0
    invalid_source_score: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy images whose CSV score and image filename score are both "
            "0 or in (0, max-score]."
        )
    )
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=DEFAULT_CSV,
        help=f"Eval CSV with 'token' and 'score' columns. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory containing '<token>_score*.png'. Default: {DEFAULT_IMAGE_DIR}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to copy selected images into. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.5,
        help="Upper bound for the non-zero low-score bucket. Default: 0.5",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without copying files.",
    )
    return parser.parse_args()


def is_low_score(score: float, max_score: float) -> bool:
    return math.isfinite(score) and 0 <= score <= max_score


def score_bucket(score: float, max_score: float) -> str:
    return "0" if score == 0 else f"(0, {max_score:g}]"


def format_score(score: float | None) -> str:
    if score is None or not math.isfinite(score):
        return ""
    return f"{score:.12g}"


def load_low_score_rows(csv_path: Path, max_score: float) -> list[LowScoreRow]:
    rows: list[LowScoreRow] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV is empty: {csv_path}")
        missing = {"token", "score"} - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"CSV missing columns {sorted(missing)}: {csv_path}")

        for row in reader:
            score = float(row["score"])
            if is_low_score(score, max_score):
                rows.append(LowScoreRow(token=row["token"], score=score))

    return rows


def parse_image_score(path: Path) -> float | None:
    match = IMAGE_SCORE_RE.search(path.name)
    if match is None:
        return None

    score_text = match.group("score")
    if score_text == "NA":
        return None

    score = float(score_text)
    if not math.isfinite(score):
        return None
    return score


def find_image(image_dir: Path, token: str) -> SourceImage | None:
    matches = sorted(image_dir.glob(f"{token}_score*.png"))
    if not matches:
        return None
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise SystemExit(f"Multiple images found for token {token}: {names}")

    image_path = matches[0]
    return SourceImage(path=image_path, score=parse_image_score(image_path))


def manifest_entry(
    row: LowScoreRow,
    max_score: float,
    status: str,
    source: SourceImage | None = None,
    destination: Path | None = None,
) -> dict[str, str]:
    return {
        "token": row.token,
        "csv_score": format_score(row.score),
        "source_score": format_score(source.score) if source is not None else "",
        "bucket": score_bucket(row.score, max_score),
        "source": str(source.path) if source is not None else "",
        "destination": str(destination) if destination is not None else "",
        "status": status,
    }


def copy_images(
    rows: list[LowScoreRow],
    image_dir: Path,
    output_dir: Path,
    max_score: float,
    overwrite: bool,
    dry_run: bool,
) -> CopySummary:
    copied = 0
    skipped = 0
    missing = 0
    source_score_out_of_range = 0
    invalid_source_score = 0
    manifest_rows: list[dict[str, str]] = []

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        source = find_image(image_dir, row.token)
        if source is None:
            missing += 1
            manifest_rows.append(
                manifest_entry(row, max_score, status="missing_source")
            )
            continue

        if source.score is None:
            invalid_source_score += 1
            manifest_rows.append(
                manifest_entry(
                    row,
                    max_score,
                    status="invalid_source_score",
                    source=source,
                )
            )
            continue

        if not is_low_score(source.score, max_score):
            source_score_out_of_range += 1
            manifest_rows.append(
                manifest_entry(
                    row,
                    max_score,
                    status="source_score_out_of_range",
                    source=source,
                )
            )
            continue

        dst = output_dir / f"{row.token}_csvscore{row.score:.3f}.png"
        status = "copied"
        if dst.exists() and not overwrite:
            skipped += 1
            status = "exists"
        else:
            copied += 1
            if not dry_run:
                shutil.copy2(source.path, dst)

        manifest_rows.append(
            manifest_entry(
                row,
                max_score,
                status=status,
                source=source,
                destination=dst,
            )
        )

    if not dry_run:
        manifest_path = output_dir / DEFAULT_MANIFEST
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "token",
                    "csv_score",
                    "source_score",
                    "bucket",
                    "source",
                    "destination",
                    "status",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    return CopySummary(
        copied=copied,
        skipped=skipped,
        missing=missing,
        source_score_out_of_range=source_score_out_of_range,
        invalid_source_score=invalid_source_score,
    )


def main() -> None:
    args = parse_args()

    if args.max_score <= 0:
        raise SystemExit("--max-score must be positive.")
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")
    if not args.image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {args.image_dir}")

    rows = load_low_score_rows(args.csv, args.max_score)
    zero_count = sum(1 for row in rows if row.score == 0)
    low_count = len(rows) - zero_count
    summary = copy_images(
        rows=rows,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        max_score=args.max_score,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    action = "Would copy" if args.dry_run else "Copied"
    print(
        f"{action} {summary.copied} images to {args.output_dir} "
        f"(csv score=0: {zero_count}, 0<csv score<={args.max_score:g}: {low_count}, "
        f"existing: {summary.skipped}, missing: {summary.missing}, "
        f"source score out of range: {summary.source_score_out_of_range}, "
        f"invalid source score: {summary.invalid_source_score})."
    )
    if not args.dry_run:
        print(f"Wrote {args.output_dir / DEFAULT_MANIFEST}")


if __name__ == "__main__":
    main()
