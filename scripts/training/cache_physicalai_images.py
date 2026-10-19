#!/usr/bin/env python3
"""Cache the PhysicalAI frames consumed by Qwen-Drive as resized, lossless RGB.

Requires a completed ``extract_physicalai_av.py --video-check frames`` output.
Each uncompressed tar shard contains PNG images; its JSON index records byte
offsets for random reads. The cached images are already resized onto the Qwen
patch grid, so a future reader must skip the processor's resize steps.

Example:
    python scripts/training/cache_physicalai_images.py \
        --root /pcpt/pcpt/data/VLA/PhysicalAI-AV_unzip \
        --dest /path/to/physicalai-image-cache --workers 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, __version__ as pillow_version
from torchvision import __version__ as torchvision_version
from navsim.agents.qwen_drive_planner.modeling.scene import resize_image_to_grid
from navsim.planning.training.physicalai_dataset import (
    PHYSICALAI_CAMERA_BY_VIEW,
    PhysicalAIDataset,
)


LOG = logging.getLogger("physicalai_image_cache")
DEFAULT_ROOT = Path("/pcpt/pcpt/data/VLA/PhysicalAI-AV_unzip")
# Cache every PhysicalAI camera exposed by the adapter. The extraction step
# must have been run with the same six-view selection first.
DEFAULT_VIEWS = ("cam_f0", "cam_l0", "cam_r0", "cam_l1", "cam_r1", "cam_b0")
CACHE_VERSION = 1


def _signature(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _source_signatures(root: Path, clip_id: str, views: tuple[str, ...]) -> dict[str, list[list[int]]]:
    signatures = {}
    for view in views:
        camera = PHYSICALAI_CAMERA_BY_VIEW[view]
        prefix = root / "camera" / camera / f"{clip_id}.{camera}"
        signatures[view] = [
            _signature(Path(f"{prefix}.timestamps.parquet")),
            _signature(Path(f"{prefix}.mp4")),
        ]
    return signatures


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".tmp-", delete=False) as stream:
        temporary_path = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, path)


def _manifest_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resized_image(frame, *, current: bool, resize: dict[str, int]) -> Image.Image:
    image = Image.fromarray(frame.copy()).convert("RGB")
    factor = resize["patch_size"] * resize["merge_size"]
    return resize_image_to_grid(
        image,
        resize["current_image_pixels"] if current else resize["history_image_pixels"],
        factor,
    )


def _reusable(
    tar_path: Path, index_path: Path, plan: dict[str, Any],
    clip_ids: list[str], signatures: dict[str, Any],
) -> dict[str, Any] | None:
    if not tar_path.is_file() or not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text())
        stat = tar_path.stat()
    except (OSError, ValueError):
        return None
    if (index.get("plan") == plan
            and index.get("clip_ids") == clip_ids
            and index.get("source_signatures") == signatures
            and index.get("tar_bytes") == stat.st_size
            and index.get("tar_mtime_ns") == stat.st_mtime_ns):
        return index
    return None


def _cache_shard(
    dataset: PhysicalAIDataset, split: str, shard_number: int,
    clips: list[tuple[str, int]], dest: Path, plan: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    stem = f"{split}-{shard_number:06d}"
    tar_path = dest / "shards" / f"{stem}.tar"
    index_path = dest / "shards" / f"{stem}.json"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    views = tuple(plan["views"])
    clip_ids = [clip_id for clip_id, _ in clips]
    signatures = {
        clip_id: _source_signatures(dataset.root, clip_id, views)
        for clip_id in clip_ids
    }
    old_index = _reusable(tar_path, index_path, plan, clip_ids, signatures)
    if old_index is not None:
        return old_index, True

    with tempfile.NamedTemporaryFile("w+b", dir=tar_path.parent, prefix=f".{stem}-", suffix=".tmp", delete=False) as stream:
        temporary_path = Path(stream.name)
        records = []
        try:
            with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for clip_id, chunk in clips:
                    record = {"clip_id": clip_id, "chunk": chunk, "frames": {}}
                    for view in views:
                        camera = PHYSICALAI_CAMERA_BY_VIEW[view]
                        group = dataset._camera_frames(clip_id, chunk, camera)[0].group
                        try:
                            frames = group.frames()
                            if len(frames) != 4:
                                raise ValueError(f"Expected four frames for {clip_id}, {view}")
                            entries = []
                            for position, (frame_index, frame) in enumerate(zip(group.frame_indices, frames)):
                                image = _resized_image(
                                    frame, current=position == 3, resize=plan["resize"],
                                )
                                buffer = BytesIO()
                                image.save(buffer, format="PNG", compress_level=1)
                                payload = buffer.getvalue()
                                member = f"{clip_id}/{view}/{position}.png"
                                info = tarfile.TarInfo(member)
                                info.size = len(payload)
                                info.mode = 0o644
                                archive.addfile(info, BytesIO(payload))
                                entries.append({
                                    "member": member, "frame_index": frame_index,
                                    "width": image.width, "height": image.height,
                                })
                            record["frames"][view] = entries
                        finally:
                            group.release()
                    records.append(record)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    # Offsets let future readers seek to one PNG without scanning the shard.
    try:
        with tarfile.open(temporary_path, "r:") as archive:
            members = {member.name: member for member in archive}
        for record in records:
            for entries in record["frames"].values():
                for entry in entries:
                    member = members[entry["member"]]
                    entry["offset"] = member.offset_data
                    entry["bytes"] = member.size
        os.replace(temporary_path, tar_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    stat = tar_path.stat()
    index = {
        "plan": plan, "split": split, "clip_ids": clip_ids,
        "source_signatures": signatures, "records": records,
        "tar_bytes": stat.st_size, "tar_mtime_ns": stat.st_mtime_ns,
    }
    _atomic_json(index_path, index)
    return index, False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Completed PhysicalAI extraction")
    parser.add_argument("--dest", type=Path, help="Cache directory; defaults to ROOT/.physicalai_image_cache")
    parser.add_argument("--views", nargs="+", choices=tuple(PHYSICALAI_CAMERA_BY_VIEW), default=DEFAULT_VIEWS)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
    parser.add_argument("--anchor-time-s", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=96, help="Concurrent shard writers")
    parser.add_argument("--clips-per-shard", type=int, default=128)
    parser.add_argument("--max-clips", type=int, help="Only cache the first N clips per split for a smoke run")
    parser.add_argument("--history-image-pixels", type=int, default=174080)
    parser.add_argument("--current-image-pixels", type=int, default=921600)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--merge-size", type=int, default=2)
    args = parser.parse_args(argv)
    if (args.workers < 1 or args.clips_per_shard < 1
            or (args.max_clips is not None and args.max_clips < 1)
            or min(args.history_image_pixels, args.current_image_pixels,
                   args.patch_size, args.merge_size) < 1):
        parser.error("workers, shard size, max clips, and resize parameters must be positive")
    if len(set(args.views)) != len(args.views) or len(set(args.splits)) != len(args.splits):
        parser.error("views and splits must not contain duplicates")

    root = args.root.expanduser().resolve()
    dest = (args.dest or root / ".physicalai_image_cache").expanduser().resolve()
    state_dir = root / ".physicalai_extract"
    with (state_dir / "summary.json").open() as stream:
        extraction = json.load(stream)
    resize = {
        "history_image_pixels": args.history_image_pixels,
        "current_image_pixels": args.current_image_pixels,
        "patch_size": args.patch_size,
        "merge_size": args.merge_size,
    }
    plan = {
        "cache_version": CACHE_VERSION, "source_root": str(root),
        "extraction_plan": extraction["plan"],
        "valid_manifest_sha256": _manifest_hash(state_dir / "valid_clips.jsonl"),
        "anchor_us": round(args.anchor_time_s * 1_000_000),
        "views": list(args.views), "resize": resize,
        "interpolation": "torchvision_pil_bicubic", "format": "png_rgb_uint8_post_resize",
        "pillow_version": pillow_version, "torchvision_version": torchvision_version,
    }
    manifest_path = dest / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("plan") != plan:
        parser.error(f"cache settings differ from {manifest_path}; choose another --dest")

    jobs = []
    for split in args.splits:
        dataset = PhysicalAIDataset(
            root, split, args.views, anchor_time_s=args.anchor_time_s,
            max_samples=args.max_clips, storage="files",
        )
        for start in range(0, len(dataset.samples), args.clips_per_shard):
            jobs.append((dataset, split, start // args.clips_per_shard,
                         dataset.samples[start:start + args.clips_per_shard], dest, plan))
    dest.mkdir(parents=True, exist_ok=True)
    shards = []
    reused = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, (job, result) in enumerate(
            zip(jobs, pool.map(lambda item: _cache_shard(*item), jobs)), 1
        ):
            index, was_reused = result
            split, shard_number = job[1], job[2]
            stem = f"{split}-{shard_number:06d}"
            shards.append({
                "tar": f"shards/{stem}.tar", "index": f"shards/{stem}.json",
                "clips": len(index["clip_ids"]), "bytes": index["tar_bytes"],
            })
            reused += was_reused
            if number % 10 == 0 or number == len(jobs):
                LOG.info("Shards %d/%d; reused %d", number, len(jobs), reused)

    _atomic_json(manifest_path, {
        "plan": plan, "splits": list(args.splits),
        "complete": args.max_clips is None,
        "clips": sum(shard["clips"] for shard in shards), "shards": shards,
    })
    LOG.info("Finished %d clips in %d shards at %s", sum(row["clips"] for row in shards), len(shards), dest)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
