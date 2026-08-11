#!/usr/bin/env python3
"""Restore missing NavSim camera calibration fields from a protobuf text file.

The MOGO NavSim logs contain camera images but may have ``None`` for the
calibration fields under ``cams``.  This utility parses the camera entries in
``calibrated_sensor.pb.txt`` and fills those fields in matching PKL frames.

By default the command is a dry-run.  Use ``--output-root`` to write updated
copies while preserving the ``navsim_logs`` directory layout, or use
``--in-place --yes-i-understand`` only after inspecting a dry-run.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import re
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image


# These are the six camera streams represented by the NavSim CAM_* fields.
CAMERA_TOPICS = {
    "/sensor/camera/sensing/image_raw_120": "CAM_F0",
    "/sensor/camera/sensing/image_raw_left": "CAM_L0",
    "/sensor/camera/sensing/image_raw_right": "CAM_R0",
    "/sensor/camera/sensing/image_raw_back": "CAM_B0",
    "/sensor/camera/sensing/image_raw_left_back": "CAM_L1",
    "/sensor/camera/sensing/image_raw_right_back": "CAM_R1",
}
CAMERA_DIRECTORIES = {
    "CAM_F0": "front",
    "CAM_L0": "left",
    "CAM_R0": "right",
    "CAM_B0": "back",
    "CAM_L1": "left_back",
    "CAM_R1": "right_back",
}
CALIBRATION_FIELDS = (
    "sensor2lidar_rotation",
    "sensor2lidar_translation",
    "cam_intrinsic",
    "distortion",
)
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _remove_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        kept: list[str] = []
        for char in line:
            if char == "#" and not in_string:
                break
            kept.append(char)
            if char == '"' and not escaped:
                in_string = not in_string
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append("".join(kept))
    return "\n".join(lines)


def _named_blocks(text: str, name: str) -> list[str]:
    """Return the contents of all protobuf-style ``name { ... }`` blocks."""
    blocks: list[str] = []
    pattern = re.compile(rf"\b{re.escape(name)}\s*\{{")
    position = 0
    while match := pattern.search(text, position):
        start = match.end()
        depth = 1
        in_string = False
        escaped = False
        index = start
        while index < len(text) and depth:
            char = text[index]
            if char == '"' and not escaped:
                in_string = not in_string
            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
            index += 1
        if depth:
            raise ValueError(f"Unclosed {name} block")
        blocks.append(text[start : index - 1])
        position = index
    return blocks


def _one_block(text: str, name: str) -> str:
    blocks = _named_blocks(text, name)
    if len(blocks) != 1:
        raise ValueError(f"Expected one {name} block, found {len(blocks)}")
    return blocks[0]


def _string_field(text: str, name: str) -> str | None:
    match = re.search(rf'\b{re.escape(name)}\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
    return match.group(1) if match else None


def _number_fields(text: str, name: str) -> list[float]:
    return [
        float(value)
        for value in re.findall(rf"\b{re.escape(name)}\s*:\s*({NUMBER_RE})", text)
    ]


def _xyz(block: str) -> np.ndarray:
    values: list[float] = []
    for axis in "xyz":
        found = _number_fields(block, axis)
        if len(found) != 1:
            raise ValueError(f"Expected one {axis} value, found {len(found)}")
        values.append(found[0])
    return np.asarray(values, dtype=np.float64)


def _quaternion_rotation(block: str) -> np.ndarray:
    values: dict[str, float] = {}
    for component in "xyzw":
        found = _number_fields(block, component)
        if len(found) != 1:
            raise ValueError(f"Expected one quaternion {component} value, found {len(found)}")
        values[component] = found[0]

    x, y, z, w = (values[key] for key in "xyzw")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("Quaternion has invalid norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("Quaternion produced an invalid rotation matrix")
    return rotation


def parse_calibration(path: Path) -> dict[str, dict[str, np.ndarray | tuple[int, int]]]:
    text = _remove_comments(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, np.ndarray | tuple[int, int]]] = {}
    seen_topics: set[str] = set()

    for sensor in _named_blocks(text, "sensor_info"):
        topic = _string_field(sensor, "topic")
        if topic not in CAMERA_TOPICS:
            continue
        if topic in seen_topics:
            raise ValueError(f"Duplicate camera topic {topic} in {path}")
        seen_topics.add(topic)

        extrinsic = _one_block(sensor, "extrinsic")
        translation = _xyz(_one_block(extrinsic, "translation"))
        rotation = _quaternion_rotation(_one_block(extrinsic, "rotation"))
        intrinsic_block = _one_block(sensor, "intrinsic")
        intrinsic_values = _number_fields(intrinsic_block, "matrix")
        if len(intrinsic_values) != 9:
            raise ValueError(f"Expected 9 intrinsic values for {topic}, found {len(intrinsic_values)}")
        distortion_values = _number_fields(_one_block(sensor, "distcoeff"), "distort_matrix")
        if len(distortion_values) != 5:
            raise ValueError(f"Expected 5 distortion values for {topic}, found {len(distortion_values)}")
        width = _number_fields(intrinsic_block, "width")
        height = _number_fields(intrinsic_block, "height")
        if len(width) != 1 or len(height) != 1 or width[0] <= 0 or height[0] <= 0:
            raise ValueError(f"Missing or invalid intrinsic resolution for {topic}")

        camera_name = CAMERA_TOPICS[topic]
        result[camera_name] = {
            "sensor2lidar_rotation": rotation,
            "sensor2lidar_translation": translation,
            "cam_intrinsic": np.asarray(intrinsic_values, dtype=np.float64).reshape(3, 3),
            "distortion": np.asarray(distortion_values, dtype=np.float64),
            "calibration_size": (int(width[0]), int(height[0])),
        }

    expected = set(CAMERA_TOPICS.values())
    if set(result) != expected:
        missing = ", ".join(sorted(expected - set(result)))
        raise ValueError(f"Missing target cameras in {path}: {missing}")
    return result


def _dataset_split(pkl_path: Path) -> str | None:
    parts = pkl_path.parts
    try:
        return parts[parts.index("navsim_logs") + 1]
    except (ValueError, IndexError):
        return None


def _sensor_image_path(
    data_path: object,
    pkl_path: Path,
    dataset_root: Path,
    sensor_blobs_root: Path | None,
) -> Path | None:
    if not data_path:
        return None
    relative = Path(str(data_path))
    if relative.is_absolute():
        return relative

    root = sensor_blobs_root or (dataset_root / "sensor_blobs")
    split_names: list[str] = []
    for candidate_path in (pkl_path, pkl_path.resolve()):
        candidate_split = _dataset_split(candidate_path)
        if candidate_split and candidate_split not in split_names:
            split_names.append(candidate_split)
    candidates = [root / split / relative for split in split_names]
    candidates.append(root / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Image for {pkl_path.name} was not found under {root}: {relative}"
    )


def _scaled_intrinsic(
    intrinsic: np.ndarray,
    calibration_size: tuple[int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    calibration_width, calibration_height = calibration_size
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size {image_size}")
    scaled = np.asarray(intrinsic, dtype=np.float64).copy()
    scaled[0, :] *= image_width / calibration_width
    scaled[1, :] *= image_height / calibration_height
    return scaled


def _as_float32(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _missing(value: object) -> bool:
    return value is None


def update_frames(
    data: object,
    calibration: dict[str, dict[str, np.ndarray | tuple[int, int]]],
    *,
    pkl_path: Path,
    dataset_root: Path,
    sensor_blobs_root: Path | None,
    overwrite_existing: bool,
) -> dict[str, int]:
    if not isinstance(data, list):
        raise TypeError(f"PKL root must be a list, got {type(data).__name__}")

    image_size_cache: dict[tuple[str, str], tuple[int, int]] = {}
    stats = {"frames": len(data), "camera_frames": 0, "fields": 0, "skipped_no_image": 0}
    for frame_index, frame in enumerate(data):
        if not isinstance(frame, dict) or not isinstance(frame.get("cams"), dict):
            raise TypeError(f"Frame {frame_index} has no cams dictionary")
        cams = frame["cams"]

        for camera_name, values in calibration.items():
            camera = cams.get(camera_name)
            if not isinstance(camera, dict):
                raise TypeError(f"Frame {frame_index} camera {camera_name} is not a dictionary")
            data_path = camera.get("data_path")
            expected_directory = CAMERA_DIRECTORIES[camera_name]
            actual_directory = Path(str(data_path)).parent.name if data_path else expected_directory
            if data_path and actual_directory != expected_directory:
                raise ValueError(
                    f"Frame {frame_index} {camera_name} path is in {actual_directory!r}, "
                    f"expected {expected_directory!r}"
                )
            if not data_path:
                stats["skipped_no_image"] += 1
                continue

            image_path = _sensor_image_path(data_path, pkl_path, dataset_root, sensor_blobs_root)
            assert image_path is not None
            cache_key = (camera_name, str(image_path.parent))
            image_size = image_size_cache.get(cache_key)
            if image_size is None:
                with Image.open(image_path) as image:
                    image_size = (int(image.width), int(image.height))
                image_size_cache[cache_key] = image_size

            calibration_size = values["calibration_size"]
            assert isinstance(calibration_size, tuple)
            scaled_intrinsic = _scaled_intrinsic(
                values["cam_intrinsic"], calibration_size, image_size
            )
            updated_values: dict[str, np.ndarray] = {
                "sensor2lidar_rotation": _as_float32(values["sensor2lidar_rotation"]),
                "sensor2lidar_translation": _as_float32(values["sensor2lidar_translation"]),
                "cam_intrinsic": _as_float32(scaled_intrinsic),
                "distortion": _as_float32(values["distortion"]),
            }
            changed = False
            for field in CALIBRATION_FIELDS:
                if overwrite_existing or _missing(camera.get(field)):
                    camera[field] = updated_values[field]
                    stats["fields"] += 1
                    changed = True
            if changed:
                stats["camera_frames"] += 1
    return stats


def _atomic_pickle_dump(data: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as output:
            pickle.dump(data, output, protocol=pickle.HIGHEST_PROTOCOL)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _discover_pkls(dataset_root: Path, split: str) -> list[Path]:
    logs_root = dataset_root / "navsim_logs"
    if split == "all":
        return sorted(logs_root.glob("*/*.pkl"))
    return sorted((logs_root / split).glob("*.pkl"))


def _destination(
    pkl_path: Path,
    *,
    dataset_root: Path,
    output_root: Path | None,
    in_place: bool,
) -> Path | None:
    if in_place:
        return pkl_path.resolve() if pkl_path.is_symlink() else pkl_path
    if output_root is None:
        return None
    try:
        relative = pkl_path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"PKL {pkl_path} is outside dataset root {dataset_root}") from exc
    return output_root / relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--split", default="all", help="PKL split to process, or all")
    parser.add_argument("--pkl", action="append", type=Path, default=[], help="Process one or more explicit PKLs")
    parser.add_argument("--sensor-blobs-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--in-place", action="store_true", help="Atomically overwrite input PKLs")
    parser.add_argument(
        "--yes-i-understand",
        action="store_true",
        help="Required with --in-place because input PKLs will be overwritten",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    calibration_path = args.calibration.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    if not calibration_path.is_file():
        raise SystemExit(f"Calibration file does not exist: {calibration_path}")
    if args.in_place and not args.yes_i_understand:
        raise SystemExit("--in-place requires --yes-i-understand")
    if args.in_place and args.output_root is not None:
        raise SystemExit("Use either --in-place or --output-root, not both")
    output_root = args.output_root.resolve() if args.output_root else None
    if output_root is not None and output_root == dataset_root:
        raise SystemExit("--output-root must differ from --dataset-root")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    calibration = parse_calibration(calibration_path)
    pkl_paths = [path.resolve() for path in args.pkl] if args.pkl else _discover_pkls(dataset_root, args.split)
    if args.limit is not None:
        pkl_paths = pkl_paths[: args.limit]
    if not pkl_paths:
        raise SystemExit("No PKLs found")

    sensor_blobs_root = args.sensor_blobs_root.resolve() if args.sensor_blobs_root else None
    total = {"files": 0, "frames": 0, "camera_frames": 0, "fields": 0, "skipped_no_image": 0}
    for pkl_path in pkl_paths:
        if not pkl_path.is_file():
            raise SystemExit(f"PKL does not exist: {pkl_path}")
        with pkl_path.open("rb") as source:
            data = pickle.load(source)
        stats = update_frames(
            data,
            calibration,
            pkl_path=pkl_path,
            dataset_root=dataset_root,
            sensor_blobs_root=sensor_blobs_root,
            overwrite_existing=args.overwrite_existing,
        )
        destination = _destination(
            pkl_path,
            dataset_root=dataset_root,
            output_root=output_root,
            in_place=args.in_place,
        )
        if destination is not None:
            if destination.exists() and output_root is not None and not args.overwrite_output:
                raise SystemExit(f"Output already exists; pass --overwrite-output: {destination}")
            _atomic_pickle_dump(data, destination)
            action = f"WRITE {destination}"
        else:
            action = "DRY-RUN"
        print(
            f"{action}: {pkl_path} | frames={stats['frames']} "
            f"camera_frames={stats['camera_frames']} fields={stats['fields']} "
            f"skipped_no_image={stats['skipped_no_image']}"
        )
        total["files"] += 1
        for key in total:
            if key != "files":
                total[key] += stats[key]

    print(
        f"Done: files={total['files']} frames={total['frames']} "
        f"camera_frames={total['camera_frames']} fields={total['fields']} "
        f"skipped_no_image={total['skipped_no_image']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
