"""Persistent, bounded-memory access to raw NAVSIM logs.

Each immutable pack contains independent pickles of small frame blocks. Every
rank/worker mmaps the same pack, but unpickles only a sample's blocks. No model
features or VLM outputs are cached. The original raw frames are preserved.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import logging
import mmap
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Dict, Iterator, List, Optional

from tqdm import tqdm

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader, _load_log_frames
from navsim.common.scene_timestamps import timestamp_grid_error
from navsim.common.scene_cameras import camera_grid_error, camera_presence_mask

logger = logging.getLogger(__name__)
_VERSION = 1
_BLOCK_FRAMES = 32
_MAX_OPEN_PACKS = 4


def _signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _stat_key(path: Path) -> tuple:
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


def _filter_key(scene_filter: SceneFilter) -> dict:
    # Hydra can supply ListConfig; normalize everything into ordinary containers.
    value = asdict(scene_filter)
    if scene_filter.timestamp_interval_s is None:
        # Preserve existing index signatures when the optional check is disabled.
        value.pop("timestamp_interval_s")
        value.pop("max_timestamp_error_s")
    if not scene_filter.required_camera_frames:
        value.pop("required_camera_frames")
    else:
        value["required_camera_frames"] = {
            str(view): list(indices) for view, indices in scene_filter.required_camera_frames.items()
        }
    for key in ("log_names", "tokens"):
        if value[key] is not None:
            value[key] = sorted(str(item) for item in value[key])
    return value


def _atomic_pickle(path: Path, value: Any) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _build_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "build.lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@dataclass(frozen=True)
class PackedLog:
    path: str
    # End offset of each independently serialized block, starting with zero.
    offsets: tuple
    num_frames: int


def _pack_available(packed: PackedLog) -> bool:
    try:
        return Path(packed.path).stat().st_size == packed.offsets[-1]
    except FileNotFoundError:
        return False


def _read_pack_frame_metadata(packed: PackedLog) -> list:
    """Upgrade both time and camera metadata in one pass over existing frame packs."""
    metadata = []
    with open(packed.path, "rb") as stream:
        for _ in range(len(packed.offsets) - 1):
            block = pickle.load(stream)
            metadata.extend((int(frame["timestamp"]), camera_presence_mask(frame.get("cams", {})))
                            for frame in block)
            del block
    return metadata


def _log_scene_rejections(index: dict, index_path: Path) -> None:
    for key, reason in (("rejected_timestamps", "invalid timestamps"),
                        ("rejected_cameras", "missing camera views")):
        rejected = index.get(key, {})
        if rejected:
            logger.warning("Excluded %d scenes with %s; details in %s",
                           len(rejected), reason, index_path)
            for token, info in list(rejected.items())[:5]:
                logger.warning("Excluded scene log=%s token=%s: %s",
                               info["log_name"], token, info["error"])


def _prepare_log(request: tuple) -> tuple:
    source_key, directory, need_timestamps, need_cameras = request
    source = Path(source_key[0])
    identity = _signature([_VERSION, _BLOCK_FRAMES, source_key])
    pack_path = directory / (identity + ".frames")
    meta_path = directory / (identity + ".meta.pkl")
    if meta_path.exists() and pack_path.exists():
        with meta_path.open("rb") as stream:
            metadata = pickle.load(stream)
        if _pack_available(metadata[0]):
            packed, rows = metadata
            required_columns = 5 if need_cameras else (4 if need_timestamps else 3)
            if rows and len(rows[0]) < required_columns:
                frame_metadata = _read_pack_frame_metadata(packed)
                if len(frame_metadata) != len(rows):
                    raise ValueError(f"Raw frame metadata length mismatch: {meta_path}")
                metadata = (packed, [(*row[:3], *extra) for row, extra in zip(rows, frame_metadata)])
                _atomic_pickle(meta_path, metadata)
            return metadata
        logger.warning("Rebuilding incomplete raw frame pack: %s", pack_path)

    frames = _load_log_frames(source)
    offsets = [0]
    fd, temporary = tempfile.mkstemp(prefix=identity + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            for start in range(0, len(frames), _BLOCK_FRAMES):
                pickle.dump(frames[start:start + _BLOCK_FRAMES], stream, protocol=pickle.HIGHEST_PROTOCOL)
                offsets.append(stream.tell())
            stream.flush()
            os.fsync(stream.fileno())
        if _stat_key(source) != source_key:
            raise RuntimeError(f"Raw log changed while packing: {source}")
        metadata = (
            PackedLog(str(pack_path), tuple(offsets), len(frames)),
            [(frame["token"], len(frame["roadblock_ids"]) > 0, frame["log_name"],
              int(frame["timestamp"]), camera_presence_mask(frame.get("cams", {}))) for frame in frames],
        )
        os.replace(temporary, pack_path)
        # Metadata is the commit marker; incomplete logs are rebuilt on restart.
        _atomic_pickle(meta_path, metadata)
        return metadata
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _iter_prepared_logs(requests: list, workers: int) -> Iterator[tuple]:
    """At most workers raw logs are decoded at once, only metadata is retained."""
    if workers == 1:
        for request in requests:
            yield _prepare_log(request)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        remaining = iter(requests)
        pending = deque()
        for _ in range(min(workers, len(requests))):
            pending.append(executor.submit(_prepare_log, next(remaining)))
        try:
            while pending:
                yield pending.popleft().result()
                request = next(remaining, None)
                if request is not None:
                    pending.append(executor.submit(_prepare_log, request))
        finally:
            for future in pending:
                future.cancel()


def prepare_scene_index(
    data_path: Path, scene_filter: SceneFilter, cache_dir: Path, workers: int = 2,
) -> Path:
    """Prepare or reuse an immutable source/split index; intended for rank zero.

    A complete index is keyed by source stat fingerprints and every filter field.
    Per-log packs survive a failed build and are reused across different splits.
    """
    if workers < 1:
        raise ValueError("scene_cache_workers must be positive")
    data_path = data_path.resolve()
    directory = cache_dir.expanduser().resolve() / f"raw-v{_VERSION}" / _signature(str(data_path))[:24]
    log_names = set(scene_filter.log_names) if scene_filter.log_names is not None else None
    # Preserve legacy iterdir order, including max_scenes/duplicate-token semantics.
    paths = [path for path in data_path.iterdir()
             if (log_names is None or path.name.replace(".pkl", "") in log_names)]
    source_keys = [_stat_key(path) for path in paths]
    spec = {"version": _VERSION, "source": str(data_path), "filter": _filter_key(scene_filter),
            "logs": source_keys}
    index_path = directory / (_signature(spec) + ".scenes.pkl")
    with _build_lock(directory):
        if index_path.exists():
            with index_path.open("rb") as stream:
                existing = pickle.load(stream)
            if all(_pack_available(packed) for packed in existing["packs"]):
                logger.info("Reusing raw scene index: %s", index_path)
                _log_scene_rejections(existing, index_path)
                return index_path
            logger.warning("Repairing missing/incomplete raw scene cache: %s", index_path)
            del existing
        logger.info("Preparing raw scene index: %d logs, workers=%d, directory=%s",
                    len(paths), workers, directory)
        records, packs = {}, []
        rejected_timestamps = {}
        rejected_cameras = {}
        tokens = set(scene_filter.tokens) if scene_filter.tokens is not None else None
        requests = [(key, directory, scene_filter.timestamp_interval_s is not None,
                     bool(scene_filter.required_camera_frames)) for key in source_keys]
        iterator = _iter_prepared_logs(requests, workers)
        try:
            for packed, metadata in tqdm(iterator, total=len(requests), desc="Packing raw logs"):
                pack_id = len(packs)
                packs.append(packed)
                for start in range(0, len(metadata) - scene_filter.num_raw_frames + 1,
                                   scene_filter.frame_interval):
                    current = start + (scene_filter.num_history_frames - 1) * scene_filter.frame_stride
                    token, has_route, _ = metadata[current][:3]
                    if scene_filter.has_route and not has_route:
                        continue
                    if tokens is not None and token not in tokens:
                        continue
                    if scene_filter.timestamp_interval_s is not None:
                        window = metadata[start:start + scene_filter.num_raw_frames:scene_filter.frame_stride]
                        error = timestamp_grid_error(
                            [row[3] for row in window], scene_filter.num_history_frames - 1,
                            scene_filter.timestamp_interval_s, scene_filter.max_timestamp_error_s,
                        )
                        if error:
                            rejected_timestamps[token] = {"log_name": metadata[current][2], "error": error}
                            continue
                    if scene_filter.required_camera_frames:
                        history_masks = [metadata[start + i * scene_filter.frame_stride][4]
                                         for i in range(scene_filter.num_history_frames)]
                        error = camera_grid_error(history_masks, scene_filter.required_camera_frames)
                        if error:
                            rejected_cameras[token] = {"log_name": metadata[current][2], "error": error}
                            continue
                    records[token] = (pack_id, start, metadata[start][2])
                    if scene_filter.max_scenes is not None and len(records) >= scene_filter.max_scenes:
                        break
                if scene_filter.max_scenes is not None and len(records) >= scene_filter.max_scenes:
                    break
        finally:
            iterator.close()
        # Reject a mixed snapshot if any source changed during the build.
        if [_stat_key(path) for path in paths] != source_keys:
            raise RuntimeError(f"Raw logs changed while building index: {data_path}")
        index = {"spec": spec, "packs": packs, "records": records,
                 "rejected_timestamps": rejected_timestamps, "rejected_cameras": rejected_cameras}
        _atomic_pickle(index_path, index)
        _log_scene_rejections(index, index_path)
    return index_path


class _IndexedFrames(Mapping):
    """Read-only mapping, with a bounded LRU of mappings and no decoded log cache."""
    def __init__(self, records: dict, packs: list, scene_filter: SceneFilter):
        self.records = records
        self.packs = packs
        self.scene_filter = scene_filter
        self._pid = os.getpid()
        self._maps = OrderedDict()

    def __getstate__(self):
        # DataLoader spawn must not pickle mmap objects or the underlying data.
        state = self.__dict__.copy()
        state["_maps"] = OrderedDict()
        return state

    def close(self):
        for mapping in self._maps.values():
            mapping.close()
        self._maps.clear()

    def __del__(self):
        self.close()

    def _mapping(self, pack_id: int):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if pack_id not in self._maps:
            if len(self._maps) >= _MAX_OPEN_PACKS:
                _, oldest = self._maps.popitem(last=False)
                oldest.close()
            with open(self.packs[pack_id].path, "rb") as stream:
                self._maps[pack_id] = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        self._maps.move_to_end(pack_id)
        return self._maps[pack_id]

    def __getitem__(self, token):
        pack_id, start, _ = self.records[token]
        pack = self.packs[pack_id]
        end = start + self.scene_filter.num_raw_frames
        first, last = start // _BLOCK_FRAMES, (end - 1) // _BLOCK_FRAMES
        mapping = self._mapping(pack_id)
        frames = []
        for block in range(first, last + 1):
            payload = memoryview(mapping)[pack.offsets[block]:pack.offsets[block + 1]]
            try:
                frames.extend(pickle.loads(payload))
            finally:
                payload.release()
        offset = start - first * _BLOCK_FRAMES
        return frames[offset:offset + self.scene_filter.num_raw_frames:self.scene_filter.frame_stride]

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    def __contains__(self, token):
        return token in self.records


class IndexedSceneLoader(SceneLoader):
    """SceneLoader interface backed by shared packs, never an eager raw dataset."""
    def __init__(
        self, data_path: Path, sensor_blobs_path: Path, scene_filter: SceneFilter,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
        load_image_path: bool = False, map_api_builder=None,
        *, index_path: Path,
    ):
        self.index_path = Path(index_path)
        with self.index_path.open("rb") as stream:
            index = pickle.load(stream)
        spec = index["spec"]
        if (spec["version"] != _VERSION or spec["source"] != str(data_path.resolve())
                or spec["filter"] != _filter_key(scene_filter)):
            raise ValueError(f"Raw scene index does not match source/filter: {index_path}")
        for packed in index["packs"]:
            if Path(packed.path).stat().st_size != packed.offsets[-1]:
                raise RuntimeError(f"Missing or incomplete raw frame pack: {packed.path}")
        self.scene_frames_dicts = _IndexedFrames(index["records"], index["packs"], scene_filter)
        self._tokens = list(index["records"])
        self.token_to_log_file = {token: record[2] for token, record in index["records"].items()}
        self._sensor_blobs_path = sensor_blobs_path
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config
        self.load_image_path = load_image_path
        self._map_api_builder = map_api_builder

    @property
    def tokens(self) -> List[str]:
        return self._tokens
