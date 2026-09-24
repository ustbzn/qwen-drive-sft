from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import io
from typing import Any, Callable, Dict, Iterator, List, Optional
from pathlib import Path
import os

from tqdm import tqdm
import pickle
import lzma
import logging

from navsim.common.scene_timestamps import timestamp_grid_error
from navsim.common.scene_cameras import camera_grid_error, camera_presence_mask

logger = logging.getLogger(__name__)

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig
from navsim.planning.metric_caching.metric_cache import MetricCache


class _NumpyCoreCompatUnpickler(pickle.Unpickler):
    """Resolve NumPy 2 module paths when reading data with NumPy 1.x."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def _pickle_load_compatible(stream) -> Any:
    """Read a pickle, retrying NumPy 2 module paths on NumPy 1.x."""
    position = stream.tell()
    try:
        return pickle.load(stream)
    except ModuleNotFoundError as error:
        missing_module = error.name or ""
        if missing_module != "numpy._core" and not missing_module.startswith("numpy._core."):
            raise
        stream.seek(position)
        return _NumpyCoreCompatUnpickler(stream).load()


def _pickle_loads_compatible(payload) -> Any:
    """Load a pickle payload without copying unless compatibility is needed."""
    try:
        return pickle.loads(payload)
    except ModuleNotFoundError as error:
        missing_module = error.name or ""
        if missing_module != "numpy._core" and not missing_module.startswith("numpy._core."):
            raise
        return _NumpyCoreCompatUnpickler(io.BytesIO(payload)).load()


def _load_log_frames(log_path: Path) -> List[Dict[str, Any]]:
    with log_path.open("rb") as stream:
        return _pickle_load_compatible(stream)


def _iter_loaded_logs(
    log_files: List[Path], num_workers: int,
) -> Iterator[List[Dict[str, Any]]]:
    """Prefetch at most num_workers logs, preserving the original log order.

    Bound the queue: eagerly submitting every pickle can retain an entire extra
    copy of the dataset while filtering is still consuming the early results.
    Threads overlap storage I/O without serializing large frame dicts through IPC.
    """
    worker_count = min(num_workers, len(log_files))
    if worker_count <= 1:
        for log_path in log_files:
            yield _load_log_frames(log_path)
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        remaining = iter(log_files)
        pending = deque(executor.submit(_load_log_frames, next(remaining)) for _ in range(worker_count))
        try:
            while pending:
                yield pending.popleft().result()
                log_path = next(remaining, None)
                if log_path is not None:
                    pending.append(executor.submit(_load_log_frames, log_path))
        finally:
            # A max_scenes limit can stop consumption before the queue is empty.
            for future in pending:
                future.cancel()


def filter_scenes(
    data_path: Path, scene_filter: SceneFilter, num_workers: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load filtered raw scenes; num_workers controls startup log I/O threads."""
    if num_workers < 1:
        raise ValueError("log loading num_workers must be positive")

    filtered_scenes: Dict[str, List[Dict[str, Any]]] = {}
    rejected_timestamps = 0
    rejected_cameras = 0
    log_files = list(data_path.iterdir())
    if scene_filter.log_names is not None:
        log_names = set(scene_filter.log_names)
        log_files = [log_file for log_file in log_files if log_file.name.replace(".pkl", "") in log_names]
    tokens = set(scene_filter.tokens) if scene_filter.tokens is not None else None
    description = f"Loading logs [workers={min(num_workers, len(log_files))}]"

    # Keep insertion order, duplicate-token handling and max_scenes identical to
    # serial loading, irrespective of which worker finishes first.
    with closing(_iter_loaded_logs(log_files, num_workers)) as loaded_logs:
        for scene_dict_list in tqdm(loaded_logs, total=len(log_files), desc=description):
            for offset in range(0, len(scene_dict_list), scene_filter.frame_interval):
                frame_list = scene_dict_list[
                    offset : offset + scene_filter.num_raw_frames : scene_filter.frame_stride
                ]
                if len(frame_list) < scene_filter.num_frames:
                    continue
                current_frame = frame_list[scene_filter.num_history_frames - 1]
                if scene_filter.has_route and len(current_frame["roadblock_ids"]) == 0:
                    continue
                token = current_frame["token"]
                if tokens is not None and token not in tokens:
                    continue
                if scene_filter.timestamp_interval_s is not None:
                    error = timestamp_grid_error(
                        [frame["timestamp"] for frame in frame_list],
                        scene_filter.num_history_frames - 1,
                        scene_filter.timestamp_interval_s, scene_filter.max_timestamp_error_s,
                    )
                    if error:
                        rejected_timestamps += 1
                        if rejected_timestamps <= 5:
                            logger.warning("Excluding scene log=%s token=%s: %s",
                                           current_frame["log_name"], token, error)
                        continue
                if scene_filter.required_camera_frames:
                    error = camera_grid_error(
                        [camera_presence_mask(frame.get("cams", {}))
                         for frame in frame_list[:scene_filter.num_history_frames]],
                        scene_filter.required_camera_frames,
                    )
                    if error:
                        rejected_cameras += 1
                        if rejected_cameras <= 5:
                            logger.warning("Excluding scene log=%s token=%s: %s",
                                           current_frame["log_name"], token, error)
                        continue
                filtered_scenes[token] = frame_list
                if scene_filter.max_scenes is not None and len(filtered_scenes) >= scene_filter.max_scenes:
                    return filtered_scenes
    if rejected_cameras:
        logger.warning("Excluded %d scenes with missing camera views from %s",
                       rejected_cameras, data_path)
    if rejected_timestamps:
        logger.warning("Excluded %d scenes with invalid timestamps from %s",
                       rejected_timestamps, data_path)
    return filtered_scenes


class SceneLoader:
    """Simple data loader of scenes from logs."""

    def __init__(
        self,
        data_path: Path,
        sensor_blobs_path: Path,
        scene_filter: SceneFilter,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
        load_image_path: bool = False,
        map_api_builder: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
        log_load_workers: int = 1,
    ):
        """
        Initializes the scene data loader.
        :param data_path: root directory of log folder
        :param sensor_blobs_path: root directory of sensor data
        :param scene_filter: dataclass for scene filtering specification
        :param log_load_workers: startup log I/O threads, separate from DataLoader workers
        :param sensor_config: dataclass for sensor loading specification, defaults to no sensors
        """

        self.scene_frames_dicts = filter_scenes(data_path, scene_filter, num_workers=log_load_workers)
        self._sensor_blobs_path = sensor_blobs_path
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config
        self.load_image_path = load_image_path
        self._map_api_builder = map_api_builder
        
        self.token_to_log_file: Dict[str, str] = {}
        self._build_token_to_log_file()  # 构建 token_to_log_file 字典

    def _build_token_to_log_file(self):
        """Builds the token_to_log_file dictionary."""
        for token, scene_dict_list in self.scene_frames_dicts.items():
            log_name = scene_dict_list[0]["log_name"]
            self.token_to_log_file[token] = log_name

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.scene_frames_dicts.keys())

    def __len__(self) -> int:
        """
        :return: number for scenes possible to load.
        """
        return len(self.tokens)

    def __getitem__(self, idx) -> str:
        """
        :param idx: index of scene
        :return: unique scene identifier
        """
        return self.tokens[idx]

    def get_scene_from_token(self, token: str) -> Scene:
        """
        Loads scene given a scene identifier string (token).
        :param token: scene identifier string.
        :return: scene dataclass
        """
        assert token in self.scene_frames_dicts
        scene_frames = self.scene_frames_dicts[token]
        map_api = self._map_api_builder(scene_frames) if self._map_api_builder else None
        return Scene.from_scene_dict_list(
            scene_frames,
            self._sensor_blobs_path,
            num_history_frames=self._scene_filter.num_history_frames,
            num_future_frames=self._scene_filter.num_future_frames,
            sensor_config=self._sensor_config,
            load_image_path=self.load_image_path,
            map_api=map_api,
        )

    def get_agent_input_from_token(self, token: str) -> AgentInput:
        """
        Loads agent input given a scene identifier string (token).
        :param token: scene identifier string.
        :return: agent input dataclass
        """
        assert token in self.scene_frames_dicts
        return AgentInput.from_scene_dict_list(
            self.scene_frames_dicts[token],
            self._sensor_blobs_path,
            num_history_frames=self._scene_filter.num_history_frames,
            sensor_config=self._sensor_config,
            load_image_path=self.load_image_path
        )

    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        """
        Collect tokens for each logs file given filtering.
        :return: dictionary of logs names and tokens
        """
        # generate a dict that contains a list of tokens for each log-name
        tokens_per_logs: Dict[str, List[str]] = {}
        for token, scene_dict_list in self.scene_frames_dicts.items():
            log_name = scene_dict_list[0]["log_name"]
            if tokens_per_logs.get(log_name):
                tokens_per_logs[log_name].append(token)
            else:
                tokens_per_logs.update({log_name: [token]})
        return tokens_per_logs


class MetricCacheLoader:
    """Simple dataloader for one or more metric cache roots."""

    def __init__(self, cache_path: Any, file_name: str = "metric_cache.pkl"):
        """
        Initializes the metric cache loader.
        :param cache_path: cache folder path, path-list, or os.pathsep-separated path string
        :param file_name: file name of cached files, defaults to "metric_cache.pkl"
        """

        self._file_name = file_name
        self.metric_cache_paths = self._load_metric_cache_paths(cache_path)

    @staticmethod
    def _normalize_cache_paths(cache_path: Any) -> List[Path]:
        if isinstance(cache_path, str):
            return [Path(path) for path in cache_path.split(os.pathsep) if path]
        if isinstance(cache_path, os.PathLike):
            return [Path(cache_path)]
        if hasattr(cache_path, "__iter__"):
            return [Path(path) for path in cache_path if str(path)]
        return [Path(cache_path)]

    def _load_metric_cache_paths(self, cache_path: Any) -> Dict[str, Path]:
        """
        Helper function to load all cache file paths from one or more folders.
        :param cache_path: directory/directories of cache folders
        :return: dictionary of token and file path
        """
        metric_cache_dict: Dict[str, Path] = {}
        duplicate_tokens: List[str] = []
        for cache_root in self._normalize_cache_paths(cache_path):
            for token, path in self._load_metric_cache_paths_from_root(cache_root).items():
                if token in metric_cache_dict:
                    duplicate_tokens.append(token)
                    continue
                metric_cache_dict[token] = path
        if duplicate_tokens:
            raise ValueError(
                "Duplicate token(s) across metric cache roots; token-only reward lookup is ambiguous: "
                f"{sorted(duplicate_tokens)[:10]}"
            )
        return metric_cache_dict

    def _load_metric_cache_paths_from_root(self, cache_path: Path) -> Dict[str, Path]:
        """Load all metric cache file paths from a single cache folder."""
        metadata_dir = cache_path / "metadata"
        metadata_file = [file for file in metadata_dir.iterdir() if ".csv" in str(file)][0]
        with open(str(metadata_file), "r") as f:
            cache_paths = f.read().splitlines()[1:]
        metric_cache_dict = {cache_path.split("/")[-2]: Path(cache_path) for cache_path in cache_paths}
        return metric_cache_dict

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.metric_cache_paths.keys())

    def __len__(self):
        """
        :return: number for scenes possible to load.
        """
        return len(self.metric_cache_paths)

    def __getitem__(self, idx: int) -> MetricCache:
        """
        :param idx: index of cache to cache to load
        :return: metric cache dataclass
        """
        return self.get_from_token(self.tokens[idx])

    def get_from_token(self, token: str) -> MetricCache:
        """
        Load metric cache from scene identifier
        :param token: unique identifier of scene
        :return: metric cache dataclass
        """
        with lzma.open(self.metric_cache_paths[token], "rb") as f:
            metric_cache: MetricCache = pickle.load(f)
        return metric_cache

    def to_pickle(self, path: Path) -> None:
        """
        Dumps complete metric cache into pickle.
        :param path: directory of cache folder
        """
        full_metric_cache = {}
        for token in tqdm(self.tokens):
            full_metric_cache[token] = self.get_from_token(token)
        with open(path, "wb") as f:
            pickle.dump(full_metric_cache, f)
