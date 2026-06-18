from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, BinaryIO, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import io
import os

import numpy as np
import numpy.typing as npt
from PIL import Image
from pyquaternion import Quaternion


from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

NAVSIM_INTERVAL_LENGTH: float = 0.5
OPENSCENE_DATA_ROOT = os.environ.get("OPENSCENE_DATA_ROOT")
NUPLAN_MAPS_ROOT = os.environ.get("NUPLAN_MAPS_ROOT")


@dataclass
class Camera:
    """Camera dataclass for image and parameters."""

    image: Optional[npt.NDArray[np.float32]] = None

    sensor2lidar_rotation: Optional[npt.NDArray[np.float32]] = None
    sensor2lidar_translation: Optional[npt.NDArray[np.float32]] = None
    intrinsics: Optional[npt.NDArray[np.float32]] = None
    distortion: Optional[npt.NDArray[np.float32]] = None


@dataclass
class Cameras:
    """Multi-camera dataclass."""

    cam_f0: Camera
    cam_l0: Camera
    cam_l1: Camera
    cam_l2: Camera
    cam_r0: Camera
    cam_r1: Camera
    cam_r2: Camera
    cam_b0: Camera

    @classmethod
    def from_camera_dict(
        cls,
        sensor_blobs_path: Path,
        camera_dict: Dict[str, Any],
        sensor_names: List[str],
        load_image_path: bool = False 
    ) -> Cameras:
        """
        Load camera dataclass from dictionary.
        :param sensor_blobs_path: root directory of sensor data.
        :param camera_dict: dictionary containing camera specifications.
        :param sensor_names: list of camera identifiers to include.
        :return: Cameras dataclass.
        """

        data_dict: Dict[str, Camera] = {}
        for camera_name in camera_dict.keys():
            camera_identifier = camera_name.lower()
            if camera_identifier in sensor_names:
                camera_data = camera_dict[camera_name]
                data_path = camera_data.get("data_path")
                if not data_path:
                    data_dict[camera_identifier] = Camera()
                    continue

                image_path = sensor_blobs_path / data_path
                if not image_path.is_file():
                    raise FileNotFoundError(f"Camera image not found for {camera_name}: {image_path}")

                if not load_image_path:
                    data_dict[camera_identifier] = Camera(
                        image=np.array(Image.open(image_path)),
                        sensor2lidar_rotation=camera_data["sensor2lidar_rotation"],
                        sensor2lidar_translation=camera_data["sensor2lidar_translation"],
                        intrinsics=camera_data["cam_intrinsic"],
                        distortion=camera_data["distortion"],
                    )
                else:
                    data_dict[camera_identifier] = Camera(
                        image=image_path,
                        sensor2lidar_rotation=camera_data["sensor2lidar_rotation"],
                        sensor2lidar_translation=camera_data["sensor2lidar_translation"],
                        intrinsics=camera_data["cam_intrinsic"],
                        distortion=camera_data["distortion"],
                    )
            else:
                data_dict[camera_identifier] = Camera()  # empty camera

        return Cameras(
            cam_f0=data_dict["cam_f0"],
            cam_l0=data_dict["cam_l0"],
            cam_l1=data_dict["cam_l1"],
            cam_l2=data_dict["cam_l2"],
            cam_r0=data_dict["cam_r0"],
            cam_r1=data_dict["cam_r1"],
            cam_r2=data_dict["cam_r2"],
            cam_b0=data_dict["cam_b0"],
        )


@dataclass
class Lidar:
    """Lidar point cloud dataclass."""

    # NOTE:
    # merged lidar point cloud as (6,n) float32 array with n points
    # first axis: (x, y, z, intensity, ring, lidar_id), see LidarIndex
    lidar_pc: Optional[npt.NDArray[np.float32]] = None

    @staticmethod
    def _load_bytes(lidar_path: Path) -> BinaryIO:
        """Helper static method to load lidar point cloud stream."""
        with open(lidar_path, "rb") as fp:
            return io.BytesIO(fp.read())

    @classmethod
    def from_paths(cls, sensor_blobs_path: Path, lidar_path: Path, sensor_names: List[str]) -> Lidar:
        """
        Loads lidar point cloud dataclass in log loading.
        :param sensor_blobs_path: root directory to sensor data
        :param lidar_path: relative lidar path from logs.
        :param sensor_names: list of sensor identifiers to load`
        :return: lidar point cloud dataclass
        """

        # NOTE: this could be extended to load specific LiDARs in the merged pc
        if "lidar_pc" in sensor_names:
            if not str(lidar_path):
                return Lidar()
            global_lidar_path = sensor_blobs_path / lidar_path
            if global_lidar_path.is_dir():
                return Lidar()
            lidar_pc = LidarPointCloud.from_buffer(cls._load_bytes(global_lidar_path), "pcd").points
            return Lidar(lidar_pc)
        return Lidar()  # empty lidar


@dataclass
class EgoStatus:
    """Ego vehicle status dataclass."""

    ego_pose: npt.NDArray[np.float64]
    ego_velocity: npt.NDArray[np.float32]
    ego_acceleration: npt.NDArray[np.float32]
    driving_command: npt.NDArray[np.int]
    ego_yaw_rate: float = 0.0
    ego_curvature: float = 0.0
    ego_curvature_rate: float = 0.0
    ego_steering_wheel_angle: float = 0.0
    ego_steering_wheel_rate: float = 0.0
    in_global_frame: bool = False  # False for AgentInput


def _extract_can_bus_value(scene_frame: Dict, value_index: int) -> float:
    """Extract a scalar can_bus value with zero fallback for missing fields."""
    can_bus = scene_frame.get("can_bus", None)
    if can_bus is None or len(can_bus) <= value_index:
        return 0.0
    return float(can_bus[value_index])


def _extract_ego_yaw_rate(scene_frame: Dict) -> float:
    """Extract ego yaw rate from can_bus angular velocity z."""
    return _extract_can_bus_value(scene_frame, 15)


def _extract_ego_steering_wheel_angle(scene_frame: Dict) -> float:
    """Extract steering wheel angle when available; NAVSIM/nuPlan falls back to zero."""
    return _extract_can_bus_value(scene_frame, 16)


def _extract_ego_steering_wheel_rate(scene_frame: Dict) -> float:
    """Extract steering wheel rate when available; NAVSIM/nuPlan falls back to zero."""
    return _extract_can_bus_value(scene_frame, 17)


def _compute_ego_curvature(
    ego_velocity: npt.NDArray[np.float32],
    ego_yaw_rate: float,
    min_speed: float = 0.5,
) -> float:
    """Compute ego curvature from yaw rate and speed with low-speed protection."""
    speed = float(np.hypot(ego_velocity[0], ego_velocity[1]))
    if speed < min_speed:
        return 0.0
    return float(ego_yaw_rate / speed)


def _extract_frame_timestamp_s(scene_frame: Dict) -> Optional[float]:
    """Extract frame timestamp in seconds. NAVSIM stores timestamps in microseconds."""
    timestamp = scene_frame.get("timestamp", None)
    if timestamp is None:
        return None
    return float(timestamp) * 1e-6


def _compute_ego_curvature_rates(
    ego_curvatures: List[float],
    scene_frames: List[Dict],
    default_interval_s: float = NAVSIM_INTERVAL_LENGTH,
) -> List[float]:
    """Compute curvature rate from past-to-current curvature differences."""
    ego_curvature_rates: List[float] = []
    timestamps_s = [_extract_frame_timestamp_s(scene_frame) for scene_frame in scene_frames]

    for frame_idx, ego_curvature in enumerate(ego_curvatures):
        if frame_idx == 0:
            ego_curvature_rates.append(0.0)
            continue

        dt_s = default_interval_s
        current_timestamp_s = timestamps_s[frame_idx]
        previous_timestamp_s = timestamps_s[frame_idx - 1]
        if current_timestamp_s is not None and previous_timestamp_s is not None:
            timestamp_dt_s = current_timestamp_s - previous_timestamp_s
            if timestamp_dt_s > 1e-6:
                dt_s = timestamp_dt_s

        if dt_s <= 1e-6:
            ego_curvature_rates.append(0.0)
            continue

        ego_curvature_rates.append(float((ego_curvature - ego_curvatures[frame_idx - 1]) / dt_s))

    return ego_curvature_rates


@dataclass
class AgentInput:
    """Dataclass for agent inputs with current and past ego statuses and sensors."""

    ego_statuses: List[EgoStatus]
    cameras: List[Cameras]
    lidars: List[Lidar]

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        sensor_config: SensorConfig,
        load_image_path: bool = False 
    ) -> AgentInput:
        """
        Load agent input from scene dictionary.
        :param scene_dict_list: list of scene frames (in logs).
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of agent input frames
        :param sensor_config: sensor config dataclass
        :return: agent input dataclass
        """
        assert len(scene_dict_list) > 0, "Scene list is empty!"

        global_ego_poses = []
        for frame_idx in range(num_history_frames):
            ego_translation = scene_dict_list[frame_idx]["ego2global_translation"]
            ego_quaternion = Quaternion(*scene_dict_list[frame_idx]["ego2global_rotation"])
            global_ego_pose = np.array(
                [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
                dtype=np.float64,
            )
            global_ego_poses.append(global_ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[-1]), np.array(global_ego_poses, dtype=np.float64)
        )

        ego_statuses: List[EgoStatus] = []
        cameras: List[EgoStatus] = []
        lidars: List[Lidar] = []

        ego_velocities: List[npt.NDArray[np.float32]] = []
        ego_accelerations: List[npt.NDArray[np.float32]] = []
        ego_yaw_rates: List[float] = []
        ego_curvatures: List[float] = []
        ego_steering_wheel_angles: List[float] = []
        ego_steering_wheel_rates: List[float] = []
        for frame_idx in range(num_history_frames):
            ego_dynamic_state = scene_dict_list[frame_idx]["ego_dynamic_state"]
            ego_velocity = np.array(ego_dynamic_state[:2], dtype=np.float32)
            ego_yaw_rate = _extract_ego_yaw_rate(scene_dict_list[frame_idx])

            ego_velocities.append(ego_velocity)
            ego_accelerations.append(np.array(ego_dynamic_state[2:], dtype=np.float32))
            ego_yaw_rates.append(ego_yaw_rate)
            ego_curvatures.append(_compute_ego_curvature(ego_velocity, ego_yaw_rate))
            ego_steering_wheel_angles.append(_extract_ego_steering_wheel_angle(scene_dict_list[frame_idx]))
            ego_steering_wheel_rates.append(_extract_ego_steering_wheel_rate(scene_dict_list[frame_idx]))

        ego_curvature_rates = _compute_ego_curvature_rates(
            ego_curvatures, scene_dict_list[:num_history_frames]
        )

        for frame_idx in range(num_history_frames):
            ego_status = EgoStatus(
                ego_pose=np.array(local_ego_poses[frame_idx], dtype=np.float32),
                ego_velocity=ego_velocities[frame_idx],
                ego_acceleration=ego_accelerations[frame_idx],
                driving_command=scene_dict_list[frame_idx]["driving_command"],
                ego_yaw_rate=ego_yaw_rates[frame_idx],
                ego_curvature=ego_curvatures[frame_idx],
                ego_curvature_rate=ego_curvature_rates[frame_idx],
                ego_steering_wheel_angle=ego_steering_wheel_angles[frame_idx],
                ego_steering_wheel_rate=ego_steering_wheel_rates[frame_idx],
            )
            ego_statuses.append(ego_status)

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            cameras.append(
                Cameras.from_camera_dict(
                    sensor_blobs_path=sensor_blobs_path,
                    camera_dict=scene_dict_list[frame_idx]["cams"],
                    sensor_names=sensor_names,
                    load_image_path=load_image_path,
                )
            )

            lidars.append(
                Lidar.from_paths(
                    sensor_blobs_path=sensor_blobs_path,
                    lidar_path=Path(scene_dict_list[frame_idx]["lidar_path"]),
                    sensor_names=sensor_names,
                )
            )

        return AgentInput(ego_statuses, cameras, lidars)


@dataclass
class Annotations:
    """Dataclass of annotations (e.g. bounding boxes) per frame."""

    boxes: npt.NDArray[np.float32]
    names: List[str]
    velocity_3d: npt.NDArray[np.float32]
    instance_tokens: List[str]
    track_tokens: List[str]

    def __post_init__(self):
        annotation_lengths: Dict[str, int] = {
            attribute_name: len(attribute) for attribute_name, attribute in vars(self).items()
        }
        assert (
            len(set(annotation_lengths.values())) == 1
        ), f"Annotations expects all attributes to have equal length, but got {annotation_lengths}"


@dataclass
class Trajectory:
    """Trajectory dataclass in NAVSIM."""

    poses: npt.NDArray[np.float32]  # local coordinates
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    def __post_init__(self):
        assert self.poses.ndim == 2, "Trajectory poses should have two dimensions for samples and poses."
        assert (
            self.poses.shape[0] == self.trajectory_sampling.num_poses
        ), "Trajectory poses and sampling have unequal number of poses."
        assert self.poses.shape[1] == 3, "Trajectory requires (x, y, heading) at last dim."


@dataclass
class SceneMetadata:
    """Dataclass of scene metadata (e.g. location) per scene."""

    log_name: str
    scene_token: str
    map_name: str
    initial_token: str

    num_history_frames: int
    num_future_frames: int


@dataclass
class Frame:
    """Frame dataclass with privileged information."""

    token: str
    timestamp: int
    roadblock_ids: List[str]
    traffic_lights: List[Tuple[str, bool]]
    annotations: Annotations

    ego_status: EgoStatus
    lidar: Lidar
    cameras: Cameras
    route_centerline: Optional[Any] = None
    route_centerline_segments: Optional[Any] = None


@dataclass
class Scene:
    """Scene dataclass defining a single sample in NAVSIM."""

    # Ground truth information
    scene_metadata: SceneMetadata
    map_api: AbstractMap
    frames: List[Frame]

    def get_future_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:
        """
        Extracts future trajectory of the human operator in local coordinates (ie. ego rear-axle).
        :param num_trajectory_frames: optional number frames to extract poses, defaults to None
        :return: trajectory dataclass
        """

        if num_trajectory_frames is None:
            num_trajectory_frames = self.scene_metadata.num_future_frames

        start_frame_idx = self.scene_metadata.num_history_frames - 1

        global_ego_poses = []
        for frame_idx in range(start_frame_idx, start_frame_idx + num_trajectory_frames + 1):
            global_ego_poses.append(self.frames[frame_idx].ego_status.ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[0]), np.array(global_ego_poses[1:], dtype=np.float64)
        )

        return Trajectory(
            local_ego_poses,
            TrajectorySampling(
                num_poses=len(local_ego_poses),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
        )

    def get_history_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:
        """
        Extracts past trajectory of ego vehicles in local coordinates (ie. ego rear-axle).
        :param num_trajectory_frames: optional number frames to extract poses, defaults to None
        :return: trajectory dataclass
        """

        if num_trajectory_frames is None:
            num_trajectory_frames = self.scene_metadata.num_history_frames

        global_ego_poses = []
        for frame_idx in range(num_trajectory_frames):
            global_ego_poses.append(self.frames[frame_idx].ego_status.ego_pose)

        origin = StateSE2(*global_ego_poses[-1])
        local_ego_poses = convert_absolute_to_relative_se2_array(origin, np.array(global_ego_poses, dtype=np.float64))

        return Trajectory(
            local_ego_poses,
            TrajectorySampling(
                num_poses=len(local_ego_poses),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
        )

    def get_agent_input(self) -> AgentInput:
        """
        Extracts agents input dataclass (without privileged information) from scene.
        :return: agent input dataclass
        """

        local_ego_poses = self.get_history_trajectory().poses
        ego_statuses: List[EgoStatus] = []
        cameras: List[Cameras] = []
        lidars: List[Lidar] = []

        for frame_idx in range(self.scene_metadata.num_history_frames):
            frame_ego_status = self.frames[frame_idx].ego_status

            ego_statuses.append(
                EgoStatus(
                    ego_pose=local_ego_poses[frame_idx],
                    ego_velocity=frame_ego_status.ego_velocity,
                    ego_acceleration=frame_ego_status.ego_acceleration,
                    driving_command=frame_ego_status.driving_command,
                    ego_yaw_rate=frame_ego_status.ego_yaw_rate,
                    ego_curvature=frame_ego_status.ego_curvature,
                    ego_curvature_rate=frame_ego_status.ego_curvature_rate,
                    ego_steering_wheel_angle=frame_ego_status.ego_steering_wheel_angle,
                    ego_steering_wheel_rate=frame_ego_status.ego_steering_wheel_rate,
                )
            )
            cameras.append(self.frames[frame_idx].cameras)
            lidars.append(self.frames[frame_idx].lidar)

        return AgentInput(ego_statuses, cameras, lidars)

    @classmethod
    def _build_map_api(cls, map_name: str) -> Optional[AbstractMap]:
        """Helper classmethod to load map api from name."""
        if map_name == "las_vegas":
            # Mogo conversion uses this as a placeholder. Metric caching injects
            # a Mogo map adapter explicitly; training caches do not need map_api.
            return None
        assert map_name in MAP_LOCATIONS, f"The map name {map_name} is invalid, must be in {MAP_LOCATIONS}"
        return get_maps_api(NUPLAN_MAPS_ROOT, "nuplan-maps-v1.0", map_name)

    @classmethod
    def _build_annotations(cls, scene_frame: Dict) -> Annotations:
        """Helper classmethod to load annotation dataclass from logs."""
        return Annotations(
            boxes=scene_frame["anns"]["gt_boxes"],
            names=scene_frame["anns"]["gt_names"],
            velocity_3d=scene_frame["anns"]["gt_velocity_3d"],
            instance_tokens=scene_frame["anns"]["instance_tokens"],
            track_tokens=scene_frame["anns"]["track_tokens"],
        )

    @classmethod
    def _build_ego_status(cls, scene_frame: Dict) -> EgoStatus:
        """Helper classmethod to load ego status dataclass from logs."""
        ego_translation = scene_frame["ego2global_translation"]
        ego_quaternion = Quaternion(*scene_frame["ego2global_rotation"])
        global_ego_pose = np.array(
            [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
            dtype=np.float64,
        )
        ego_dynamic_state = scene_frame["ego_dynamic_state"]
        ego_velocity = np.array(ego_dynamic_state[:2], dtype=np.float32)
        ego_yaw_rate = _extract_ego_yaw_rate(scene_frame)
        ego_curvature = _compute_ego_curvature(ego_velocity, ego_yaw_rate)
        return EgoStatus(
            ego_pose=global_ego_pose,
            ego_velocity=ego_velocity,
            ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
            driving_command=scene_frame["driving_command"],
            ego_yaw_rate=ego_yaw_rate,
            ego_curvature=ego_curvature,
            ego_steering_wheel_angle=_extract_ego_steering_wheel_angle(scene_frame),
            ego_steering_wheel_rate=_extract_ego_steering_wheel_rate(scene_frame),
            in_global_frame=True,
        )

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        num_future_frames: int,
        sensor_config: SensorConfig,
        load_image_path: bool = False,
        map_api: Optional[AbstractMap] = None,
    ) -> Scene:
        """
        Load scene dataclass from scene dictionary list (for log loading).
        :param scene_dict_list: list of scene frames (in logs)
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of past and current frames to load
        :param num_future_frames: number of future frames to load
        :param sensor_config: sensor config dataclass
        :param map_api: optional injected map api, used by Mogo/internal data adapters.
        :return: scene dataclass
        """
        assert len(scene_dict_list) >= 0, "Scene list is empty!"
        scene_metadata = SceneMetadata(
            log_name=scene_dict_list[num_history_frames - 1]["log_name"],
            scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
            map_name=scene_dict_list[num_history_frames - 1]["map_location"],
            initial_token=scene_dict_list[num_history_frames - 1]["token"],
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
        )
        map_api = map_api or cls._build_map_api(scene_metadata.map_name)

        frames: List[Frame] = []
        for frame_idx in range(len(scene_dict_list)):
            global_ego_status = cls._build_ego_status(scene_dict_list[frame_idx])
            annotations = cls._build_annotations(scene_dict_list[frame_idx])

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)

            cameras = Cameras.from_camera_dict(
                sensor_blobs_path=sensor_blobs_path,
                camera_dict=scene_dict_list[frame_idx]["cams"],
                sensor_names=sensor_names,
                load_image_path=load_image_path,
            )

            lidar = Lidar.from_paths(
                sensor_blobs_path=sensor_blobs_path,
                lidar_path=Path(scene_dict_list[frame_idx]["lidar_path"]),
                sensor_names=sensor_names,
            )

            frame = Frame(
                token=scene_dict_list[frame_idx]["token"],
                timestamp=scene_dict_list[frame_idx]["timestamp"],
                roadblock_ids=scene_dict_list[frame_idx]["roadblock_ids"],
                traffic_lights=scene_dict_list[frame_idx]["traffic_lights"],
                annotations=annotations,
                ego_status=global_ego_status,
                lidar=lidar,
                cameras=cameras,
                route_centerline=scene_dict_list[frame_idx].get("route_centerline"),
                route_centerline_segments=scene_dict_list[frame_idx].get("route_centerline_segments"),
            )
            frames.append(frame)

        ego_curvature_rates = _compute_ego_curvature_rates(
            [frame.ego_status.ego_curvature for frame in frames], scene_dict_list
        )
        for frame, ego_curvature_rate in zip(frames, ego_curvature_rates):
            frame.ego_status.ego_curvature_rate = ego_curvature_rate

        return Scene(scene_metadata=scene_metadata, map_api=map_api, frames=frames)


@dataclass
class SceneFilter:
    """Scene filtering configuration for scene loading."""

    num_history_frames: int = 4
    num_future_frames: int = 10
    frame_interval: Optional[int] = None
    frame_stride: int = 1
    has_route: bool = True

    max_scenes: Optional[int] = None
    log_names: Optional[List[str]] = None
    tokens: Optional[List[str]] = None
    # TODO: expand filter options

    def __post_init__(self):

        if self.frame_interval is None:
            self.frame_interval = self.num_raw_frames

        assert self.num_history_frames >= 1, "SceneFilter: num_history_frames must greater equal one."
        assert self.num_future_frames >= 0, "SceneFilter: num_future_frames must greater equal zero."
        assert self.frame_interval >= 1, "SceneFilter: frame_interval must greater equal one."
        assert self.frame_stride >= 1, "SceneFilter: frame_stride must greater equal one."

    @property
    def num_frames(self) -> int:
        """
        :return: total number for frames for scenes to extract.
        """
        return self.num_history_frames + self.num_future_frames

    @property
    def num_raw_frames(self) -> int:
        """
        :return: number of raw frames required before applying frame_stride.
        """
        return (self.num_frames - 1) * self.frame_stride + 1


@dataclass
class SensorConfig:
    """Configuration dataclass of agent sensors for memory management."""

    # Config values of sensors are either
    # - bool: Whether to load history or not
    # - List[int]: For loading specific history steps
    cam_f0: Union[bool, List[int]]
    cam_l0: Union[bool, List[int]]
    cam_l1: Union[bool, List[int]]
    cam_l2: Union[bool, List[int]]
    cam_r0: Union[bool, List[int]]
    cam_r1: Union[bool, List[int]]
    cam_r2: Union[bool, List[int]]
    cam_b0: Union[bool, List[int]]
    lidar_pc: Union[bool, List[int]]

    def get_sensors_at_iteration(self, iteration: int) -> List[str]:
        """
        Creates a list of sensor identifiers given iteration.
        :param iteration: integer indicating the history iteration.
        :return: list of sensor identifiers to load.
        """
        sensors_at_iteration: List[str] = []
        for sensor_name, sensor_include in asdict(self).items():
            if isinstance(sensor_include, bool) and sensor_include:
                sensors_at_iteration.append(sensor_name)
            elif isinstance(sensor_include, list) and iteration in sensor_include:
                sensors_at_iteration.append(sensor_name)
        return sensors_at_iteration

    @classmethod
    def build_all_sensors(cls, include: Union[bool, List[int]] = True) -> SensorConfig:
        """
        Classmethod to load all sensors with the same specification.
        :param include: boolean or integers for sensors to include, defaults to True
        :return: sensor configuration dataclass
        """
        return SensorConfig(
            cam_f0=include,
            cam_l0=include,
            cam_l1=include,
            cam_l2=include,
            cam_r0=include,
            cam_r1=include,
            cam_r2=include,
            cam_b0=include,
            lidar_pc=include,
        )

    @classmethod
    def build_no_sensors(cls) -> SensorConfig:
        """
        Classmethod to load no sensors.
        :return: sensor configuration dataclass
        """
        return cls.build_all_sensors(include=False)


@dataclass
class PDMResults:
    """Helper dataclass to record PDM results."""

    no_at_fault_collisions: float
    drivable_area_compliance: float
    no_reverse_driving: float

    ego_progress: float
    time_to_collision_within_bound: float
    comfort: float
    driving_direction_compliance: float
    obstacle_clearance: float
    following_safety: float
    centerline_convergence: float
    start_anchor_comfort: float

    score: float
