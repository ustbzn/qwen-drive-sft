from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type, Union
import warnings

import numpy as np
import numpy.typing as npt
from shapely.creation import linestrings
from shapely.geometry import LineString, Point
from shapely.ops import substring
from scipy.interpolate import interp1d

from nuplan.common.actor_state.state_representation import StateSE2

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import calculate_progress, normalize_angle
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import SE2Index
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    array_to_states_se2,
    states_se2_to_array,
)


@dataclass
class PDMPathBoundaryPoint:
    """Boundary point represented in a PDMPath's centerline Frenet frame."""

    s: float
    l: float
    line_type: int


@dataclass
class PDMPathBoundaryProfile:
    """Left/right boundary profile aligned to a PDMPath by progress."""

    left_boundary: List[PDMPathBoundaryPoint]
    right_boundary: List[PDMPathBoundaryPoint]


class PDMPath:
    """Class representing a path to interpolate for PDM."""

    def __init__(
        self,
        discrete_path: List[StateSE2],
        boundary_profile: Optional[PDMPathBoundaryProfile] = None,
    ):
        """
        Constructor for PDMPath
        :param discrete_path: list of (x,y,θ) values
        :param boundary_profile: optional left/right boundary profile in centerline coordinates
        """

        # attribute
        self._discrete_path = discrete_path
        self._boundary_profile = boundary_profile

        # loaded during initialization
        self._states_se2_array = states_se2_to_array(discrete_path)
        self._states_se2_array[:, SE2Index.HEADING] = np.unwrap(self._states_se2_array[:, SE2Index.HEADING], axis=0)
        self._progress = calculate_progress(discrete_path)
        self._linestring = linestrings(self._states_se2_array[:, : SE2Index.HEADING])
        self._interpolator = interp1d(self._progress, self._states_se2_array, axis=0)

    def __reduce__(self) -> Tuple[Type[PDMPath], Tuple[Any, ...]]:
        """Helper for pickling."""
        return self.__class__, (self._discrete_path, self._boundary_profile)

    @property
    def discrete_path(self):
        """Getter for discrete StateSE2 objects of path."""
        return self._discrete_path

    @property
    def length(self):
        """Getter for length of path."""
        return self._progress[-1]

    @property
    def linestring(self) -> LineString:
        """Getter for shapely's linestring of path."""
        return self._linestring

    @property
    def boundary_profile(self) -> Optional[PDMPathBoundaryProfile]:
        """Getter for optional path boundary profile."""
        return self._boundary_profile

    def set_boundary_profile(self, boundary_profile: Optional[PDMPathBoundaryProfile]) -> None:
        """Set optional path boundary profile after path construction."""
        self._boundary_profile = boundary_profile

    def has_boundary_profile(self) -> bool:
        """Return whether both left and right boundary profiles are available."""
        return bool(
            self._boundary_profile
            and self._boundary_profile.left_boundary
            and self._boundary_profile.right_boundary
        )

    def project(self, points: Any) -> Any:
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in line_locate_point", category=RuntimeWarning
        )
        return self._linestring.project(points)

    def interpolate(
        self,
        distances: Union[List[float], npt.NDArray[np.float64]],
        as_array=False,
    ) -> Union[npt.NDArray[np.object_], npt.NDArray[np.float64]]:
        """
        Calculates (x,y,θ) for a given distance along the path.
        :param distances: list of array of distance values
        :param as_array: whether to return in array representation, defaults to False
        :return: array of StateSE2 class or (x,y,θ) values
        """
        clipped_distances = np.clip(distances, 1e-5, self.length)
        interpolated_se2_array = self._interpolator(clipped_distances)
        interpolated_se2_array[..., 2] = normalize_angle(interpolated_se2_array[..., 2])
        interpolated_se2_array[np.isnan(interpolated_se2_array)] = 0.0

        if as_array:
            return interpolated_se2_array

        return array_to_states_se2(interpolated_se2_array)

    @staticmethod
    def _interpolate_boundary_point(
        boundary_points: List[PDMPathBoundaryPoint],
        progress_s: float,
    ) -> Optional[PDMPathBoundaryPoint]:
        """Linearly interpolate lateral offset and use the left interval boundary type."""
        if not boundary_points:
            return None

        if len(boundary_points) == 1:
            point = boundary_points[0]
            return PDMPathBoundaryPoint(float(progress_s), float(point.l), int(point.line_type))

        progress = np.asarray([point.s for point in boundary_points], dtype=np.float64)
        progress_s = float(progress_s)

        if progress_s <= progress[0]:
            point = boundary_points[0]
            return PDMPathBoundaryPoint(progress_s, float(point.l), int(point.line_type))
        if progress_s >= progress[-1]:
            point = boundary_points[-1]
            return PDMPathBoundaryPoint(progress_s, float(point.l), int(point.line_type))

        left_idx = int(np.searchsorted(progress, progress_s, side="right") - 1)
        right_idx = left_idx + 1
        left_point = boundary_points[left_idx]
        right_point = boundary_points[right_idx]

        ds = max(float(right_point.s - left_point.s), 1e-6)
        ratio = (progress_s - float(left_point.s)) / ds
        lateral_offset = float(left_point.l) + ratio * (float(right_point.l) - float(left_point.l))
        return PDMPathBoundaryPoint(progress_s, lateral_offset, int(left_point.line_type))

    def get_boundary_point_at_progress(
        self,
        progress_s: float,
        side: str,
    ) -> Optional[PDMPathBoundaryPoint]:
        """Query one side boundary point at a centerline progress."""
        if self._boundary_profile is None:
            return None

        if side == "left":
            return self._interpolate_boundary_point(self._boundary_profile.left_boundary, progress_s)
        if side == "right":
            return self._interpolate_boundary_point(self._boundary_profile.right_boundary, progress_s)
        raise ValueError(f"Unknown boundary side: {side}")

    def get_boundary_points_at_progress(
        self,
        progress_s: float,
    ) -> Optional[Tuple[PDMPathBoundaryPoint, PDMPathBoundaryPoint]]:
        """Query left and right boundary points at a centerline progress."""
        left_point = self.get_boundary_point_at_progress(progress_s, "left")
        right_point = self.get_boundary_point_at_progress(progress_s, "right")
        if left_point is None or right_point is None:
            return None
        return left_point, right_point

    def get_boundary_points_at_point(
        self,
        xy: npt.NDArray[np.float64],
    ) -> Optional[Tuple[PDMPathBoundaryPoint, PDMPathBoundaryPoint]]:
        """Project a point to the path and query left/right boundary points."""
        progress_s = float(self.project(Point(float(xy[0]), float(xy[1]))))
        return self.get_boundary_points_at_progress(progress_s)

    def substring(self, start_distance: float, end_distance: float) -> LineString:
        """
        Creates a sub-linestring between start and ending distances.
        :param start_distance: distance along the path to start [m]
        :param end_distance:  distance along the path to end [m]
        :return: LineString
        """

        # try faster method fist
        start_distance = np.clip(start_distance, 0.0, self.length)
        end_distance = np.clip(end_distance, 0.0, self.length)
        in_interval = np.logical_and(start_distance <= self._progress, self._progress <= end_distance)
        coordinates = self._states_se2_array[in_interval, :2]
        if len(coordinates) > 1:
            return LineString(coordinates)

        # fallback to slower method of shapely
        return substring(self.linestring, start_distance, end_distance)
