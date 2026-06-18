"""Route-result based centerline builder for Mogo data.

This module intentionally depends only on the RoutingResult-like object. It does
not use the nuPlan map API, the Mogo SQLite adapter, or the PDM Dijkstra route
graph. The goal is to provide one shared interpretation of route-result lanes
and links for metric-cache generation, local-map rendering, and future driving
command extensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nuplan.common.actor_state.state_representation import StateSE2

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


class MogoRouteSegmentType(str, Enum):
    """Atomic route centerline segment type."""

    LANE = "lane"
    LINK = "link"
    EMPTY_LINK = "empty_link"


class MogoRouteCenterlineEndReason(str, Enum):
    """Reason why route-result centerline generation stopped."""

    REACHED_TARGET_LENGTH = "reached_target_length"
    REACHED_TARGET_POST_ROAD = "reached_target_post_road"
    ROUTE_END = "route_end"
    NO_START_MATCH = "no_start_match"
    NO_SUCCESSOR_LINK = "no_successor_link"
    NEXT_ROAD_NOT_FOUND = "next_road_not_found"
    INVALID_GEOMETRY = "invalid_geometry"


@dataclass(frozen=True)
class MogoRouteLaneRef:
    """Lane identity in a route result."""

    road_pkid: int
    section_id: int
    lane_id: int


@dataclass
class MogoRouteRoadEntry:
    """Flattened route_roadlink entry."""

    global_idx: int
    path_idx: int
    roadlink_idx: int
    road_pkid: int
    route_roadlink: Any


@dataclass
class MogoRouteStartMatch:
    """Optional caller-provided start match."""

    kind: str  # "lane" or "link"
    road_pkid: int
    section_id: Optional[int] = None
    lane_id: Optional[int] = None
    route_global_idx: Optional[int] = None
    progress: float = 0.0
    link: Optional[Any] = None
    distance: float = math.inf


@dataclass
class MogoRouteCenterlineSegment:
    """Atomic segment used to build the final centerline geometry."""

    segment_type: MogoRouteSegmentType
    points: List[Tuple[float, float]]
    road_pkid: Optional[int] = None
    section_id: Optional[int] = None
    lane_id: Optional[int] = None
    link_pkid: Optional[int] = None
    from_lane: Optional[MogoRouteLaneRef] = None
    to_lane: Optional[MogoRouteLaneRef] = None
    link_type: Optional[int] = None
    recommended: Optional[bool] = None
    rank: Optional[int] = None


@dataclass
class MogoRouteCenterlineResult:
    """Route-result centerline and the structured evidence used to build it."""

    centerline: Optional[PDMPath]
    segments: List[MogoRouteCenterlineSegment]
    start_match: Optional[MogoRouteStartMatch]
    end_reason: MogoRouteCenterlineEndReason
    length: float
    reached_target_post_road: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MogoRouteCenterlineConfig:
    """Config for route-result centerline generation."""

    target_length_m: float = 200.0
    max_steps: int = 256
    nearest_link_distance_margin_m: float = 0.5
    duplicate_point_eps_m: float = 1e-2


def _has_field(message: Any, field_name: str) -> bool:
    """Safely check proto2 optional field presence."""
    if message is None or not hasattr(message, "HasField"):
        return hasattr(message, field_name)
    try:
        return bool(message.HasField(field_name))
    except (TypeError, ValueError):
        return hasattr(message, field_name)


def _points_to_xy(points: Sequence[Any]) -> List[Tuple[float, float]]:
    """Convert proto geometry points to xy tuples."""
    return [(float(point.x), float(point.y)) for point in points]


def _polyline_length(points: Sequence[Tuple[float, float]]) -> float:
    """Compute polyline length."""
    if len(points) < 2:
        return 0.0
    return float(
        sum(
            math.hypot(points[idx][0] - points[idx - 1][0], points[idx][1] - points[idx - 1][1])
            for idx in range(1, len(points))
        )
    )


def _project_point_to_polyline(
    point_xy: Tuple[float, float],
    polyline_xy: Sequence[Tuple[float, float]],
) -> Tuple[float, float, float]:
    """Project point to polyline and return distance, progress, length."""
    if len(polyline_xy) < 2:
        return math.inf, 0.0, 0.0

    px, py = point_xy
    best_distance = math.inf
    best_progress = 0.0
    total_length = 0.0

    for idx in range(1, len(polyline_xy)):
        ax, ay = polyline_xy[idx - 1]
        bx, by = polyline_xy[idx]
        vx = bx - ax
        vy = by - ay
        segment_length_sq = vx * vx + vy * vy
        if segment_length_sq <= 1e-12:
            continue

        segment_length = math.sqrt(segment_length_sq)
        ratio = ((px - ax) * vx + (py - ay) * vy) / segment_length_sq
        ratio = min(1.0, max(0.0, ratio))
        proj_x = ax + ratio * vx
        proj_y = ay + ratio * vy
        distance = math.hypot(px - proj_x, py - proj_y)
        progress = total_length + ratio * segment_length
        if distance < best_distance:
            best_distance = distance
            best_progress = progress
        total_length += segment_length

    return best_distance, best_progress, total_length


def _slice_polyline_from_progress(
    points: Sequence[Tuple[float, float]],
    start_progress: float,
) -> List[Tuple[float, float]]:
    """Slice a polyline from progress to its end."""
    if len(points) < 2:
        return list(points)

    start_progress = max(0.0, float(start_progress))
    total_length = 0.0
    output: List[Tuple[float, float]] = []

    for idx in range(1, len(points)):
        start = points[idx - 1]
        end = points[idx]
        segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
        if segment_length <= 1e-12:
            continue

        next_length = total_length + segment_length
        if start_progress <= next_length:
            ratio = min(1.0, max(0.0, (start_progress - total_length) / segment_length))
            sliced_start = (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
            output.append(sliced_start)
            output.extend(points[idx:])
            return output
        total_length = next_length

    return [points[-1]]


def _points_to_states(points: Sequence[Tuple[float, float]]) -> List[StateSE2]:
    """Convert xy points to StateSE2 with finite-difference heading."""
    deduped: List[Tuple[float, float]] = []
    for point in points:
        if deduped and math.hypot(point[0] - deduped[-1][0], point[1] - deduped[-1][1]) <= 1e-6:
            continue
        deduped.append(point)

    if not deduped:
        return []
    if len(deduped) == 1:
        return [StateSE2(deduped[0][0], deduped[0][1], 0.0)]

    states: List[StateSE2] = []
    for idx, point in enumerate(deduped):
        if idx < len(deduped) - 1:
            next_point = deduped[idx + 1]
            heading = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
        else:
            previous_point = deduped[idx - 1]
            heading = math.atan2(point[1] - previous_point[1], point[0] - previous_point[0])
        states.append(StateSE2(float(point[0]), float(point[1]), float(heading)))
    return states


class MogoRouteResultCenterlineBuilder:
    """Build a route-result based centerline from route roads and lane links."""

    def __init__(self, route_result: Any, config: Optional[MogoRouteCenterlineConfig] = None):
        self._route_result = route_result
        self._config = config or MogoRouteCenterlineConfig()
        self._entries = self._flatten_route_entries(route_result)
        self._entry_by_pair = {(entry.path_idx, entry.roadlink_idx): entry for entry in self._entries}

    @property
    def route_entries(self) -> List[MogoRouteRoadEntry]:
        """Return flattened route entries."""
        return self._entries

    def build(
        self,
        ego_xy: Tuple[float, float],
        start_match: Optional[MogoRouteStartMatch] = None,
        target_post_road_pkid: Optional[int] = None,
    ) -> MogoRouteCenterlineResult:
        """Build centerline from route result."""
        if not self._entries:
            return self._empty_result(start_match, MogoRouteCenterlineEndReason.NO_START_MATCH)

        start_match = start_match or self.find_nearest_start_match(ego_xy)
        if start_match is None:
            return self._empty_result(None, MogoRouteCenterlineEndReason.NO_START_MATCH)

        segments: List[MogoRouteCenterlineSegment] = []
        current_entry: Optional[MogoRouteRoadEntry]
        current_lane: Optional[Any]
        lane_start_progress = 0.0

        if start_match.kind == "link" and start_match.link is not None:
            link_points = _points_to_xy(start_match.link.geom)
            link_points = _slice_polyline_from_progress(link_points, start_match.progress)
            if len(link_points) >= 2:
                segments.append(self._make_link_segment(start_match.link, link_points))
            current_entry = self._find_entry_after(start_match.route_global_idx, int(start_match.link.to_road_pkid))
            current_lane = self._find_lane(current_entry, int(start_match.link.to_section_id), int(start_match.link.to_lane_id))
        else:
            current_entry = self._entry_by_global_idx(start_match.route_global_idx)
            current_lane = self._find_lane(current_entry, start_match.section_id, start_match.lane_id)
            lane_start_progress = start_match.progress

        if current_entry is None or current_lane is None:
            return self._empty_result(start_match, MogoRouteCenterlineEndReason.INVALID_GEOMETRY)

        reached_target_post_road = False
        end_reason = MogoRouteCenterlineEndReason.ROUTE_END

        for _ in range(self._config.max_steps):
            lane_points = _points_to_xy(current_lane.point)
            lane_points = _slice_polyline_from_progress(lane_points, lane_start_progress)
            if len(lane_points) < 2:
                end_reason = MogoRouteCenterlineEndReason.INVALID_GEOMETRY
                break

            segments.append(self._make_lane_segment(current_lane, lane_points))
            lane_start_progress = 0.0

            length = _polyline_length(self._segments_to_points(segments))
            if target_post_road_pkid is not None and int(current_entry.road_pkid) == int(target_post_road_pkid):
                reached_target_post_road = True
            if reached_target_post_road and length >= self._config.target_length_m:
                end_reason = MogoRouteCenterlineEndReason.REACHED_TARGET_LENGTH
                break
            if target_post_road_pkid is None and length >= self._config.target_length_m:
                end_reason = MogoRouteCenterlineEndReason.REACHED_TARGET_LENGTH
                break

            link = self._select_successor_link(current_entry, current_lane)
            if link is None:
                end_reason = MogoRouteCenterlineEndReason.NO_SUCCESSOR_LINK
                break

            link_points = _points_to_xy(link.geom)
            if len(link_points) >= 2:
                segments.append(self._make_link_segment(link, link_points))
            else:
                segments.append(self._make_empty_link_segment(link))

            next_entry = self._find_entry_after(current_entry.global_idx, int(link.to_road_pkid))
            if next_entry is None:
                end_reason = MogoRouteCenterlineEndReason.NEXT_ROAD_NOT_FOUND
                break
            next_lane = self._find_lane(next_entry, int(link.to_section_id), int(link.to_lane_id))
            if next_lane is None:
                end_reason = MogoRouteCenterlineEndReason.INVALID_GEOMETRY
                break

            current_entry = next_entry
            current_lane = next_lane

        else:
            end_reason = MogoRouteCenterlineEndReason.ROUTE_END

        points = self._segments_to_points(segments)
        states = _points_to_states(points)
        centerline = PDMPath(states) if len(states) >= 2 else None
        return MogoRouteCenterlineResult(
            centerline=centerline,
            segments=segments,
            start_match=start_match,
            end_reason=end_reason,
            length=_polyline_length(points),
            reached_target_post_road=reached_target_post_road,
        )

    def find_nearest_start_match(self, ego_xy: Tuple[float, float]) -> Optional[MogoRouteStartMatch]:
        """Find nearest route lane/link using the same idea as the pkl exporter."""
        lane_match = self._find_nearest_route_lane(ego_xy)
        if lane_match is None:
            return None

        link_match = self._find_nearest_route_link_around_lane(ego_xy, lane_match)
        if link_match is not None and link_match.distance <= lane_match.distance + self._config.nearest_link_distance_margin_m:
            return link_match
        return lane_match

    @staticmethod
    def _flatten_route_entries(route_result: Any) -> List[MogoRouteRoadEntry]:
        """Flatten route paths while removing adjacent path stitching duplicates."""
        entries: List[MogoRouteRoadEntry] = []
        previous_path_last_id: Optional[int] = None

        for path_idx, path in enumerate(getattr(route_result, "paths", [])):
            path_entries: List[MogoRouteRoadEntry] = []
            for roadlink_idx, route_roadlink in enumerate(getattr(path, "route_roadlinks", [])):
                if not _has_field(route_roadlink, "route_road"):
                    continue
                route_road = route_roadlink.route_road
                if not _has_field(route_road, "pkid") or not route_road.pkid:
                    continue
                path_entries.append(
                    MogoRouteRoadEntry(
                        global_idx=-1,
                        path_idx=path_idx,
                        roadlink_idx=roadlink_idx,
                        road_pkid=int(route_road.pkid),
                        route_roadlink=route_roadlink,
                    )
                )

            if not path_entries:
                continue
            entries_to_append = path_entries
            if previous_path_last_id is not None and path_entries[0].road_pkid == previous_path_last_id:
                entries_to_append = path_entries[1:]
            for entry in entries_to_append:
                entry.global_idx = len(entries)
                entries.append(entry)
            previous_path_last_id = path_entries[-1].road_pkid

        return entries

    def _find_nearest_route_lane(self, ego_xy: Tuple[float, float]) -> Optional[MogoRouteStartMatch]:
        best_match: Optional[MogoRouteStartMatch] = None
        for entry in self._entries:
            for lane in self._iter_lanes(entry):
                lane_points = _points_to_xy(lane.point)
                distance, progress, _ = _project_point_to_polyline(ego_xy, lane_points)
                if math.isinf(distance):
                    continue
                if best_match is None or distance < best_match.distance:
                    best_match = MogoRouteStartMatch(
                        kind="lane",
                        route_global_idx=entry.global_idx,
                        road_pkid=int(lane.road_pkid),
                        section_id=int(lane.section_id),
                        lane_id=int(lane.lane_id),
                        progress=progress,
                        distance=distance,
                    )
        return best_match

    def _find_nearest_route_link_around_lane(
        self,
        ego_xy: Tuple[float, float],
        lane_match: MogoRouteStartMatch,
    ) -> Optional[MogoRouteStartMatch]:
        best_match: Optional[MogoRouteStartMatch] = None
        current_idx = lane_match.route_global_idx
        if current_idx is None:
            return None

        for idx in (current_idx - 1, current_idx, current_idx + 1):
            entry = self._entry_by_global_idx(idx)
            if entry is None:
                continue
            for link in getattr(entry.route_roadlink, "route_links", []):
                if len(getattr(link, "geom", [])) < 2:
                    continue
                is_predecessor = int(link.to_road_pkid) == int(lane_match.road_pkid)
                is_successor = int(link.from_road_pkid) == int(lane_match.road_pkid)
                if not is_predecessor and not is_successor:
                    continue
                link_points = _points_to_xy(link.geom)
                distance, progress, _ = _project_point_to_polyline(ego_xy, link_points)
                if math.isinf(distance):
                    continue
                if best_match is None or distance < best_match.distance:
                    best_match = MogoRouteStartMatch(
                        kind="link",
                        route_global_idx=entry.global_idx,
                        road_pkid=int(link.from_road_pkid),
                        progress=progress,
                        link=link,
                        distance=distance,
                    )
        return best_match

    def _select_successor_link(self, current_entry: MogoRouteRoadEntry, current_lane: Any) -> Optional[Any]:
        candidates = self._candidate_links(current_entry, current_lane)
        if not candidates:
            return None

        current_recommended = self._recommended_lane(current_entry)
        if current_recommended is None:
            return self._shortest_link(candidates)

        alignment = self._recommended_alignment_link(current_entry, current_recommended)
        if alignment is None:
            return self._shortest_link(candidates)

        next_entry = self._find_entry_after(current_entry.global_idx, int(alignment.to_road_pkid))
        if next_entry is None:
            return self._shortest_link(candidates)

        base_next_lane = self._find_lane(next_entry, int(alignment.to_section_id), int(alignment.to_lane_id))
        if base_next_lane is None:
            return self._shortest_link(candidates)

        current_offset = self._lane_order_offset(current_entry, current_lane, current_recommended)
        desired_lane = self._parallel_lane_by_offset(next_entry, base_next_lane, current_offset)
        if desired_lane is None:
            return self._shortest_link(candidates)

        return self._closest_link_to_lane(candidates, next_entry, desired_lane) or self._shortest_link(candidates)

    def _recommended_alignment_link(self, current_entry: MogoRouteRoadEntry, current_recommended: Any) -> Optional[Any]:
        rec_links = [
            link
            for link in self._candidate_links(current_entry, current_recommended)
            if len(getattr(link, "geom", [])) >= 2
        ]
        if not rec_links:
            return None

        scored_links: List[Tuple[float, float, Any]] = []
        for link in rec_links:
            next_entry = self._find_entry_after(current_entry.global_idx, int(link.to_road_pkid))
            next_recommended = self._recommended_lane(next_entry) if next_entry is not None else None
            if next_recommended is None:
                lane_distance = 0.0
            else:
                lane_distance = abs(int(link.to_lane_id) - int(next_recommended.lane_id))
            scored_links.append((lane_distance, _polyline_length(_points_to_xy(link.geom)), link))
        scored_links.sort(key=lambda item: (item[0], item[1]))
        return scored_links[0][2]

    def _candidate_links(self, current_entry: MogoRouteRoadEntry, lane: Any) -> List[Any]:
        return [
            link
            for link in getattr(current_entry.route_roadlink, "route_links", [])
            if int(link.from_road_pkid) == int(lane.road_pkid)
            and int(link.from_section_id) == int(lane.section_id)
            and int(link.from_lane_id) == int(lane.lane_id)
        ]

    @staticmethod
    def _shortest_link(links: Sequence[Any]) -> Optional[Any]:
        if not links:
            return None
        return sorted(links, key=lambda link: _polyline_length(_points_to_xy(getattr(link, "geom", []))))[0]

    def _closest_link_to_lane(self, links: Sequence[Any], next_entry: MogoRouteRoadEntry, desired_lane: Any) -> Optional[Any]:
        scored_links: List[Tuple[int, float, Any]] = []
        desired_order = self._lane_order(next_entry, desired_lane)
        for link in links:
            if int(link.to_road_pkid) != int(next_entry.road_pkid):
                continue
            to_lane = self._find_lane(next_entry, int(link.to_section_id), int(link.to_lane_id))
            if to_lane is None:
                continue
            scored_links.append(
                (
                    abs(self._lane_order(next_entry, to_lane) - desired_order),
                    _polyline_length(_points_to_xy(getattr(link, "geom", []))),
                    link,
                )
            )
        if not scored_links:
            return None
        scored_links.sort(key=lambda item: (item[0], item[1]))
        return scored_links[0][2]

    def _lane_order_offset(self, entry: MogoRouteRoadEntry, lane: Any, reference_lane: Any) -> int:
        return self._lane_order(entry, lane) - self._lane_order(entry, reference_lane)

    def _parallel_lane_by_offset(
        self,
        entry: MogoRouteRoadEntry,
        base_lane: Any,
        offset: int,
    ) -> Optional[Any]:
        lanes = self._iter_lanes(entry)
        if not lanes:
            return None
        if offset == 0:
            return base_lane

        base_order = self._lane_order(entry, base_lane)
        candidates: List[Tuple[int, int, Any]] = []
        for lane in lanes:
            lane_offset = self._lane_order(entry, lane) - base_order
            if lane_offset == offset:
                return lane
            if lane_offset == 0:
                continue
            if (lane_offset > 0) == (offset > 0) and abs(lane_offset) <= abs(offset):
                candidates.append((abs(abs(offset) - abs(lane_offset)), -abs(lane_offset), lane))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]))
            return candidates[0][2]
        return base_lane

    def _recommended_lane(self, entry: Optional[MogoRouteRoadEntry]) -> Optional[Any]:
        if entry is None:
            return None
        recommended = [lane for lane in self._iter_lanes(entry) if bool(getattr(lane, "recommended", False))]
        if not recommended:
            return None
        recommended.sort(key=lambda lane: (abs(int(getattr(lane, "rank", 0))), int(lane.lane_id)))
        return recommended[0]

    def _find_entry_after(self, current_global_idx: Optional[int], road_pkid: int) -> Optional[MogoRouteRoadEntry]:
        start_idx = -1 if current_global_idx is None else int(current_global_idx)
        for entry in self._entries:
            if entry.global_idx <= start_idx:
                continue
            if int(entry.road_pkid) == int(road_pkid):
                return entry
        return None

    def _entry_by_global_idx(self, global_idx: Optional[int]) -> Optional[MogoRouteRoadEntry]:
        if global_idx is None:
            return None
        for entry in self._entries:
            if entry.global_idx == int(global_idx):
                return entry
        return None

    def _find_lane(self, entry: Optional[MogoRouteRoadEntry], section_id: Optional[int], lane_id: Optional[int]) -> Optional[Any]:
        if entry is None or section_id is None or lane_id is None:
            return None
        for lane in self._iter_lanes(entry):
            if int(lane.section_id) == int(section_id) and int(lane.lane_id) == int(lane_id):
                return lane
        return None

    @staticmethod
    def _iter_lanes(entry: MogoRouteRoadEntry) -> List[Any]:
        lanes: List[Any] = []
        road = entry.route_roadlink.route_road
        for section in getattr(road, "sections", []):
            lanes.extend(list(getattr(section, "lanes", [])))
        lanes.sort(key=lambda lane: (int(lane.section_id), int(lane.lane_id)))
        return lanes

    def _lane_order(self, entry: MogoRouteRoadEntry, lane: Any) -> int:
        lanes = self._iter_lanes(entry)
        for idx, candidate in enumerate(lanes):
            if int(candidate.section_id) == int(lane.section_id) and int(candidate.lane_id) == int(lane.lane_id):
                return idx
        return 0

    @staticmethod
    def _make_lane_segment(lane: Any, points: List[Tuple[float, float]]) -> MogoRouteCenterlineSegment:
        return MogoRouteCenterlineSegment(
            segment_type=MogoRouteSegmentType.LANE,
            points=points,
            road_pkid=int(lane.road_pkid),
            section_id=int(lane.section_id),
            lane_id=int(lane.lane_id),
            recommended=bool(getattr(lane, "recommended", False)),
            rank=int(getattr(lane, "rank", 0)),
        )

    @staticmethod
    def _make_link_segment(link: Any, points: List[Tuple[float, float]]) -> MogoRouteCenterlineSegment:
        return MogoRouteCenterlineSegment(
            segment_type=MogoRouteSegmentType.LINK,
            points=points,
            link_pkid=int(link.pkid) if _has_field(link, "pkid") else None,
            from_lane=MogoRouteLaneRef(int(link.from_road_pkid), int(link.from_section_id), int(link.from_lane_id)),
            to_lane=MogoRouteLaneRef(int(link.to_road_pkid), int(link.to_section_id), int(link.to_lane_id)),
            link_type=int(link.type) if _has_field(link, "type") else None,
        )

    @staticmethod
    def _make_empty_link_segment(link: Any) -> MogoRouteCenterlineSegment:
        return MogoRouteCenterlineSegment(
            segment_type=MogoRouteSegmentType.EMPTY_LINK,
            points=[],
            link_pkid=int(link.pkid) if _has_field(link, "pkid") else None,
            from_lane=MogoRouteLaneRef(int(link.from_road_pkid), int(link.from_section_id), int(link.from_lane_id)),
            to_lane=MogoRouteLaneRef(int(link.to_road_pkid), int(link.to_section_id), int(link.to_lane_id)),
            link_type=int(link.type) if _has_field(link, "type") else None,
        )

    def _segments_to_points(self, segments: Sequence[MogoRouteCenterlineSegment]) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        for segment in segments:
            for point in segment.points:
                if points and math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) < self._config.duplicate_point_eps_m:
                    continue
                points.append(point)
        return points

    @staticmethod
    def _empty_result(
        start_match: Optional[MogoRouteStartMatch],
        end_reason: MogoRouteCenterlineEndReason,
    ) -> MogoRouteCenterlineResult:
        return MogoRouteCenterlineResult(
            centerline=None,
            segments=[],
            start_match=start_match,
            end_reason=end_reason,
            length=0.0,
        )
