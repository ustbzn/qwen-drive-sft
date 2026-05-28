"""Mogo internal HD map adapter skeleton.

This module adapts Mogo SQLite HD map data to the nuPlan/NAVSIM map
interfaces consumed by metric-cache and PDM scoring.

The current implementation intentionally completes the first SQLite reader
closure only: database inspection, route subgraph reads, local-radius reads,
SQLite geometry BLOB parsing, and WGS84-to-UTM conversion. Semantic layer
construction and nuPlan map-object wrappers are kept as explicit follow-up
steps.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from nuplan.common.actor_state.state_representation import Point2D, StateSE2
except ModuleNotFoundError:  # pragma: no cover - fallback for standalone reader checks
    @dataclass(frozen=True)
    class Point2D:  # type: ignore[no-redef]
        x: float
        y: float

        @property
        def array(self) -> np.ndarray:
            return np.asarray([self.x, self.y], dtype=np.float64)

    @dataclass(frozen=True)
    class StateSE2:  # type: ignore[no-redef]
        x: float
        y: float
        heading: float

        @property
        def point(self) -> Point2D:
            return Point2D(self.x, self.y)

        @property
        def array(self) -> np.ndarray:
            return np.asarray([self.x, self.y, self.heading], dtype=np.float64)

try:
    from nuplan.common.maps.abstract_map import AbstractMap
    from nuplan.common.maps.abstract_map_objects import MapObject
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
except ModuleNotFoundError:  # pragma: no cover - reader can run without full nuPlan map deps
    class AbstractMap:  # type: ignore[no-redef]
        pass

    MapObject = Any  # type: ignore[misc, assignment]

    class SemanticMapLayer:  # type: ignore[no-redef]
        LANE = "LANE"
        LANE_CONNECTOR = "LANE_CONNECTOR"
        ROADBLOCK = "ROADBLOCK"
        ROADBLOCK_CONNECTOR = "ROADBLOCK_CONNECTOR"
        INTERSECTION = "INTERSECTION"
        CARPARK_AREA = "CARPARK_AREA"
        DRIVABLE_AREA = "DRIVABLE_AREA"
        BASELINE_PATHS = "BASELINE_PATHS"
        WALKWAYS = "WALKWAYS"

from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


logger = logging.getLogger(__name__)


MOGO_NUPLAN_LAYERS: Tuple[str, ...] = (
    "LANE",
    "LANE_CONNECTOR",
    "ROADBLOCK",
    "ROADBLOCK_CONNECTOR",
    "INTERSECTION",
    "STOP_LINE",
    "CROSSWALK",
    "DRIVABLE_AREA",
    "TRAFFIC_LIGHT",
    "STOP_SIGN",
    "BOUNDARIES",
    "BASELINE_PATHS",
    "WALKWAYS",
    "CARPARK_AREA",
    "PUDO",
    "SPEED_BUMP",
)


MOGO_CORE_TABLE_COLUMNS: Mapping[str, Tuple[str, ...]] = {
    "TX_LANE": (
        "road_pkid",
        "section_id",
        "lane_id",
        "geom",
        "left_boundary",
        "right_boundary",
        "lane_type",
        "lane_arrow",
        "speed_limit",
    ),
    "TX_LANE_BOUNDARY": ("pkid", "lane_mark", "geom"),
    "TX_LANE_LINK": (
        "pkid",
        "from_road_pkid",
        "from_section_id",
        "from_lane_id",
        "to_road_pkid",
        "to_section_id",
        "to_lane_id",
        "link_type",
        "geom",
    ),
    "TX_ROAD": ("pkid", "id", "name", "junction_pkid", "road_type", "direction", "geom"),
    "TX_JUNCTION": ("pkid", "name", "type", "road_pkid", "enter_road_pkid", "inside_road_pkid", "exit_road_pkid", "geom"),
}


MOGO_OPTIONAL_TABLES: Tuple[str, ...] = (
    "TX_OBJECT",
    "TX_OBJECT_GEOM",
    "TX_OBJECT_REL",
    "TX_ROAD_BOUNDARY",
    "TX_GREENBELTS",
    "TX_SAFETYISLAND",
)


@dataclass(frozen=True)
class MogoSQLiteMapConfig:
    """Configuration for building a Mogo SQLite-backed map adapter."""

    sqlite_path: Path
    map_name: str = "mogo"
    query_radius_m: float = 150.0
    max_segment_length_m: float = 2.0
    enable_route_edge_filter: bool = True
    geometry_srid: int = 4326
    utm_zone: int = 50
    utm_south: bool = False


@dataclass(frozen=True)
class MogoRouteContext:
    """Optional route-level context used to narrow route lane graph branches."""

    roadblock_ids: Sequence[str]
    edge_ids: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class MogoSQLiteInspection:
    """One-time diagnostics for a Mogo SQLite map database."""

    tables: Set[str]
    table_columns: Dict[str, Set[str]]
    row_counts: Dict[str, int]
    geometry_srid: int
    rtree_tables: Set[str]


@dataclass(frozen=True)
class MogoRawHDMap:
    """Raw map records read from SQLite before layer building and object wrapping."""

    lanes: List[Dict[str, Any]] = field(default_factory=list)
    lane_boundaries: List[Dict[str, Any]] = field(default_factory=list)
    lane_links: List[Dict[str, Any]] = field(default_factory=list)
    roads: List[Dict[str, Any]] = field(default_factory=list)
    junctions: List[Dict[str, Any]] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)


def _mogo_lane_id(road_pkid: Any, section_id: Any, lane_id: Any) -> str:
    """Build the stable lane identifier used by the Mogo/NAVSIM adapter."""
    return f"lane_{int(road_pkid)}_{int(section_id)}_{int(lane_id)}"


def _mogo_roadblock_id(road_pkid: Any) -> str:
    """Build the roadblock identifier used by the Mogo/NAVSIM adapter."""
    return f"roadblock_{int(road_pkid)}"


def _mogo_link_id(link_pkid: Any) -> str:
    """Build the lane connector identifier used by the Mogo/NAVSIM adapter."""
    return f"link_{int(link_pkid)}"


def _mogo_roadblock_connector_id(from_roadblock_id: str, to_roadblock_id: str) -> str:
    """Build a stable roadblock-connector id for a pair of connected roadblocks."""
    return f"roadblock_connector_{from_roadblock_id}_{to_roadblock_id}"


def _mogo_junction_id(junction_pkid: Any) -> str:
    """Build the intersection identifier used by the Mogo/NAVSIM adapter."""
    return f"intersection_{int(junction_pkid)}"


def _mogo_parse_roadblock_id(roadblock_id: str) -> Optional[int]:
    """Parse `roadblock_<pkid>` back to an integer road pkid."""
    prefix = "roadblock_"
    if not roadblock_id.startswith(prefix):
        return None
    try:
        return int(roadblock_id[len(prefix) :])
    except ValueError:
        return None


class MogoSQLiteReader:
    """Reads raw Mogo HD map records from SQLite.

    The reader deliberately avoids SpatiaLite SQL functions in the first
    runnable version. The current environment can open SQLite files but cannot
    load `mod_spatialite`, so geometry is decoded from SQLite BLOBs directly and
    converted from WGS84 lon/lat into the same UTM frame used by localization.
    """

    def __init__(self, config: MogoSQLiteMapConfig):
        self._config = config
        self._connection: Optional[sqlite3.Connection] = None
        self._inspection: Optional[MogoSQLiteInspection] = None
        self._wgs84_to_utm = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{self._utm_epsg_code}",
            always_xy=True,
        )
        self._utm_to_wgs84 = Transformer.from_crs(
            f"EPSG:{self._utm_epsg_code}",
            "EPSG:4326",
            always_xy=True,
        )

    @property
    def _utm_epsg_code(self) -> int:
        """Return the EPSG code for the configured UTM zone."""
        return (32700 if self._config.utm_south else 32600) + int(self._config.utm_zone)

    @property
    def inspection(self) -> MogoSQLiteInspection:
        """Return the cached database inspection result."""
        if self._inspection is None:
            self.inspect_database()
        assert self._inspection is not None
        return self._inspection

    def connect(self) -> sqlite3.Connection:
        """Open the SQLite connection if needed."""
        if self._connection is not None:
            return self._connection

        sqlite_path = Path(self._config.sqlite_path)
        if not sqlite_path.is_file():
            raise FileNotFoundError(f"Mogo map sqlite not found: {sqlite_path}")

        self._connection = sqlite3.connect(str(sqlite_path))
        self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._inspection = None

    def inspect_database(self, force: bool = False) -> MogoSQLiteInspection:
        """Validate required tables/columns and confirm BLOB geometry once."""
        if self._inspection is not None and not force:
            return self._inspection

        connection = self.connect()
        tables = self._read_table_names(connection)
        missing_tables = set(MOGO_CORE_TABLE_COLUMNS) - tables
        if missing_tables:
            raise RuntimeError(f"Mogo sqlite missing core tables: {sorted(missing_tables)}")

        table_columns: Dict[str, Set[str]] = {}
        row_counts: Dict[str, int] = {}
        for table in sorted(set(MOGO_CORE_TABLE_COLUMNS) | set(MOGO_OPTIONAL_TABLES)):
            if table not in tables:
                continue
            columns = self._read_table_columns(connection, table)
            table_columns[table] = columns
            row_counts[table] = self._read_row_count(connection, table)

        missing_columns = {
            table: sorted(set(required_columns) - table_columns[table])
            for table, required_columns in MOGO_CORE_TABLE_COLUMNS.items()
            if set(required_columns) - table_columns[table]
        }
        if missing_columns:
            raise RuntimeError(f"Mogo sqlite missing core columns: {missing_columns}")

        srids = {
            self._validate_geometry_blob(connection, "TX_LANE", expect_polygon=False),
            self._validate_geometry_blob(connection, "TX_LANE_BOUNDARY", expect_polygon=False),
            self._validate_geometry_blob(connection, "TX_LANE_LINK", expect_polygon=False),
            self._validate_geometry_blob(connection, "TX_ROAD", expect_polygon=False),
            self._validate_geometry_blob(connection, "TX_JUNCTION", expect_polygon=True),
        }
        srids.discard(None)
        if not srids:
            raise RuntimeError("Mogo sqlite contains no readable geometry BLOB samples")
        if len(srids) > 1:
            raise RuntimeError(f"Mogo sqlite contains mixed geometry SRIDs: {sorted(srids)}")

        geometry_srid = srids.pop()
        if geometry_srid != self._config.geometry_srid:
            raise RuntimeError(
                f"Mogo sqlite geometry SRID is {geometry_srid}, expected {self._config.geometry_srid}"
            )

        rtree_tables = {table for table in tables if table.startswith("idx_TX_") and table.endswith("_geom")}
        self._inspection = MogoSQLiteInspection(
            tables=tables,
            table_columns=table_columns,
            row_counts=row_counts,
            geometry_srid=geometry_srid,
            rtree_tables=rtree_tables,
        )
        return self._inspection

    def read_route_subgraph(self, road_pkids: Sequence[int]) -> MogoRawHDMap:
        """Read route-related road/lane/link/boundary/junction records by road ids."""
        road_pkid_set = {int(road_pkid) for road_pkid in road_pkids}
        if not road_pkid_set:
            return MogoRawHDMap()

        roads = self.read_roads_by_ids(sorted(road_pkid_set))
        lanes = self.read_lanes_by_road_ids(sorted(road_pkid_set))
        lane_links = self.read_lane_links_by_road_ids(sorted(road_pkid_set))

        boundary_ids = self._collect_boundary_ids(lanes)
        lane_boundaries = self.read_lane_boundaries_by_ids(boundary_ids)
        junctions = self.read_junctions_by_ids(self._collect_junction_ids(roads))

        return MogoRawHDMap(
            lanes=lanes,
            lane_boundaries=lane_boundaries,
            lane_links=lane_links,
            roads=roads,
            junctions=junctions,
        )

    def read_local_region(self, ego_position: Point2D, radius_m: float) -> MogoRawHDMap:
        """Read local map records around ego position using SQLite RTree MBRs."""
        lon, lat = self._utm_to_lonlat(float(ego_position.x), float(ego_position.y))
        lon_min, lon_max, lat_min, lat_max = self._lonlat_bbox(lon, lat, radius_m)

        lane_rowids = self._read_rtree_pkids("idx_TX_LANE_geom", lon_min, lon_max, lat_min, lat_max)
        lanes = self.read_lanes_by_rowids(lane_rowids)

        road_pkid_set = {int(lane["road_pkid"]) for lane in lanes if lane.get("road_pkid") is not None}
        road_pkid_set.update(self._read_rtree_pkids("idx_TX_ROAD_geom", lon_min, lon_max, lat_min, lat_max))
        roads = self.read_roads_by_ids(sorted(road_pkid_set))

        link_pkids = self._read_rtree_pkids("idx_TX_LANE_LINK_geom", lon_min, lon_max, lat_min, lat_max)
        lane_links_by_pkid = {link["pkid"]: link for link in self.read_lane_links_by_ids(link_pkids)}
        for link in self.read_lane_links_by_road_ids(sorted(road_pkid_set)):
            lane_links_by_pkid.setdefault(link["pkid"], link)
        lane_links = list(lane_links_by_pkid.values())

        junction_pkids = set(self._read_rtree_pkids("idx_TX_JUNCTION_geom", lon_min, lon_max, lat_min, lat_max))
        junction_pkids.update(self._collect_junction_ids(roads))
        junctions = self.read_junctions_by_ids(sorted(junction_pkids))

        boundary_ids = self._collect_boundary_ids(lanes)
        lane_boundaries = self.read_lane_boundaries_by_ids(boundary_ids)

        return MogoRawHDMap(
            lanes=lanes,
            lane_boundaries=lane_boundaries,
            lane_links=lane_links,
            roads=roads,
            junctions=junctions,
        )

    def read_lanes_by_road_ids(self, road_pkids: Sequence[int]) -> List[Dict[str, Any]]:
        """Read lane attributes and centerline geometry for requested road ids."""
        if not road_pkids:
            return []

        query = f"""
            SELECT
                rowid,
                road_pkid,
                section_id,
                lane_id,
                name,
                left_boundary,
                right_boundary,
                lane_type,
                lane_arrow,
                speed_limit,
                geom
            FROM TX_LANE
            WHERE road_pkid IN ({self._placeholders(road_pkids)})
        """
        rows = self._fetch_all(query, tuple(int(road_pkid) for road_pkid in road_pkids))
        return self._rows_to_lanes(rows)

    def read_lanes_by_rowids(self, rowids: Sequence[int]) -> List[Dict[str, Any]]:
        """Read lane records by SQLite rowid values from the RTree index."""
        if not rowids:
            return []

        query = f"""
            SELECT
                rowid,
                road_pkid,
                section_id,
                lane_id,
                name,
                left_boundary,
                right_boundary,
                lane_type,
                lane_arrow,
                speed_limit,
                geom
            FROM TX_LANE
            WHERE rowid IN ({self._placeholders(rowids)})
        """
        rows = self._fetch_all(query, tuple(int(rowid) for rowid in rowids))
        return self._rows_to_lanes(rows)

    def read_lane_boundaries_by_ids(self, boundary_ids: Sequence[int]) -> List[Dict[str, Any]]:
        """Read lane boundary geometry for the requested boundary ids."""
        if not boundary_ids:
            return []

        query = f"""
            SELECT
                pkid,
                lane_mark,
                geom
            FROM TX_LANE_BOUNDARY
            WHERE pkid IN ({self._placeholders(boundary_ids)})
        """
        rows = self._fetch_all(query, tuple(int(boundary_id) for boundary_id in boundary_ids))
        return [
            {
                "id": f"boundary_{row['pkid']}",
                "pkid": row["pkid"],
                "lane_mark": row["lane_mark"],
                "polyline": self._blob_to_utm_points(row["geom"]),
            }
            for row in rows
        ]

    def read_lane_links_by_ids(self, link_pkids: Sequence[int]) -> List[Dict[str, Any]]:
        """Read lane links by link pkids."""
        if not link_pkids:
            return []

        query = f"""
            SELECT
                pkid,
                from_road_pkid,
                from_section_id,
                from_lane_id,
                to_road_pkid,
                to_section_id,
                to_lane_id,
                link_type,
                geom
            FROM TX_LANE_LINK
            WHERE pkid IN ({self._placeholders(link_pkids)})
        """
        rows = self._fetch_all(query, tuple(int(link_pkid) for link_pkid in link_pkids))
        return self._rows_to_lane_links(rows)

    def read_lane_links_by_road_ids(self, road_pkids: Sequence[int]) -> List[Dict[str, Any]]:
        """Read lane links connected to any requested road id."""
        if not road_pkids:
            return []

        placeholders = self._placeholders(road_pkids)
        params = tuple(int(road_pkid) for road_pkid in road_pkids)
        query = f"""
            SELECT
                pkid,
                from_road_pkid,
                from_section_id,
                from_lane_id,
                to_road_pkid,
                to_section_id,
                to_lane_id,
                link_type,
                geom
            FROM TX_LANE_LINK
            WHERE from_road_pkid IN ({placeholders})
               OR to_road_pkid IN ({placeholders})
        """
        rows = self._fetch_all(query, params + params)
        return self._rows_to_lane_links(rows)

    def read_roads_by_ids(self, road_pkids: Sequence[int]) -> List[Dict[str, Any]]:
        """Read road metadata and center geometry for requested road ids."""
        if not road_pkids:
            return []

        query = f"""
            SELECT
                pkid,
                id AS external_id,
                name,
                junction_pkid,
                road_type,
                direction,
                geom
            FROM TX_ROAD
            WHERE pkid IN ({self._placeholders(road_pkids)})
        """
        rows = self._fetch_all(query, tuple(int(road_pkid) for road_pkid in road_pkids))
        return [
            {
                "id": _mogo_roadblock_id(row["pkid"]),
                "pkid": row["pkid"],
                "external_id": row["external_id"] or "",
                "name": row["name"] or "",
                "junction_pkid": row["junction_pkid"],
                "road_type": row["road_type"],
                "direction": row["direction"],
                "polyline": self._blob_to_utm_points(row["geom"]),
            }
            for row in rows
        ]

    def read_junctions_by_ids(self, junction_pkids: Sequence[int]) -> List[Dict[str, Any]]:
        """Read junction polygons for requested junction ids."""
        if not junction_pkids:
            return []

        query = f"""
            SELECT
                pkid,
                name,
                type,
                road_pkid,
                enter_road_pkid,
                inside_road_pkid,
                exit_road_pkid,
                geom
            FROM TX_JUNCTION
            WHERE pkid IN ({self._placeholders(junction_pkids)})
        """
        rows = self._fetch_all(query, tuple(int(junction_pkid) for junction_pkid in junction_pkids))
        return [
            {
                "id": _mogo_junction_id(row["pkid"]),
                "pkid": row["pkid"],
                "name": row["name"] or "",
                "type": row["type"],
                "road_pkid_raw": row["road_pkid"] or "",
                "enter_road_pkid_raw": row["enter_road_pkid"] or "",
                "inside_road_pkid_raw": row["inside_road_pkid"] or "",
                "exit_road_pkid_raw": row["exit_road_pkid"] or "",
                "road_pkids": self._parse_pkid_list(row["road_pkid"]),
                "enter_road_pkids": self._parse_pkid_list(row["enter_road_pkid"]),
                "inside_road_pkids": self._parse_pkid_list(row["inside_road_pkid"]),
                "exit_road_pkids": self._parse_pkid_list(row["exit_road_pkid"]),
                "polygon": self._blob_to_utm_points(row["geom"], close_polygon=True),
            }
            for row in rows
        ]

    def _rows_to_lanes(self, rows: Sequence[sqlite3.Row]) -> List[Dict[str, Any]]:
        lanes = []
        for row in rows:
            lanes.append(
                {
                    "id": _mogo_lane_id(row["road_pkid"], row["section_id"], row["lane_id"]),
                    "rowid": row["rowid"],
                    "road_pkid": row["road_pkid"],
                    "roadblock_id": _mogo_roadblock_id(row["road_pkid"]),
                    "section_id": row["section_id"],
                    "lane_id": row["lane_id"],
                    "name": row["name"] or "",
                    "left_boundary_id": row["left_boundary"],
                    "right_boundary_id": row["right_boundary"],
                    "lane_type": row["lane_type"],
                    "lane_arrow": row["lane_arrow"],
                    "speed_limit": row["speed_limit"],
                    "polyline": self._blob_to_utm_points(row["geom"]),
                }
            )
        return lanes

    def _rows_to_lane_links(self, rows: Sequence[sqlite3.Row]) -> List[Dict[str, Any]]:
        lane_links = []
        for row in rows:
            from_lane_id = _mogo_lane_id(row["from_road_pkid"], row["from_section_id"], row["from_lane_id"])
            to_lane_id = _mogo_lane_id(row["to_road_pkid"], row["to_section_id"], row["to_lane_id"])
            polyline = self._blob_to_utm_points(row["geom"])
            lane_links.append(
                {
                    "id": _mogo_link_id(row["pkid"]),
                    "pkid": row["pkid"],
                    "from_lane_id": from_lane_id,
                    "to_lane_id": to_lane_id,
                    "from_road_pkid": row["from_road_pkid"],
                    "to_road_pkid": row["to_road_pkid"],
                    "from_roadblock_id": _mogo_roadblock_id(row["from_road_pkid"]),
                    "to_roadblock_id": _mogo_roadblock_id(row["to_road_pkid"]),
                    "link_type": row["link_type"],
                    "polyline": polyline,
                    "is_empty_link": len(polyline) < 2,
                }
            )
        return lane_links

    @staticmethod
    def _read_table_names(connection: sqlite3.Connection) -> Set[str]:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        return {str(row[0]) for row in rows}

    @staticmethod
    def _read_table_columns(connection: sqlite3.Connection, table: str) -> Set[str]:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row[1]) for row in rows}

    @staticmethod
    def _read_row_count(connection: sqlite3.Connection, table: str) -> int:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0

    def _validate_geometry_blob(
        self,
        connection: sqlite3.Connection,
        table: str,
        expect_polygon: bool,
    ) -> Optional[int]:
        row = connection.execute(f"SELECT geom FROM {table} WHERE geom IS NOT NULL LIMIT 1").fetchone()
        if row is None:
            return None

        geometry_type, srid, points = self._decode_spatialite_blob(row["geom"])
        if not points:
            raise RuntimeError(f"Mogo sqlite geometry BLOB sample for {table} has no points")
        base_type = geometry_type % 1000
        if expect_polygon and base_type != 3:
            raise RuntimeError(f"Mogo sqlite expected polygon geometry in {table}, got type {geometry_type}")
        if not expect_polygon and base_type not in (1, 2, 5):
            raise RuntimeError(f"Mogo sqlite expected line geometry in {table}, got type {geometry_type}")
        return srid

    def _read_rtree_pkids(
        self,
        index_table: str,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> List[int]:
        if index_table not in self.inspection.rtree_tables:
            return []

        query = f"""
            SELECT pkid
            FROM {index_table}
            WHERE xmax >= ?
              AND xmin <= ?
              AND ymax >= ?
              AND ymin <= ?
        """
        rows = self._fetch_all(query, (lon_min, lon_max, lat_min, lat_max))
        return [int(row["pkid"]) for row in rows]

    def _fetch_all(self, query: str, params: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
        connection = self.connect()
        return list(connection.execute(query, params).fetchall())

    @staticmethod
    def _placeholders(values: Sequence[Any]) -> str:
        return ",".join("?" for _ in values)

    @staticmethod
    def _parse_pkid_list(value: Any) -> List[int]:
        """Parse comma-separated pkid strings used by TX_JUNCTION road binding fields."""
        if value in (None, ""):
            return []
        pkids: List[int] = []
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                pkids.append(int(item))
            except ValueError:
                logger.warning("Mogo junction road pkid parse failed: value=%r, item=%r", value, item)
        return pkids

    @staticmethod
    def _collect_boundary_ids(lanes: Sequence[Mapping[str, Any]]) -> List[int]:
        return sorted(
            {
                int(boundary_id)
                for lane in lanes
                for boundary_id in (lane.get("left_boundary_id"), lane.get("right_boundary_id"))
                if boundary_id not in (None, "")
            }
        )

    @staticmethod
    def _collect_junction_ids(roads: Sequence[Mapping[str, Any]]) -> List[int]:
        return sorted(
            {
                int(road["junction_pkid"])
                for road in roads
                if road.get("junction_pkid") not in (None, "", 0)
            }
        )

    def _blob_to_utm_points(self, geometry_blob: Any, close_polygon: bool = False) -> List[List[float]]:
        if geometry_blob is None:
            return []

        _, srid, lonlat_points = self._decode_spatialite_blob(geometry_blob)
        if srid != self._config.geometry_srid:
            raise RuntimeError(f"Unexpected geometry SRID {srid}; expected {self._config.geometry_srid}")

        utm_points = []
        for lon, lat, z in lonlat_points:
            x, y = self._wgs84_to_utm.transform(lon, lat)
            utm_points.append([float(x), float(y), float(z)])
        if close_polygon and utm_points and utm_points[0] != utm_points[-1]:
            utm_points.append(utm_points[0])
        return utm_points

    @staticmethod
    def _decode_spatialite_blob(geometry_blob: Any) -> Tuple[int, int, List[List[float]]]:
        """Decode the SpatiaLite-like geometry BLOB used by the Mogo SQLite map.

        The verified files store SRID + MBR + an internal WKB-like geometry body.
        We only need point/line/polygon families for first-version map reads.
        """
        data = bytes(geometry_blob)
        if len(data) < 43:
            raise RuntimeError("Geometry BLOB is too short")

        endian = "<" if data[1] == 1 else ">"
        srid = struct.unpack(endian + "I", data[2:6])[0]
        if data[38] != 0x7C:
            raise RuntimeError("Unsupported geometry BLOB header marker")

        offset = 39
        geometry_type, offset = MogoSQLiteReader._read_u32(data, offset, endian)
        base_type = geometry_type % 1000
        dim, has_z = MogoSQLiteReader._geometry_dimension(geometry_type)

        if base_type == 1:
            point, offset = MogoSQLiteReader._read_point(data, offset, endian, dim, has_z)
            points = [point]
        elif base_type == 2:
            points, offset = MogoSQLiteReader._read_line_string(data, offset, endian, dim, has_z)
        elif base_type == 3:
            points, offset = MogoSQLiteReader._read_polygon_exterior(data, offset, endian, dim, has_z)
        elif base_type == 5:
            points, offset = MogoSQLiteReader._read_multi_line_string(data, offset, endian)
        elif base_type == 6:
            points, offset = MogoSQLiteReader._read_multi_polygon(data, offset, endian)
        else:
            raise RuntimeError(f"Unsupported geometry type {geometry_type}")

        return geometry_type, srid, points

    @staticmethod
    def _geometry_dimension(geometry_type: int) -> Tuple[int, bool]:
        modifier = geometry_type // 1000
        has_z = modifier in (1, 3)
        has_m = modifier in (2, 3)
        return 2 + int(has_z) + int(has_m), has_z

    @staticmethod
    def _read_u32(data: bytes, offset: int, endian: str) -> Tuple[int, int]:
        return struct.unpack(endian + "I", data[offset : offset + 4])[0], offset + 4

    @staticmethod
    def _read_point(data: bytes, offset: int, endian: str, dim: int, has_z: bool) -> Tuple[List[float], int]:
        values = struct.unpack(endian + "d" * dim, data[offset : offset + 8 * dim])
        z = values[2] if has_z else 0.0
        return [float(values[0]), float(values[1]), float(z)], offset + 8 * dim

    @staticmethod
    def _read_line_string(data: bytes, offset: int, endian: str, dim: int, has_z: bool) -> Tuple[List[List[float]], int]:
        point_count, offset = MogoSQLiteReader._read_u32(data, offset, endian)
        points = []
        for _ in range(point_count):
            point, offset = MogoSQLiteReader._read_point(data, offset, endian, dim, has_z)
            points.append(point)
        return points, offset

    @staticmethod
    def _read_polygon_exterior(data: bytes, offset: int, endian: str, dim: int, has_z: bool) -> Tuple[List[List[float]], int]:
        ring_count, offset = MogoSQLiteReader._read_u32(data, offset, endian)
        exterior: List[List[float]] = []
        for ring_index in range(ring_count):
            ring, offset = MogoSQLiteReader._read_line_string(data, offset, endian, dim, has_z)
            if ring_index == 0:
                exterior = ring
        return exterior, offset

    @staticmethod
    def _read_child_geometry(data: bytes, offset: int, endian: str) -> Tuple[List[List[float]], int]:
        # SpatiaLite collections prefix each child with 0x69. Be permissive for
        # files that omit the marker and put the child geometry type directly.
        if offset < len(data) and data[offset] == 0x69:
            offset += 1
        child_type, offset = MogoSQLiteReader._read_u32(data, offset, endian)
        child_base_type = child_type % 1000
        dim, has_z = MogoSQLiteReader._geometry_dimension(child_type)
        if child_base_type == 2:
            return MogoSQLiteReader._read_line_string(data, offset, endian, dim, has_z)
        if child_base_type == 3:
            return MogoSQLiteReader._read_polygon_exterior(data, offset, endian, dim, has_z)
        raise RuntimeError(f"Unsupported child geometry type {child_type}")

    @staticmethod
    def _read_multi_line_string(data: bytes, offset: int, endian: str) -> Tuple[List[List[float]], int]:
        geometry_count, offset = MogoSQLiteReader._read_u32(data, offset, endian)
        merged: List[List[float]] = []
        for _ in range(geometry_count):
            points, offset = MogoSQLiteReader._read_child_geometry(data, offset, endian)
            if merged and points and merged[-1] == points[0]:
                merged.extend(points[1:])
            else:
                merged.extend(points)
        return merged, offset

    @staticmethod
    def _read_multi_polygon(data: bytes, offset: int, endian: str) -> Tuple[List[List[float]], int]:
        geometry_count, offset = MogoSQLiteReader._read_u32(data, offset, endian)
        largest: List[List[float]] = []
        largest_area = -1.0
        for _ in range(geometry_count):
            points, offset = MogoSQLiteReader._read_child_geometry(data, offset, endian)
            area = MogoSQLiteReader._signed_polygon_area(points)
            if abs(area) > largest_area:
                largest = points
                largest_area = abs(area)
        return largest, offset

    @staticmethod
    def _signed_polygon_area(points: Sequence[Sequence[float]]) -> float:
        if len(points) < 3:
            return 0.0
        area = 0.0
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            area += point[0] * next_point[1] - next_point[0] * point[1]
        return 0.5 * area

    def _utm_to_lonlat(self, x: float, y: float) -> Tuple[float, float]:
        lon, lat = self._utm_to_wgs84.transform(x, y)
        return float(lon), float(lat)

    @staticmethod
    def _lonlat_bbox(lon: float, lat: float, radius_m: float) -> Tuple[float, float, float, float]:
        lat_delta = radius_m / 111_320.0
        cos_lat = max(math.cos(math.radians(lat)), 1e-6)
        lon_delta = radius_m / (111_320.0 * cos_lat)
        return lon - lon_delta, lon + lon_delta, lat - lat_delta, lat + lat_delta


class MogoPolylineMapObject:
    """Small polyline object implementing the subset consumed by PDM."""

    def __init__(self, object_id: str, points: Sequence[Sequence[float]]):
        self.id = str(object_id)
        self._points = [list(point) for point in points]
        if len(self._points) == 1:
            x, y = self._points[0][:2]
            self._points = [[x, y, self._points[0][2]], [x + 1e-3, y, self._points[0][2]]]
        self._linestring = LineString([(point[0], point[1]) for point in self._points]) if self._points else LineString()
        self._discrete_path = self._build_discrete_path(self._points)

    @property
    def linestring(self) -> LineString:
        return self._linestring

    @property
    def length(self) -> float:
        return float(self._linestring.length)

    @property
    def discrete_path(self) -> List[StateSE2]:
        return self._discrete_path

    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        if self._linestring.is_empty:
            return 0.0
        return float(self._linestring.project(Point(point.x, point.y)))

    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        arc_length = self.get_nearest_arc_length_from_position(point)
        return self._pose_at_arc_length(arc_length)

    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        # First-version adapter only needs a stable baseline path for PDM graph search.
        return 0.0

    def _pose_at_arc_length(self, arc_length: float) -> StateSE2:
        if not self._points:
            return StateSE2(0.0, 0.0, 0.0)
        if len(self._points) == 1 or self._linestring.length <= 1e-6:
            point = self._points[0]
            return StateSE2(point[0], point[1], 0.0)

        arc_length = float(np.clip(arc_length, 0.0, self._linestring.length))
        point = self._linestring.interpolate(arc_length)
        delta = min(0.5, max(self._linestring.length * 0.01, 1e-3))
        prev_point = self._linestring.interpolate(max(0.0, arc_length - delta))
        next_point = self._linestring.interpolate(min(self._linestring.length, arc_length + delta))
        heading = math.atan2(next_point.y - prev_point.y, next_point.x - prev_point.x)
        return StateSE2(float(point.x), float(point.y), float(heading))

    @classmethod
    def _build_discrete_path(cls, points: Sequence[Sequence[float]]) -> List[StateSE2]:
        if not points:
            return []
        if len(points) == 1:
            return [StateSE2(float(points[0][0]), float(points[0][1]), 0.0)]

        states: List[StateSE2] = []
        for index, point in enumerate(points):
            if index < len(points) - 1:
                next_point = points[index + 1]
                heading = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
            else:
                prev_point = points[index - 1]
                heading = math.atan2(point[1] - prev_point[1], point[0] - prev_point[0])
            states.append(StateSE2(float(point[0]), float(point[1]), float(heading)))
        return states


class MogoPolygonMapObject:
    """Simple polygon object used for intersections and optional drivable layers."""

    def __init__(self, object_id: str, polygon: Polygon, metadata: Optional[Mapping[str, Any]] = None):
        self.id = str(object_id)
        self._polygon = _clean_polygon(polygon)
        self.metadata = dict(metadata or {})

    @property
    def polygon(self) -> Polygon:
        return self._polygon

    def contains_point(self, point: Point2D) -> bool:
        return bool(self._polygon.contains(Point(point.x, point.y)))


class MogoLaneMapObject:
    """Lane-like graph edge with a baseline path and polygon."""

    def __init__(
        self,
        object_id: str,
        roadblock_id: str,
        baseline_path: MogoPolylineMapObject,
        polygon: Polygon,
        left_boundary: Optional[MogoPolylineMapObject] = None,
        right_boundary: Optional[MogoPolylineMapObject] = None,
        speed_limit_mps: Optional[float] = None,
        index: int = 0,
        is_connector: bool = False,
    ):
        self.id = str(object_id)
        self._roadblock_id = str(roadblock_id)
        self._baseline_path = baseline_path
        self._polygon = _clean_polygon(polygon)
        self._left_boundary = left_boundary or baseline_path
        self._right_boundary = right_boundary or baseline_path
        self._speed_limit_mps = speed_limit_mps
        self._index = int(index)
        self._is_connector = bool(is_connector)
        self._parent: Optional[MogoRoadBlockMapObject] = None
        self._incoming_edges: List[MogoLaneMapObject] = []
        self._outgoing_edges: List[MogoLaneMapObject] = []
        self._parallel_edges: List[MogoLaneMapObject] = [self]

    @property
    def polygon(self) -> Polygon:
        return self._polygon

    @property
    def incoming_edges(self) -> List['MogoLaneMapObject']:
        return self._incoming_edges

    @property
    def outgoing_edges(self) -> List['MogoLaneMapObject']:
        return self._outgoing_edges

    @property
    def parallel_edges(self) -> List['MogoLaneMapObject']:
        return self._parallel_edges

    @property
    def baseline_path(self) -> MogoPolylineMapObject:
        return self._baseline_path

    @property
    def left_boundary(self) -> MogoPolylineMapObject:
        return self._left_boundary

    @property
    def right_boundary(self) -> MogoPolylineMapObject:
        return self._right_boundary

    @property
    def speed_limit_mps(self) -> Optional[float]:
        return self._speed_limit_mps

    @property
    def adjacent_edges(self) -> Tuple[Optional['MogoLaneMapObject'], Optional['MogoLaneMapObject']]:
        return None, None

    @property
    def stop_lines(self) -> List[Any]:
        return []

    def get_roadblock_id(self) -> str:
        return self._roadblock_id

    def parent(self) -> Optional['MogoRoadBlockMapObject']:
        return self._parent

    def has_traffic_lights(self) -> bool:
        return False

    def is_left_of(self, other: 'MogoLaneMapObject') -> bool:
        return self.index() < other.index()

    def is_right_of(self, other: 'MogoLaneMapObject') -> bool:
        return self.index() > other.index()

    def index(self) -> int:
        return self._index

    def get_width_left_right(self, point: Point2D, include_outside: bool = False) -> Tuple[float, float]:
        if not include_outside and not self.contains_point(point):
            return math.inf, math.inf
        query_point = Point(point.x, point.y)
        return (
            float(self.left_boundary.linestring.distance(query_point)),
            float(self.right_boundary.linestring.distance(query_point)),
        )

    def oriented_distance(self, point: Point2D) -> float:
        query_point = Point(point.x, point.y)
        distance = float(self.baseline_path.linestring.distance(query_point))
        arc = self.baseline_path.get_nearest_arc_length_from_position(point)
        pose = self.baseline_path._pose_at_arc_length(arc)
        cross = math.cos(pose.heading) * (point.y - pose.y) - math.sin(pose.heading) * (point.x - pose.x)
        return distance if cross >= 0.0 else -distance

    def contains_point(self, point: Point2D) -> bool:
        return bool(self.polygon.contains(Point(point.x, point.y)))


class MogoRoadBlockMapObject:
    """Roadblock-like graph edge containing lane-like interior edges."""

    def __init__(self, object_id: str, polygon: Polygon, is_connector: bool = False):
        self.id = str(object_id)
        self._polygon = _clean_polygon(polygon)
        self._is_connector = bool(is_connector)
        self._incoming_edges: List[MogoRoadBlockMapObject] = []
        self._outgoing_edges: List[MogoRoadBlockMapObject] = []
        self._parallel_edges: List[MogoRoadBlockMapObject] = [self]
        self._interior_edges: List[MogoLaneMapObject] = []

    @property
    def polygon(self) -> Polygon:
        return self._polygon

    @property
    def incoming_edges(self) -> List['MogoRoadBlockMapObject']:
        return self._incoming_edges

    @property
    def outgoing_edges(self) -> List['MogoRoadBlockMapObject']:
        return self._outgoing_edges

    @property
    def parallel_edges(self) -> List['MogoRoadBlockMapObject']:
        return self._parallel_edges

    @property
    def interior_edges(self) -> List[MogoLaneMapObject]:
        return self._interior_edges

    @property
    def children_stop_lines(self) -> List[Any]:
        return []

    def intersection(self) -> Optional[MogoPolygonMapObject]:
        return None

    def contains_point(self, point: Point2D) -> bool:
        return bool(self.polygon.contains(Point(point.x, point.y)))


def _semantic_layer(name: str) -> SemanticMapLayer:
    return getattr(SemanticMapLayer, name)


def _layer_name(layer: SemanticMapLayer) -> str:
    return getattr(layer, 'name', str(layer))


def _clean_polygon(polygon: Polygon) -> Polygon:
    if polygon.is_empty:
        return polygon
    if not polygon.is_valid:
        polygon = polygon.buffer(0.0)
    if polygon.geom_type == 'Polygon':
        return polygon
    if polygon.geom_type == 'MultiPolygon':
        return max(polygon.geoms, key=lambda item: item.area)
    return polygon.convex_hull


def _polyline_xy(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    return [(float(point[0]), float(point[1])) for point in points if len(point) >= 2]


def _safe_linestring(points: Sequence[Sequence[float]]) -> LineString:
    xy = _polyline_xy(points)
    if len(xy) == 1:
        xy.append((xy[0][0] + 1e-3, xy[0][1]))
    return LineString(xy) if xy else LineString()


def _make_lane_polygon(
    centerline: Sequence[Sequence[float]],
    left_boundary: Optional[Sequence[Sequence[float]]],
    right_boundary: Optional[Sequence[Sequence[float]]],
    fallback_half_width_m: float = 1.75,
) -> Polygon:
    left_xy = _polyline_xy(left_boundary or [])
    right_xy = _polyline_xy(right_boundary or [])
    if len(left_xy) >= 2 and len(right_xy) >= 2:
        polygon = Polygon(left_xy + list(reversed(right_xy)))
        if polygon.area > 1e-3:
            return _clean_polygon(polygon)

    line = _safe_linestring(centerline)
    if line.is_empty:
        return Polygon()
    return _clean_polygon(line.buffer(fallback_half_width_m, cap_style=2, join_style=2))


def _make_roadblock_polygon(
    roadblock_id: str,
    interior_edges: Sequence[MogoLaneMapObject],
    fallback_line: Sequence[Sequence[float]],
) -> Polygon:
    """Build ROADBLOCK polygon from the outer lane boundaries.

    The first-version Mogo roadblock surface follows the agreed semantics:
    use the left boundary of the left-most lane and the right boundary of the
    right-most lane. If that cannot form a valid polygon, fall back to the union
    of lane polygons, then to a buffered road centerline.
    """
    left_xy: List[Tuple[float, float]] = []
    right_xy: List[Tuple[float, float]] = []
    sorted_edges = sorted(interior_edges, key=lambda edge: edge.index())
    if sorted_edges:
        left_boundary = sorted_edges[0].left_boundary.linestring
        right_boundary = sorted_edges[-1].right_boundary.linestring
        left_xy = list(left_boundary.coords) if not left_boundary.is_empty else []
        right_xy = list(right_boundary.coords) if not right_boundary.is_empty else []
        if len(left_xy) >= 2 and len(right_xy) >= 2:
            polygon = _clean_polygon(Polygon(left_xy + list(reversed(right_xy))))
            if not polygon.is_empty and polygon.area > 1e-3:
                return polygon

    polygons = [edge.polygon for edge in interior_edges if not edge.polygon.is_empty]
    if polygons:
        logger.warning(
            "Mogo ROADBLOCK polygon fallback to lane polygon union: roadblock_id=%s, "
            "lane_count=%d, left_boundary_points=%d, right_boundary_points=%d",
            roadblock_id,
            len(interior_edges),
            len(left_xy),
            len(right_xy),
        )
        return _clean_polygon(unary_union(polygons))

    line = _safe_linestring(fallback_line)
    if not line.is_empty:
        logger.warning(
            "Mogo ROADBLOCK polygon fallback to road centerline buffer: roadblock_id=%s, "
            "lane_count=%d, buffer_width=2.0m",
            roadblock_id,
            len(interior_edges),
        )
        return _clean_polygon(line.buffer(2.0, cap_style=2, join_style=2))

    logger.warning("Mogo ROADBLOCK polygon empty: roadblock_id=%s, lane_count=%d", roadblock_id, len(interior_edges))
    return Polygon()


def _make_connector_polygon(
    centerline: Sequence[Sequence[float]],
    from_lane: MogoLaneMapObject,
    to_lane: MogoLaneMapObject,
    link_type: Any,
) -> Polygon:
    """Build a first-version connector polygon from link centerline and lane widths."""
    line = _safe_linestring(centerline)
    if line.is_empty:
        logger.warning(
            "Mogo LANE_CONNECTOR polygon empty because link centerline is invalid: "
            "from_lane=%s, to_lane=%s, link_type=%s, point_count=%d",
            from_lane.id,
            to_lane.id,
            link_type,
            len(centerline),
        )
        return Polygon()

    from_half_width = _estimate_half_width(from_lane)
    to_half_width = _estimate_half_width(to_lane)
    max_half_width = max(from_half_width, to_half_width)
    min_half_width = min(from_half_width, to_half_width)

    # Internal link type 28 is the U-turn-like type currently mapped to UTURN.
    if int(link_type or 0) == 28:
        left_width = min_half_width + 0.5
        right_width = max_half_width + 1.0
    else:
        left_width = max_half_width + 1.0
        right_width = max_half_width + 1.0

    polygon = _make_asymmetric_buffer_polygon(line, left_width, right_width)
    if not polygon.is_empty and polygon.area > 1e-3:
        return polygon

    logger.warning(
        "Mogo LANE_CONNECTOR polygon fallback to symmetric buffer: from_lane=%s, to_lane=%s, "
        "link_type=%s, left_width=%.2f, right_width=%.2f, line_length=%.2f",
        from_lane.id,
        to_lane.id,
        link_type,
        left_width,
        right_width,
        line.length,
    )
    fallback = _clean_polygon(line.buffer(max(left_width, right_width), cap_style=2, join_style=2))
    if fallback.is_empty or fallback.area <= 1e-3:
        logger.warning(
            "Mogo LANE_CONNECTOR polygon remains empty after fallback: from_lane=%s, to_lane=%s, link_type=%s",
            from_lane.id,
            to_lane.id,
            link_type,
        )
    return fallback


def _make_asymmetric_buffer_polygon(line: LineString, left_width: float, right_width: float) -> Polygon:
    try:
        left_offset = line.parallel_offset(left_width, 'left', join_style=2)
        right_offset = line.parallel_offset(right_width, 'right', join_style=2)
        left_coords = _extract_line_coords(left_offset)
        right_coords = _extract_line_coords(right_offset)
        if len(left_coords) >= 2 and len(right_coords) >= 2:
            return _clean_polygon(Polygon(left_coords + list(reversed(right_coords))))
    except Exception:
        return Polygon()
    return Polygon()


def _extract_line_coords(geometry: Any) -> List[Tuple[float, float]]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'LineString':
        return list(geometry.coords)
    if geometry.geom_type == 'MultiLineString':
        line = max(geometry.geoms, key=lambda item: item.length)
        return list(line.coords)
    return []


def _estimate_half_width(lane: MogoLaneMapObject) -> float:
    center = lane.baseline_path.linestring
    left = lane.left_boundary.linestring
    right = lane.right_boundary.linestring
    widths = []
    if not center.is_empty and not left.is_empty:
        widths.append(float(center.distance(left)))
    if not center.is_empty and not right.is_empty:
        widths.append(float(center.distance(right)))
    if widths:
        return max(0.5, min(max(widths), 4.0))
    if lane.polygon and not lane.polygon.is_empty and center.length > 1e-3:
        return max(0.5, min(lane.polygon.area / max(center.length, 1e-3) * 0.5, 4.0))
    return 1.75


def _speed_limit_to_mps(speed_limit: Any) -> Optional[float]:
    if speed_limit in (None, ''):
        return None
    value = float(speed_limit)
    if value <= 0.0:
        return None
    # Actual SQLite distribution is 15/20/30/.../120, so positive values are km/h.
    return value / 3.6


def _append_unique(items: List[Any], item: Any) -> None:
    if item is not None and item not in items:
        items.append(item)


class MogoSQLiteLayerBuilder:
    """Builds the minimal map object graph needed by NAVSIM/PDM."""

    def __init__(self, config: MogoSQLiteMapConfig):
        self._config = config
        self._reader = MogoSQLiteReader(config)

    def build_layers(
        self,
        ego_position: Optional[Point2D] = None,
        route_context: Optional[MogoRouteContext] = None,
    ) -> Dict[SemanticMapLayer, List[Any]]:
        """Build semantic map object layers around ego and/or route context."""
        self._reader.inspect_database()
        raw_maps: List[MogoRawHDMap] = []
        route_road_pkids: Set[int] = set()
        if route_context is not None:
            road_pkids = []
            for roadblock_id in route_context.roadblock_ids:
                road_pkid = _mogo_parse_roadblock_id(roadblock_id)
                if road_pkid is not None:
                    road_pkids.append(road_pkid)
            route_road_pkids = set(road_pkids)
            raw_maps.append(self._reader.read_route_subgraph(road_pkids))
        if ego_position is not None:
            raw_maps.append(self._reader.read_local_region(ego_position, self._config.query_radius_m))
        if not raw_maps:
            return self._empty_layers()

        raw_map = self._merge_raw_maps(raw_maps)
        return self._build_object_layers(raw_map, route_road_pkids)

    @staticmethod
    def _empty_layers() -> Dict[SemanticMapLayer, List[Any]]:
        return {
            _semantic_layer('LANE'): [],
            _semantic_layer('LANE_CONNECTOR'): [],
            _semantic_layer('ROADBLOCK'): [],
            _semantic_layer('ROADBLOCK_CONNECTOR'): [],
            _semantic_layer('INTERSECTION'): [],
            _semantic_layer('CARPARK_AREA'): [],
            _semantic_layer('BASELINE_PATHS'): [],
        }

    @staticmethod
    def _merge_raw_maps(raw_maps: Sequence[MogoRawHDMap]) -> MogoRawHDMap:
        lanes: Dict[str, Dict[str, Any]] = {}
        boundaries: Dict[int, Dict[str, Any]] = {}
        links: Dict[int, Dict[str, Any]] = {}
        roads: Dict[int, Dict[str, Any]] = {}
        junctions: Dict[int, Dict[str, Any]] = {}
        for raw_map in raw_maps:
            lanes.update({lane['id']: lane for lane in raw_map.lanes})
            boundaries.update({int(boundary['pkid']): boundary for boundary in raw_map.lane_boundaries})
            links.update({int(link['pkid']): link for link in raw_map.lane_links})
            roads.update({int(road['pkid']): road for road in raw_map.roads})
            junctions.update({int(junction['pkid']): junction for junction in raw_map.junctions})
        return MogoRawHDMap(
            lanes=list(lanes.values()),
            lane_boundaries=list(boundaries.values()),
            lane_links=list(links.values()),
            roads=list(roads.values()),
            junctions=list(junctions.values()),
        )

    def _build_object_layers(self, raw_map: MogoRawHDMap, route_road_pkids: Set[int]) -> Dict[SemanticMapLayer, List[Any]]:
        boundary_by_pkid = {int(boundary['pkid']): boundary for boundary in raw_map.lane_boundaries}
        road_by_pkid = {int(road['pkid']): road for road in raw_map.roads}

        lane_objects: Dict[str, MogoLaneMapObject] = {}
        lanes_by_roadblock: DefaultDict[str, List[MogoLaneMapObject]] = defaultdict(list)
        for raw_lane in sorted(raw_map.lanes, key=lambda lane: (int(lane['road_pkid']), int(lane['section_id']), int(lane['lane_id']))):
            left_boundary = boundary_by_pkid.get(int(raw_lane['left_boundary_id'])) if raw_lane.get('left_boundary_id') else None
            right_boundary = boundary_by_pkid.get(int(raw_lane['right_boundary_id'])) if raw_lane.get('right_boundary_id') else None
            left_path = MogoPolylineMapObject(f"{raw_lane['id']}_left_boundary", left_boundary['polyline']) if left_boundary else None
            right_path = MogoPolylineMapObject(f"{raw_lane['id']}_right_boundary", right_boundary['polyline']) if right_boundary else None
            baseline_path = MogoPolylineMapObject(f"{raw_lane['id']}_baseline", raw_lane['polyline'])
            polygon = _make_lane_polygon(
                raw_lane['polyline'],
                left_boundary['polyline'] if left_boundary else None,
                right_boundary['polyline'] if right_boundary else None,
            )
            lane = MogoLaneMapObject(
                object_id=raw_lane['id'],
                roadblock_id=raw_lane['roadblock_id'],
                baseline_path=baseline_path,
                polygon=polygon,
                left_boundary=left_path,
                right_boundary=right_path,
                speed_limit_mps=_speed_limit_to_mps(raw_lane.get('speed_limit')),
                index=abs(int(raw_lane.get('lane_id', 0))),
            )
            lane_objects[lane.id] = lane
            lanes_by_roadblock[raw_lane['roadblock_id']].append(lane)

        roadblock_objects: Dict[str, MogoRoadBlockMapObject] = {}
        for roadblock_id, interior_lanes in lanes_by_roadblock.items():
            road_pkid = _mogo_parse_roadblock_id(roadblock_id)
            fallback_line = road_by_pkid.get(int(road_pkid), {}).get('polyline', []) if road_pkid is not None else []
            sorted_lanes = sorted(interior_lanes, key=lambda lane: lane.index())
            roadblock = MogoRoadBlockMapObject(
                object_id=roadblock_id,
                polygon=_make_roadblock_polygon(roadblock_id, sorted_lanes, fallback_line),
            )
            roadblock._interior_edges = sorted_lanes
            for lane in sorted_lanes:
                lane._parent = roadblock
                lane._parallel_edges = sorted_lanes
            roadblock_objects[roadblock_id] = roadblock

        connector_objects: Dict[str, MogoLaneMapObject] = {}
        connector_blocks: Dict[str, MogoRoadBlockMapObject] = {}
        connectors_by_block: DefaultDict[str, List[MogoLaneMapObject]] = defaultdict(list)
        missing_link_lane_count = 0
        missing_link_lane_examples: List[str] = []
        empty_topology_link_count = 0
        skipped_non_route_link_count = 0
        empty_connector_polygon_count = 0

        for raw_link in raw_map.lane_links:
            from_lane = lane_objects.get(raw_link['from_lane_id'])
            to_lane = lane_objects.get(raw_link['to_lane_id'])
            if from_lane is None or to_lane is None:
                missing_link_lane_count += 1
                if len(missing_link_lane_examples) < 5:
                    missing_link_lane_examples.append(
                        f"{raw_link.get('id')}:{raw_link.get('from_lane_id')}->{raw_link.get('to_lane_id')}"
                    )
                continue

            from_roadblock = roadblock_objects.get(raw_link['from_roadblock_id'])
            to_roadblock = roadblock_objects.get(raw_link['to_roadblock_id'])
            if from_roadblock is not None and to_roadblock is not None:
                # Keep direct roadblock connectivity so route correction preserves the
                # original roadblock sequence; connector blocks remain available as a
                # separate layer for drivable area and lane-level graph search.
                _append_unique(from_roadblock._outgoing_edges, to_roadblock)
                _append_unique(to_roadblock._incoming_edges, from_roadblock)

            if raw_link.get('is_empty_link') or len(raw_link.get('polyline', [])) < 2:
                empty_topology_link_count += 1
                _append_unique(from_lane._outgoing_edges, to_lane)
                _append_unique(to_lane._incoming_edges, from_lane)
                continue

            link_connects_route_roads = (
                int(raw_link['from_road_pkid']) in route_road_pkids
                and int(raw_link['to_road_pkid']) in route_road_pkids
            )
            if not link_connects_route_roads:
                skipped_non_route_link_count += 1
                continue

            connector_block_id = _mogo_roadblock_connector_id(raw_link['from_roadblock_id'], raw_link['to_roadblock_id'])
            connector_path = MogoPolylineMapObject(f"{raw_link['id']}_baseline", raw_link['polyline'])
            connector_polygon = _make_connector_polygon(
                raw_link['polyline'],
                from_lane,
                to_lane,
                raw_link.get('link_type'),
            )
            if connector_polygon.is_empty or connector_polygon.area <= 1e-3:
                empty_connector_polygon_count += 1
            connector = MogoLaneMapObject(
                object_id=raw_link['id'],
                roadblock_id=connector_block_id,
                baseline_path=connector_path,
                polygon=connector_polygon,
                speed_limit_mps=from_lane.speed_limit_mps,
                is_connector=True,
            )
            connector._incoming_edges = [from_lane]
            connector._outgoing_edges = [to_lane]
            connector_objects[connector.id] = connector
            connectors_by_block[connector_block_id].append(connector)
            _append_unique(from_lane._outgoing_edges, connector)
            _append_unique(to_lane._incoming_edges, connector)

            if connector_block_id not in connector_blocks:
                connector_blocks[connector_block_id] = MogoRoadBlockMapObject(
                    object_id=connector_block_id,
                    polygon=connector_polygon,
                    is_connector=True,
                )
            connector_block = connector_blocks[connector_block_id]
            connector._parent = connector_block
            _append_unique(connector_block._incoming_edges, from_roadblock)
            _append_unique(connector_block._outgoing_edges, to_roadblock)
            if from_roadblock is not None:
                _append_unique(from_roadblock._outgoing_edges, connector_block)
            if to_roadblock is not None:
                _append_unique(to_roadblock._incoming_edges, connector_block)

        empty_connector_block_count = 0
        for connector_block_id, connectors in connectors_by_block.items():
            connector_block = connector_blocks[connector_block_id]
            connector_block._interior_edges = sorted(connectors, key=lambda connector: connector.id)
            polygons = [connector.polygon for connector in connectors if not connector.polygon.is_empty]
            if polygons:
                connector_block._polygon = _clean_polygon(unary_union(polygons))
                if connector_block._polygon.is_empty or connector_block._polygon.area <= 1e-3:
                    empty_connector_block_count += 1
                    logger.warning(
                        "Mogo ROADBLOCK_CONNECTOR polygon empty after union: connector_block_id=%s, connector_count=%d",
                        connector_block_id,
                        len(connectors),
                    )
            else:
                empty_connector_block_count += 1
                connector_block._polygon = Polygon()
                logger.warning(
                    "Mogo ROADBLOCK_CONNECTOR polygon empty because all connector polygons are empty: "
                    "connector_block_id=%s, connector_count=%d",
                    connector_block_id,
                    len(connectors),
                )
            for connector in connector_block._interior_edges:
                connector._parallel_edges = connector_block._interior_edges

        intersections = [
            MogoPolygonMapObject(
                junction['id'],
                Polygon(_polyline_xy(junction['polygon'])),
                metadata={
                    'pkid': junction.get('pkid'),
                    'road_pkids': junction.get('road_pkids', []),
                    'enter_road_pkids': junction.get('enter_road_pkids', []),
                    'inside_road_pkids': junction.get('inside_road_pkids', []),
                    'exit_road_pkids': junction.get('exit_road_pkids', []),
                    'road_pkid_raw': junction.get('road_pkid_raw', ''),
                    'enter_road_pkid_raw': junction.get('enter_road_pkid_raw', ''),
                    'inside_road_pkid_raw': junction.get('inside_road_pkid_raw', ''),
                    'exit_road_pkid_raw': junction.get('exit_road_pkid_raw', ''),
                },
            )
            for junction in raw_map.junctions
            if len(junction.get('polygon', [])) >= 3
        ]

        layers = self._empty_layers()
        layers[_semantic_layer('LANE')] = list(lane_objects.values())
        layers[_semantic_layer('LANE_CONNECTOR')] = list(connector_objects.values())
        layers[_semantic_layer('ROADBLOCK')] = list(roadblock_objects.values())
        layers[_semantic_layer('ROADBLOCK_CONNECTOR')] = list(connector_blocks.values())
        layers[_semantic_layer('INTERSECTION')] = intersections
        layers[_semantic_layer('BASELINE_PATHS')] = [lane.baseline_path for lane in lane_objects.values()] + [
            connector.baseline_path for connector in connector_objects.values()
        ]

        if missing_link_lane_count:
            logger.warning(
                "Mogo lane link references missing lanes: count=%d, examples=%s",
                missing_link_lane_count,
                missing_link_lane_examples,
            )
        logger.info(
            "Mogo map layer build summary: route_road_count=%d, raw_lanes=%d, raw_links=%d, "
            "lanes=%d, roadblocks=%d, lane_connectors=%d, roadblock_connectors=%d, "
            "empty_topology_links=%d, skipped_non_route_links=%d, empty_connector_polygons=%d, "
            "empty_connector_blocks=%d, intersections=%d",
            len(route_road_pkids),
            len(raw_map.lanes),
            len(raw_map.lane_links),
            len(lane_objects),
            len(roadblock_objects),
            len(connector_objects),
            len(connector_blocks),
            empty_topology_link_count,
            skipped_non_route_link_count,
            empty_connector_polygon_count,
            empty_connector_block_count,
            len(intersections),
        )
        return layers


class MogoSQLiteMapAdapter(AbstractMap):
    """SQLite-backed map API adapter for Mogo internal data."""

    def __init__(
        self,
        config: MogoSQLiteMapConfig,
        route_context: Optional[MogoRouteContext] = None,
    ):
        self._config = config
        self._route_context = route_context
        self._layer_builder = MogoSQLiteLayerBuilder(config)
        self._layers: Dict[SemanticMapLayer, List[Any]] = {}
        self._objects_by_layer: Dict[SemanticMapLayer, Dict[str, Any]] = {}
        self._last_loaded_point: Optional[Tuple[float, float]] = None

    @property
    def map_name(self) -> str:
        """Return the adapter map name."""
        return self._config.map_name

    def load(self, ego_position: Optional[Point2D] = None) -> None:
        """Load or refresh semantic layers around the requested ego position."""
        self._layers = self._layer_builder.build_layers(
            ego_position=ego_position,
            route_context=self._route_context,
        )
        self._objects_by_layer = self._wrap_layers(self._layers)
        self._last_loaded_point = None if ego_position is None else (float(ego_position.x), float(ego_position.y))

    def initialize_all_layers(self) -> None:
        self.load()

    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        return [layer for layer, objects in self._objects_by_layer.items() if objects]

    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        return []

    def get_raster_map_layer(self, layer: SemanticMapLayer) -> Any:
        raise NotImplementedError("Mogo SQLite adapter does not provide raster layers")

    def get_raster_map(self, layers: List[SemanticMapLayer]) -> Any:
        raise NotImplementedError("Mogo SQLite adapter does not provide raster layers")

    def get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[Any]:
        self._ensure_loaded(point)
        query = Point(point.x, point.y)
        return [obj for obj in self._objects_by_layer.get(layer, {}).values() if obj.polygon.contains(query)]

    def get_one_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Optional[Any]:
        objects = self.get_all_map_objects(point, layer)
        if not objects:
            return None
        if len(objects) > 1:
            raise AssertionError(f"More than one map object found for layer {layer}: {[obj.id for obj in objects]}")
        return objects[0]

    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        return self.get_one_map_object(point, layer) is not None

    def get_map_object(self, object_id: str, layer: SemanticMapLayer) -> Optional[Any]:
        if not self._objects_by_layer:
            self.load()
        return self._objects_by_layer.get(layer, {}).get(str(object_id))

    def get_proximal_map_objects(
        self,
        point: Point2D,
        radius: float,
        layers: Iterable[SemanticMapLayer],
    ) -> Dict[SemanticMapLayer, List[Any]]:
        self._ensure_loaded(point)
        query_region = Point(point.x, point.y).buffer(float(radius))
        result: Dict[SemanticMapLayer, List[Any]] = {}
        for layer in layers:
            objects = self._objects_by_layer.get(layer, {}).values()
            result[layer] = [obj for obj in objects if obj.polygon.intersects(query_region)]
        return result

    def get_distance_to_nearest_map_object(
        self,
        point: Point2D,
        layer: SemanticMapLayer,
    ) -> Tuple[Optional[str], float]:
        self._ensure_loaded(point)
        query_point = Point(point.x, point.y)
        nearest_id: Optional[str] = None
        nearest_distance = math.inf
        for obj in self._objects_by_layer.get(layer, {}).values():
            distance = float(obj.polygon.distance(query_point))
            if distance < nearest_distance:
                nearest_id = obj.id
                nearest_distance = distance
        if nearest_id is None:
            return None, float('nan')
        return nearest_id, nearest_distance

    def get_distance_to_nearest_raster_layer(self, point: Point2D, layer: SemanticMapLayer) -> float:
        return float('nan')

    def get_distances_matrix_to_nearest_map_object(
        self,
        points: List[Point2D],
        layer: SemanticMapLayer,
    ) -> Optional[np.ndarray]:
        distances = []
        for point in points:
            _, distance = self.get_distance_to_nearest_map_object(point, layer)
            distances.append(distance)
        return np.asarray(distances, dtype=np.float64)

    def _get_roadblock(self, object_id: str) -> Optional[Any]:
        return self.get_map_object(object_id, _semantic_layer('ROADBLOCK'))

    def _get_roadblock_connector(self, object_id: str) -> Optional[Any]:
        return self.get_map_object(object_id, _semantic_layer('ROADBLOCK_CONNECTOR'))

    def _ensure_loaded(self, point: Optional[Point2D] = None) -> None:
        if point is None:
            if not self._objects_by_layer:
                self.load()
            return
        point_key = (float(point.x), float(point.y))
        if not self._objects_by_layer or self._last_loaded_point != point_key:
            self.load(point)

    @staticmethod
    def _wrap_layers(layers: Mapping[SemanticMapLayer, Sequence[Any]]) -> Dict[SemanticMapLayer, Dict[str, Any]]:
        return {layer: {obj.id: obj for obj in objects} for layer, objects in layers.items()}


def build_mogo_sqlite_map_api(
    sqlite_path: Path,
    route_context: Optional[MogoRouteContext] = None,
    map_name: str = "mogo",
) -> MogoSQLiteMapAdapter:
    """Factory for creating a Mogo SQLite map API adapter."""
    config = MogoSQLiteMapConfig(sqlite_path=sqlite_path, map_name=map_name)
    return MogoSQLiteMapAdapter(config=config, route_context=route_context)
