"""Framework-neutral public contracts for single-instance refinement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Int16Array = NDArray[np.int16]
UInt64Array = NDArray[np.uint64]


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_tuple(
    value: tuple[float, ...], *, length: int, name: str
) -> tuple[float, ...]:
    if len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _unit_quaternion(
    value: tuple[float, float, float, float], name: str
) -> tuple[float, float, float, float]:
    result = _finite_tuple(value, length=4, name=name)
    if not np.isclose(float(np.linalg.norm(result)), 1.0, atol=1e-5):
        raise ValueError(f"{name} must have unit norm")
    return result


def _freeze_json(value: object, name: str) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{name} must contain only finite numbers")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{name}[{index}]") for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} object keys must be non-empty strings")
            frozen[key] = _freeze_json(item, f"{name}.{key}")
        return MappingProxyType(frozen)
    raise ValueError(f"{name} must contain JSON-compatible values")


@dataclass(frozen=True, slots=True)
class Pose3D:
    """A local-to-parent rigid pose, quaternion ordered as XYZW."""

    translation_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "translation_xyz",
            _finite_tuple(self.translation_xyz, length=3, name="translation_xyz"),
        )
        object.__setattr__(
            self,
            "orientation_xyzw",
            _unit_quaternion(self.orientation_xyzw, "orientation_xyzw"),
        )

    @classmethod
    def identity(cls) -> Pose3D:
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class Box3D:
    """An upright-compatible box whose pose is local-box to containing frame."""

    center: tuple[float, float, float]
    size_lwh: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        center = _finite_tuple(self.center, length=3, name="center")
        size = _finite_tuple(self.size_lwh, length=3, name="size_lwh")
        if any(component <= 0 for component in size):
            raise ValueError("size_lwh components must be positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size_lwh", size)
        object.__setattr__(
            self,
            "orientation_xyzw",
            _unit_quaternion(self.orientation_xyzw, "orientation_xyzw"),
        )

    @property
    def pose(self) -> Pose3D:
        return Pose3D(self.center, self.orientation_xyzw)


@dataclass(frozen=True, slots=True)
class FrameCloud:
    """One immutable full-scene point cloud in its annotation frame."""

    frame_id: str
    timestamp_ns: int
    annotation_frame_id: str
    world_from_annotation: Pose3D
    points_xyz: Float32Array
    point_features: Float32Array | None = None
    feature_names: tuple[str, ...] = ()
    point_timestamps_ns: UInt64Array | None = None
    point_sensor_index: Int16Array | None = None
    sensor_ids: tuple[str, ...] = ()
    sensor_origins: Mapping[str, tuple[float, float, float]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _nonempty(self.frame_id, "frame_id")
        _nonempty(self.annotation_frame_id, "annotation_frame_id")
        if (
            not isinstance(self.timestamp_ns, int)
            or isinstance(self.timestamp_ns, bool)
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer")

        points = np.asarray(self.points_xyz)
        if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_xyz must be float32 with shape [N, 3]")
        if not np.isfinite(points).all():
            raise ValueError("points_xyz must contain only finite values")

        features = self.point_features
        if features is not None:
            features = np.asarray(features)
            if features.dtype != np.float32 or features.ndim != 2:
                raise ValueError("point_features must be float32 with shape [N, F]")
            if features.shape[0] != len(points):
                raise ValueError("point_features must align with points_xyz")
            if features.shape[1] != len(self.feature_names):
                raise ValueError("feature_names must describe every feature column")
            if not np.isfinite(features).all():
                raise ValueError("point_features must contain only finite values")
        elif self.feature_names:
            raise ValueError("feature_names require point_features")
        if len(self.feature_names) != len(set(self.feature_names)) or any(
            not name for name in self.feature_names
        ):
            raise ValueError("feature_names must be unique non-empty strings")

        point_times = self.point_timestamps_ns
        if point_times is not None:
            point_times = np.asarray(point_times)
            if point_times.dtype != np.uint64 or point_times.shape != (len(points),):
                raise ValueError("point_timestamps_ns must be uint64 with shape [N]")

        sensor_index = self.point_sensor_index
        if sensor_index is not None:
            sensor_index = np.asarray(sensor_index)
            if sensor_index.dtype != np.int16 or sensor_index.shape != (len(points),):
                raise ValueError("point_sensor_index must be int16 with shape [N]")
            if (sensor_index < 0).any():
                raise ValueError("point_sensor_index must be non-negative")
            if not self.sensor_ids and len(sensor_index):
                raise ValueError("point_sensor_index requires sensor_ids")
            if len(sensor_index) and int(sensor_index.max()) >= len(self.sensor_ids):
                raise ValueError("point_sensor_index references an unknown sensor")
        if len(self.sensor_ids) != len(set(self.sensor_ids)) or any(
            not sensor_id for sensor_id in self.sensor_ids
        ):
            raise ValueError("sensor_ids must be unique non-empty strings")

        origins: dict[str, tuple[float, float, float]] = {}
        for sensor_id, origin in self.sensor_origins.items():
            if sensor_id not in self.sensor_ids:
                raise ValueError("sensor_origins references an unknown sensor")
            origins[sensor_id] = _finite_tuple(
                origin, length=3, name=f"sensor_origins[{sensor_id}]"
            )

        points.setflags(write=False)
        if features is not None:
            features.setflags(write=False)
        if point_times is not None:
            point_times.setflags(write=False)
        if sensor_index is not None:
            sensor_index.setflags(write=False)
        object.__setattr__(self, "points_xyz", points)
        object.__setattr__(self, "point_features", features)
        object.__setattr__(self, "point_timestamps_ns", point_times)
        object.__setattr__(self, "point_sensor_index", sensor_index)
        object.__setattr__(self, "sensor_origins", MappingProxyType(origins))


class ObservationKind(str, Enum):
    OBSERVED = "observed"
    INTERPOLATED = "interpolated"


@dataclass(frozen=True, slots=True)
class TrackObservation:
    frame_id: str
    coarse_box: Box3D
    score: float | None = None
    kind: ObservationKind = ObservationKind.OBSERVED

    def __post_init__(self) -> None:
        _nonempty(self.frame_id, "frame_id")
        if self.score is not None:
            score = float(self.score)
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("score must be finite and in [0, 1]")
            object.__setattr__(self, "score", score)
        object.__setattr__(self, "kind", ObservationKind(self.kind))


@dataclass(frozen=True, slots=True)
class InstanceTrack:
    track_id: str
    sequence_id: str
    observations: tuple[TrackObservation, ...]
    category: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.track_id, "track_id")
        _nonempty(self.sequence_id, "sequence_id")
        if self.category is not None:
            _nonempty(self.category, "category")
        if len(self.observations) < 2:
            raise ValueError("an instance track must contain at least two observations")
        frame_ids = [observation.frame_id for observation in self.observations]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("a track may contain at most one observation per frame")


@dataclass(frozen=True, slots=True)
class RefinementCase:
    """One single-instance call with full clouds for its observed frames."""

    case_id: str
    frames: tuple[FrameCloud, ...]
    track: InstanceTrack

    def __post_init__(self) -> None:
        _nonempty(self.case_id, "case_id")
        frame_ids = [frame.frame_id for frame in self.frames]
        timestamps = [frame.timestamp_ns for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("frame IDs must be unique")
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise ValueError("frames must have strictly increasing timestamps")
        observation_ids = [item.frame_id for item in self.track.observations]
        if frame_ids != observation_ids:
            raise ValueError("frames must exactly match track observations in order")


class RefinedFrameRole(str, Enum):
    """How an authoritative frame contributed to a refinement result."""

    GEOMETRY = "geometry"
    POSE_ONLY = "pose_only"


@dataclass(frozen=True, slots=True)
class RefinedFramePose:
    frame_id: str
    pose: Pose3D
    role: RefinedFrameRole = RefinedFrameRole.GEOMETRY

    def __post_init__(self) -> None:
        _nonempty(self.frame_id, "frame_id")
        object.__setattr__(self, "role", RefinedFrameRole(self.role))


@dataclass(frozen=True, slots=True)
class UnsupportedFrame:
    """An input frame for which TrackRefinery claims no refined pose."""

    frame_id: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.frame_id, "frame_id")
        if not self.reasons or any(
            not isinstance(reason, str) or not reason.strip() for reason in self.reasons
        ):
            raise ValueError("unsupported frame requires non-empty reasons")


@dataclass(frozen=True, slots=True)
class RefinementSuccess:
    track_id: str
    canonical_size_lwh: tuple[float, float, float]
    frame_poses: tuple[RefinedFramePose, ...]
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.track_id, "track_id")
        size = _finite_tuple(
            self.canonical_size_lwh,
            length=3,
            name="canonical_size_lwh",
        )
        if any(component <= 0 for component in size):
            raise ValueError("canonical_size_lwh components must be positive")
        frame_ids = [item.frame_id for item in self.frame_poses]
        if not frame_ids or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("frame_poses must contain unique frames")
        object.__setattr__(self, "canonical_size_lwh", size)
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_json(self.diagnostics, "diagnostics"),
        )

    @property
    def status(self) -> str:
        return "success"


@dataclass(frozen=True, slots=True)
class PartialRefinementSuccess:
    """A canonical result with authority over only a supported frame subset."""

    track_id: str
    canonical_size_lwh: tuple[float, float, float]
    frame_poses: tuple[RefinedFramePose, ...]
    unsupported_frames: tuple[UnsupportedFrame, ...]
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.track_id, "track_id")
        size = _finite_tuple(
            self.canonical_size_lwh,
            length=3,
            name="canonical_size_lwh",
        )
        if any(component <= 0 for component in size):
            raise ValueError("canonical_size_lwh components must be positive")
        frame_ids = [item.frame_id for item in self.frame_poses]
        unsupported_ids = [item.frame_id for item in self.unsupported_frames]
        if not frame_ids or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("frame_poses must contain unique frames")
        if not unsupported_ids or len(unsupported_ids) != len(set(unsupported_ids)):
            raise ValueError("unsupported_frames must contain unique frames")
        if set(frame_ids) & set(unsupported_ids):
            raise ValueError("refined and unsupported frames must be disjoint")
        object.__setattr__(self, "canonical_size_lwh", size)
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_json(self.diagnostics, "diagnostics"),
        )

    @property
    def status(self) -> str:
        return "partial_success"


@dataclass(frozen=True, slots=True)
class InsufficientEvidence:
    track_id: str
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.track_id, "track_id")
        if not self.reasons or any(not reason for reason in self.reasons):
            raise ValueError("insufficient evidence requires non-empty reasons")
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_json(self.diagnostics, "diagnostics"),
        )

    @property
    def status(self) -> str:
        return "insufficient_evidence"


RefinementOutcome = RefinementSuccess | PartialRefinementSuccess | InsufficientEvidence
