"""Experimental Stage 4 observable canonical-cuboid estimation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from math import ceil, degrees
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trackrefinery.contracts import Pose3D, RefinementCase
from trackrefinery.geometric.trace import (
    CanonicalShapeTrace,
    CuboidFitTrace,
    EvidenceState,
    FrameEvidenceTrace,
    FrameRegistrationTrace,
    FrameRole,
    GeometricRefinementTrace,
    validate_geometric_trace,
)
from trackrefinery.geometry import (
    angle_difference,
    compose_pose,
    inverse_pose,
    inverse_transform_points,
    yaw_from_quaternion,
)

CANONICAL_CUBOID_EXPERIMENT_CONTRACT = (
    "trackrefinery-stage4-observable-canonical-cuboid-experiment-v1"
)
CANONICAL_CUBOID_STAGE = "observable_canonical_cuboid_v4_experiment"
REQUIRED_STAGE3_STAGE = "pose_graph_aggregation_v3_experiment"


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _fraction(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class CanonicalCuboidExperimentSettings:
    """Versioned observability and stability policy for the Stage 4 experiment."""

    minimum_geometry_frames: int = 5
    canonical_voxel_size_m: float = 0.08
    normal_neighbor_count: int = 16
    normal_maximum_surface_variation: float = 0.12
    normal_maximum_absolute_z: float = 0.45
    normal_minimum_count: int = 64
    normal_minimum_fourfold_coherence: float = 0.80
    maximum_common_yaw_correction_deg: float = 4.0
    boundary_minimum_tail_points: int = 5
    boundary_tail_fraction: float = 0.001
    boundary_maximum_tail_fraction: float = 0.02
    boundary_minimum_frames: int = 3
    boundary_minimum_frame_fraction: float = 0.20
    face_band_m: float = 0.12
    face_normal_neighbor_count: int = 12
    face_maximum_surface_variation: float = 0.18
    face_minimum_absolute_normal_alignment: float = 0.70
    face_minimum_points_per_frame: int = 4
    face_minimum_tangential_span_m: float = 0.25
    ground_maximum_mad_m: float = 0.06
    leave_one_out_maximum_dimension_change_m: float = 0.08
    leave_one_out_maximum_center_change_m: float = 0.08
    leave_one_out_maximum_yaw_change_deg: float = 0.50
    resolution_scales: tuple[float, ...] = (0.75, 1.0, 1.25)
    resolution_maximum_dimension_change_m: float = 0.06
    resolution_maximum_yaw_change_deg: float = 1.0
    maximum_center_xy_correction_m: float = 0.50
    maximum_center_z_correction_m: float = 0.50

    def __post_init__(self) -> None:
        for name, minimum in (
            ("minimum_geometry_frames", 3),
            ("normal_neighbor_count", 4),
            ("face_normal_neighbor_count", 4),
            ("normal_minimum_count", 4),
            ("boundary_minimum_tail_points", 1),
            ("boundary_minimum_frames", 2),
            ("face_minimum_points_per_frame", 1),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        for name in (
            "canonical_voxel_size_m",
            "maximum_common_yaw_correction_deg",
            "face_band_m",
            "face_minimum_tangential_span_m",
            "ground_maximum_mad_m",
            "leave_one_out_maximum_dimension_change_m",
            "leave_one_out_maximum_center_change_m",
            "leave_one_out_maximum_yaw_change_deg",
            "resolution_maximum_dimension_change_m",
            "resolution_maximum_yaw_change_deg",
            "maximum_center_xy_correction_m",
            "maximum_center_z_correction_m",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in (
            "normal_maximum_surface_variation",
            "normal_maximum_absolute_z",
            "normal_minimum_fourfold_coherence",
            "face_maximum_surface_variation",
            "face_minimum_absolute_normal_alignment",
            "boundary_tail_fraction",
            "boundary_maximum_tail_fraction",
            "boundary_minimum_frame_fraction",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        if self.normal_maximum_surface_variation == 0:
            raise ValueError("normal_maximum_surface_variation must be positive")
        if self.normal_maximum_absolute_z == 0:
            raise ValueError("normal_maximum_absolute_z must be positive")
        if self.normal_minimum_fourfold_coherence == 0:
            raise ValueError("normal_minimum_fourfold_coherence must be positive")
        if self.face_maximum_surface_variation == 0:
            raise ValueError("face_maximum_surface_variation must be positive")
        if self.face_minimum_absolute_normal_alignment == 0:
            raise ValueError("face_minimum_absolute_normal_alignment must be positive")
        if self.boundary_tail_fraction == 0:
            raise ValueError("boundary_tail_fraction must be positive")
        if self.boundary_maximum_tail_fraction == 0:
            raise ValueError("boundary_maximum_tail_fraction must be positive")
        if self.boundary_tail_fraction > self.boundary_maximum_tail_fraction:
            raise ValueError(
                "boundary_tail_fraction cannot exceed its maximum fraction"
            )
        scales = tuple(float(value) for value in self.resolution_scales)
        if (
            len(scales) < 2
            or any(not np.isfinite(value) or value <= 0 for value in scales)
            or len(scales) != len(set(scales))
        ):
            raise ValueError(
                "resolution_scales must contain at least two unique positive values"
            )
        if not any(np.isclose(value, 1.0) for value in scales):
            raise ValueError("resolution_scales must contain 1.0")
        object.__setattr__(self, "resolution_scales", scales)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CuboidFaceSupportTrace:
    face: str
    boundary_m: float
    required_frame_count: int
    supporting_frame_ids: tuple[str, ...]
    supporting_point_count: int
    minimum_tangential_span_m: float | None

    @property
    def accepted(self) -> bool:
        return len(self.supporting_frame_ids) >= self.required_frame_count

    def to_dict(self) -> dict[str, object]:
        return {
            "face": self.face,
            "boundary_m": self.boundary_m,
            "required_frame_count": self.required_frame_count,
            "supporting_frame_ids": list(self.supporting_frame_ids),
            "supporting_point_count": self.supporting_point_count,
            "minimum_tangential_span_m": self.minimum_tangential_span_m,
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class CanonicalCuboidExperimentTrace:
    case_id: str
    track_id: str
    status: str
    reason_codes: tuple[str, ...]
    settings: CanonicalCuboidExperimentSettings
    geometry_frame_ids: tuple[str, ...]
    provisional_size_lwh: tuple[float, float, float] | None
    provisional_center_in_registration_xyz: tuple[float, float, float] | None
    provisional_yaw_in_registration_deg: float | None
    normal_count: int
    normal_fourfold_coherence: float | None
    ground_frame_count: int
    ground_median_m: float | None
    ground_mad_m: float | None
    face_support: tuple[CuboidFaceSupportTrace, ...]
    leave_one_out_maximum_dimension_change_m: tuple[float, float, float] | None
    leave_one_out_maximum_center_change_m: float | None
    leave_one_out_maximum_yaw_change_deg: float | None
    resolution_maximum_dimension_change_m: tuple[float, float, float] | None
    resolution_maximum_yaw_change_deg: float | None

    def __post_init__(self) -> None:
        if self.status not in {"candidate", "insufficient_evidence"}:
            raise ValueError("canonical cuboid experiment status is unsupported")
        if self.status == "candidate" and (
            self.reason_codes
            or self.provisional_size_lwh is None
            or self.provisional_center_in_registration_xyz is None
            or self.provisional_yaw_in_registration_deg is None
        ):
            raise ValueError("canonical cuboid candidate requires complete geometry")
        if self.status == "insufficient_evidence" and not self.reason_codes:
            raise ValueError("insufficient canonical cuboid requires reason codes")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": CANONICAL_CUBOID_EXPERIMENT_CONTRACT,
            "case_id": self.case_id,
            "track_id": self.track_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "settings": self.settings.to_dict(),
            "geometry_frame_ids": list(self.geometry_frame_ids),
            "provisional_size_lwh": (
                None
                if self.provisional_size_lwh is None
                else list(self.provisional_size_lwh)
            ),
            "provisional_center_in_registration_xyz": (
                None
                if self.provisional_center_in_registration_xyz is None
                else list(self.provisional_center_in_registration_xyz)
            ),
            "provisional_yaw_in_registration_deg": (
                self.provisional_yaw_in_registration_deg
            ),
            "normal_count": self.normal_count,
            "normal_fourfold_coherence": self.normal_fourfold_coherence,
            "ground_frame_count": self.ground_frame_count,
            "ground_median_m": self.ground_median_m,
            "ground_mad_m": self.ground_mad_m,
            "face_support": [item.to_dict() for item in self.face_support],
            "leave_one_out_maximum_dimension_change_m": (
                None
                if self.leave_one_out_maximum_dimension_change_m is None
                else list(self.leave_one_out_maximum_dimension_change_m)
            ),
            "leave_one_out_maximum_center_change_m": (
                self.leave_one_out_maximum_center_change_m
            ),
            "leave_one_out_maximum_yaw_change_deg": (
                self.leave_one_out_maximum_yaw_change_deg
            ),
            "resolution_maximum_dimension_change_m": (
                None
                if self.resolution_maximum_dimension_change_m is None
                else list(self.resolution_maximum_dimension_change_m)
            ),
            "resolution_maximum_yaw_change_deg": (
                self.resolution_maximum_yaw_change_deg
            ),
        }

    def write_json(self, path: str | Path) -> Path:
        output = Path(path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


@dataclass(frozen=True, slots=True)
class CanonicalCuboidExperimentRun:
    trace: GeometricRefinementTrace
    canonical_cuboid: CanonicalCuboidExperimentTrace


@dataclass(frozen=True, slots=True)
class _FrameGeometry:
    index: int
    frame_id: str
    points: NDArray[np.float64]
    ground_height_m: float | None
    ground_inlier_count: int


@dataclass(frozen=True, slots=True)
class _FitCandidate:
    size: NDArray[np.float64]
    center_in_registration: NDArray[np.float64]
    center_in_axes: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    yaw: float
    normal_count: int
    normal_coherence: float
    ground_median: float
    ground_mad: float
    required_boundary_frames: int
    rotated_groups: tuple[NDArray[np.float64], ...]


def fit_observable_canonical_cuboid(
    case: RefinementCase,
    stage3_trace: GeometricRefinementTrace,
    settings: CanonicalCuboidExperimentSettings | None = None,
) -> CanonicalCuboidExperimentRun:
    """Estimate one observable canonical cuboid from an accepted Stage 3 trace."""

    validate_geometric_trace(case, stage3_trace)
    resolved = settings or CanonicalCuboidExperimentSettings()
    preflight = _preflight_reasons(stage3_trace)
    frames = _frame_geometry(case, stage3_trace)
    if len(frames) < resolved.minimum_geometry_frames:
        preflight.append("insufficient_geometry_frames")
    if preflight:
        return _insufficient_run(
            case,
            stage3_trace,
            resolved,
            frames,
            tuple(dict.fromkeys(preflight)),
        )

    candidate = _fit_candidate(frames, resolved, resolved.canonical_voxel_size_m)
    if candidate is None:
        return _insufficient_run(
            case,
            stage3_trace,
            resolved,
            frames,
            ("insufficient_axis_normal_support",),
        )
    faces = _face_support(frames, candidate, resolved)
    leave_one_out = _leave_one_out_stability(frames, candidate, resolved)
    resolution = _resolution_stability(frames, candidate, resolved)
    reasons = _candidate_reasons(
        candidate,
        faces,
        leave_one_out,
        resolution,
        resolved,
    )
    diagnostics = _experiment_trace(
        case,
        frames,
        candidate,
        faces,
        leave_one_out,
        resolution,
        resolved,
        reasons,
    )
    if reasons:
        return CanonicalCuboidExperimentRun(
            _with_cuboid_status(stage3_trace, reasons, faces), diagnostics
        )
    materialized = _materialize_candidate(case, stage3_trace, candidate, faces)
    return CanonicalCuboidExperimentRun(materialized, diagnostics)


def _preflight_reasons(trace: GeometricRefinementTrace) -> list[str]:
    reasons: list[str] = []
    aggregation = trace.anchored_aggregation
    if trace.stage != REQUIRED_STAGE3_STAGE:
        reasons.append("unsupported_stage3_trace")
    if aggregation is None or aggregation.status != "candidate":
        reasons.append("stage3_insufficient_evidence")
    if trace.canonical_shape is None:
        reasons.append("missing_stage3_canonical_shape")
    if aggregation is not None and aggregation.status == "candidate":
        for frame in trace.frames:
            registration = frame.registration
            if registration is None:
                continue
            if (
                frame.frame_id != aggregation.anchor_frame_id
                and registration.status == "retained_coarse"
            ):
                reasons.append("stage3_candidate_retained_on_regression")
                break
    return reasons


def _frame_geometry(
    case: RefinementCase, trace: GeometricRefinementTrace
) -> list[_FrameGeometry]:
    values: list[_FrameGeometry] = []
    for index, (frame, frame_trace) in enumerate(
        zip(case.frames, trace.frames, strict=True)
    ):
        component = frame_trace.component
        registration = frame_trace.registration
        if component is None or component.frame_role is not FrameRole.GEOMETRY:
            continue
        if registration is None or registration.candidate_pose_annotation is None:
            continue
        positions = np.flatnonzero(
            frame_trace.point_states == EvidenceState.TARGET.value
        )
        point_indices = frame_trace.roi_point_indices[positions]
        points = inverse_transform_points(
            frame.points_xyz[point_indices], registration.candidate_pose_annotation
        )
        ground_height = _ground_height(frame_trace, registration)
        values.append(
            _FrameGeometry(
                index=index,
                frame_id=frame.frame_id,
                points=points,
                ground_height_m=ground_height,
                ground_inlier_count=(
                    0
                    if frame_trace.ground_plane is None
                    else frame_trace.ground_plane.inlier_count
                ),
            )
        )
    return values


def _ground_height(
    frame: FrameEvidenceTrace, registration: FrameRegistrationTrace
) -> float | None:
    ground = frame.ground_plane
    pose = registration.candidate_pose_annotation
    if ground is None or pose is None:
        return None
    x, y, _ = pose.translation_xyz
    a, b, c = ground.z_from_xyc
    point = np.asarray([[x, y, a * x + b * y + c]], dtype=np.float64)
    return float(inverse_transform_points(point, pose)[0, 2])


def _fit_candidate(
    frames: list[_FrameGeometry],
    settings: CanonicalCuboidExperimentSettings,
    voxel_size_m: float,
) -> _FitCandidate | None:
    points, support = _persistent_shape(frames, voxel_size_m)
    axis = _normal_axis(points, support, settings)
    if axis is None:
        return None
    yaw, normal_count, coherence = axis
    rotation = _yaw_rotation(yaw)
    rotated = tuple(frame.points @ rotation for frame in frames)
    required = min(
        len(frames),
        max(
            settings.boundary_minimum_frames,
            ceil(len(frames) * settings.boundary_minimum_frame_fraction),
        ),
    )
    lower_xy, upper_xy, upper_z = _supported_frontiers(rotated, required, settings)
    grounds = np.asarray(
        [
            frame.ground_height_m
            for frame in frames
            if frame.ground_height_m is not None
        ],
        dtype=np.float64,
    )
    if len(grounds):
        ground = float(np.median(grounds))
        ground_mad = float(np.median(np.abs(grounds - ground)))
    else:
        ground = np.nan
        ground_mad = np.inf
    lower = np.asarray((lower_xy[0], lower_xy[1], ground), dtype=np.float64)
    upper = np.asarray((upper_xy[0], upper_xy[1], upper_z), dtype=np.float64)
    center_axes = (lower + upper) / 2
    center_registration = center_axes @ rotation.T
    return _FitCandidate(
        size=upper - lower,
        center_in_registration=center_registration,
        center_in_axes=center_axes,
        lower=lower,
        upper=upper,
        yaw=yaw,
        normal_count=normal_count,
        normal_coherence=coherence,
        ground_median=ground,
        ground_mad=ground_mad,
        required_boundary_frames=required,
        rotated_groups=rotated,
    )


def _persistent_shape(
    frames: list[_FrameGeometry], voxel_size_m: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    cells: dict[tuple[int, int, int], list[NDArray[np.float64]]] = {}
    support: dict[tuple[int, int, int], set[int]] = {}
    for frame in frames:
        points, voxels = _per_voxel_points(frame.points, voxel_size_m)
        for point, voxel in zip(points, voxels, strict=True):
            key = tuple(int(value) for value in voxel)
            cells.setdefault(key, []).append(point)
            support.setdefault(key, set()).add(frame.index)
    keys = sorted(key for key, frame_ids in support.items() if len(frame_ids) >= 2)
    if not keys:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)
    points = np.asarray([np.median(cells[key], axis=0) for key in keys])
    weights = np.asarray([len(support[key]) for key in keys], dtype=np.float64)
    return points, weights


def _per_voxel_points(
    points: NDArray[np.float64], voxel_size_m: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    voxels = np.floor(points / voxel_size_m).astype(np.int64)
    order = np.lexsort(
        (
            points[:, 2],
            points[:, 1],
            points[:, 0],
            voxels[:, 2],
            voxels[:, 1],
            voxels[:, 0],
        )
    )
    points = points[order]
    voxels = voxels[order]
    starts = np.concatenate(
        (
            np.asarray([0]),
            np.flatnonzero(np.any(voxels[1:] != voxels[:-1], axis=1)) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.asarray([len(points)])))
    reduced = np.asarray(
        [
            np.median(points[start:end], axis=0)
            for start, end in zip(starts, ends, strict=True)
        ]
    )
    return reduced, voxels[starts]


def _normal_axis(
    points: NDArray[np.float64],
    support: NDArray[np.float64],
    settings: CanonicalCuboidExperimentSettings,
) -> tuple[float, int, float] | None:
    if len(points) < settings.normal_neighbor_count:
        return None
    tree_type = _ckdtree_type()
    tree = tree_type(points)
    _, indices = tree.query(
        points, k=min(settings.normal_neighbor_count, len(points)), workers=1
    )
    neighborhoods = points[indices]
    centered = neighborhoods - np.mean(neighborhoods, axis=1)[:, None, :]
    covariance = np.einsum("nki,nkj->nij", centered, centered) / centered.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    variation = eigenvalues[:, 0] / np.maximum(
        np.sum(eigenvalues, axis=1), np.finfo(np.float64).eps
    )
    retained = (variation <= settings.normal_maximum_surface_variation) & (
        np.abs(normals[:, 2]) <= settings.normal_maximum_absolute_z
    )
    count = int(np.count_nonzero(retained))
    if count < settings.normal_minimum_count:
        return None
    weights = support[retained] * np.square(
        1 - variation[retained] / settings.normal_maximum_surface_variation
    )
    angles = np.arctan2(normals[retained, 1], normals[retained, 0])
    resultant = np.sum(weights * np.exp(4j * angles))
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return None
    yaw = float(np.angle(resultant) / 4)
    while yaw > np.pi / 4:
        yaw -= np.pi / 2
    while yaw < -np.pi / 4:
        yaw += np.pi / 2
    return yaw, count, float(abs(resultant) / total_weight)


def _supported_frontiers(
    groups: tuple[NDArray[np.float64], ...],
    required: int,
    settings: CanonicalCuboidExperimentSettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    lower: list[NDArray[np.float64]] = []
    upper: list[NDArray[np.float64]] = []
    for points in groups:
        quantile = _frame_tail_quantile(len(points), settings)
        lower.append(np.quantile(points, quantile, axis=0))
        upper.append(np.quantile(points, 1 - quantile, axis=0))
    lower_values = np.asarray(lower)
    upper_values = np.asarray(upper)
    lower_xy = np.sort(lower_values[:, :2], axis=0)[required - 1]
    upper_xy = np.sort(upper_values[:, :2], axis=0)[-required]
    upper_z = float(np.sort(upper_values[:, 2])[-required])
    return lower_xy, upper_xy, upper_z


def _frame_tail_quantile(
    point_count: int, settings: CanonicalCuboidExperimentSettings
) -> float:
    tail_count = max(
        settings.boundary_minimum_tail_points,
        ceil(point_count * settings.boundary_tail_fraction),
    )
    return min(
        settings.boundary_maximum_tail_fraction,
        max(1 / (point_count + 1), tail_count / point_count),
    )


def _face_support(
    frames: list[_FrameGeometry],
    candidate: _FitCandidate,
    settings: CanonicalCuboidExperimentSettings,
) -> tuple[CuboidFaceSupportTrace, ...]:
    definitions = (
        ("length_lower", 0, candidate.lower[0]),
        ("length_upper", 0, candidate.upper[0]),
        ("width_lower", 1, candidate.lower[1]),
        ("width_upper", 1, candidate.upper[1]),
        ("height_upper", 2, candidate.upper[2]),
    )
    faces: list[CuboidFaceSupportTrace] = []
    for name, axis, boundary in definitions:
        frame_ids: list[str] = []
        point_count = 0
        spans: list[float] = []
        other_axes = [index for index in range(3) if index != axis]
        for frame, raw_points in zip(frames, candidate.rotated_groups, strict=True):
            points, _ = _per_voxel_points(raw_points, settings.canonical_voxel_size_m)
            surface = _surface_normals(points, settings.face_normal_neighbor_count)
            if surface is None:
                continue
            normals, variation = surface
            selected = (
                (np.abs(points[:, axis] - boundary) <= settings.face_band_m)
                & (
                    np.abs(normals[:, axis])
                    >= settings.face_minimum_absolute_normal_alignment
                )
                & (variation <= settings.face_maximum_surface_variation)
            )
            count = int(np.count_nonzero(selected))
            point_count += count
            if count < settings.face_minimum_points_per_frame:
                continue
            span = float(np.max(np.ptp(points[selected][:, other_axes], axis=0)))
            if span < settings.face_minimum_tangential_span_m:
                continue
            frame_ids.append(frame.frame_id)
            spans.append(span)
        faces.append(
            CuboidFaceSupportTrace(
                face=name,
                boundary_m=float(boundary),
                required_frame_count=candidate.required_boundary_frames,
                supporting_frame_ids=tuple(frame_ids),
                supporting_point_count=point_count,
                minimum_tangential_span_m=min(spans) if spans else None,
            )
        )
    ground_frames = [frame for frame in frames if frame.ground_height_m is not None]
    faces.append(
        CuboidFaceSupportTrace(
            face="height_lower_ground",
            boundary_m=float(candidate.lower[2]),
            required_frame_count=candidate.required_boundary_frames,
            supporting_frame_ids=tuple(frame.frame_id for frame in ground_frames),
            supporting_point_count=sum(
                frame.ground_inlier_count for frame in ground_frames
            ),
            minimum_tangential_span_m=None,
        )
    )
    return tuple(faces)


def _surface_normals(
    points: NDArray[np.float64], neighbor_count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    if len(points) < neighbor_count:
        return None
    tree_type = _ckdtree_type()
    _, indices = tree_type(points).query(
        points, k=min(neighbor_count, len(points)), workers=1
    )
    neighborhoods = points[indices]
    centered = neighborhoods - np.mean(neighborhoods, axis=1)[:, None, :]
    covariance = np.einsum("nki,nkj->nij", centered, centered) / centered.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    variation = eigenvalues[:, 0] / np.maximum(
        np.sum(eigenvalues, axis=1), np.finfo(np.float64).eps
    )
    return eigenvectors[:, :, 0], variation


def _leave_one_out_stability(
    frames: list[_FrameGeometry],
    candidate: _FitCandidate,
    settings: CanonicalCuboidExperimentSettings,
) -> tuple[NDArray[np.float64], float, float] | None:
    candidates: list[_FitCandidate] = []
    for index in range(len(frames)):
        subset = frames[:index] + frames[index + 1 :]
        fit = _fit_candidate(subset, settings, settings.canonical_voxel_size_m)
        if fit is None:
            return None
        candidates.append(fit)
    dimension = np.max(
        np.abs(np.asarray([item.size for item in candidates]) - candidate.size), axis=0
    )
    center = max(
        float(
            np.linalg.norm(
                item.center_in_registration - candidate.center_in_registration
            )
        )
        for item in candidates
    )
    yaw = max(
        abs(degrees(angle_difference(item.yaw, candidate.yaw))) for item in candidates
    )
    return dimension, center, yaw


def _resolution_stability(
    frames: list[_FrameGeometry],
    candidate: _FitCandidate,
    settings: CanonicalCuboidExperimentSettings,
) -> tuple[NDArray[np.float64], float] | None:
    candidates: list[_FitCandidate] = []
    for scale in settings.resolution_scales:
        fit = _fit_candidate(frames, settings, settings.canonical_voxel_size_m * scale)
        if fit is None:
            return None
        candidates.append(fit)
    dimension = np.max(
        np.abs(np.asarray([item.size for item in candidates]) - candidate.size), axis=0
    )
    yaw = max(
        abs(degrees(angle_difference(item.yaw, candidate.yaw))) for item in candidates
    )
    return dimension, yaw


def _candidate_reasons(
    candidate: _FitCandidate,
    faces: tuple[CuboidFaceSupportTrace, ...],
    leave_one_out: tuple[NDArray[np.float64], float, float] | None,
    resolution: tuple[NDArray[np.float64], float] | None,
    settings: CanonicalCuboidExperimentSettings,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if candidate.normal_count < settings.normal_minimum_count:
        reasons.append("insufficient_axis_normal_support")
    if candidate.normal_coherence < settings.normal_minimum_fourfold_coherence:
        reasons.append("weak_axis_normal_coherence")
    if abs(degrees(candidate.yaw)) >= settings.maximum_common_yaw_correction_deg:
        reasons.append("common_yaw_correction_bound_exceeded")
    if not np.isfinite((*candidate.size, *candidate.center_in_registration)).all():
        reasons.append("non_finite_canonical_cuboid")
    elif np.any(candidate.size <= 0):
        reasons.append("non_positive_canonical_dimension")
    if candidate.ground_mad > settings.ground_maximum_mad_m:
        reasons.append("unstable_ground_height")
    for face in faces:
        if not face.accepted:
            reasons.append(f"insufficient_face_support:{face.face}")
    center_xy = float(np.linalg.norm(candidate.center_in_registration[:2]))
    if center_xy > settings.maximum_center_xy_correction_m:
        reasons.append("center_xy_correction_bound_exceeded")
    if abs(float(candidate.center_in_registration[2])) > (
        settings.maximum_center_z_correction_m
    ):
        reasons.append("center_z_correction_bound_exceeded")
    if leave_one_out is None:
        reasons.append("leave_one_out_fit_unavailable")
    else:
        dimension, center, yaw = leave_one_out
        if np.any(dimension > settings.leave_one_out_maximum_dimension_change_m):
            reasons.append("leave_one_out_dimension_unstable")
        if center > settings.leave_one_out_maximum_center_change_m:
            reasons.append("leave_one_out_center_unstable")
        if yaw > settings.leave_one_out_maximum_yaw_change_deg:
            reasons.append("leave_one_out_yaw_unstable")
    if resolution is None:
        reasons.append("resolution_fit_unavailable")
    else:
        dimension, yaw = resolution
        if np.any(dimension > settings.resolution_maximum_dimension_change_m):
            reasons.append("resolution_dimension_unstable")
        if yaw > settings.resolution_maximum_yaw_change_deg:
            reasons.append("resolution_yaw_unstable")
    return tuple(dict.fromkeys(reasons))


def _experiment_trace(
    case: RefinementCase,
    frames: list[_FrameGeometry],
    candidate: _FitCandidate,
    faces: tuple[CuboidFaceSupportTrace, ...],
    leave_one_out: tuple[NDArray[np.float64], float, float] | None,
    resolution: tuple[NDArray[np.float64], float] | None,
    settings: CanonicalCuboidExperimentSettings,
    reasons: tuple[str, ...],
) -> CanonicalCuboidExperimentTrace:
    return CanonicalCuboidExperimentTrace(
        case_id=case.case_id,
        track_id=case.track.track_id,
        status="insufficient_evidence" if reasons else "candidate",
        reason_codes=reasons,
        settings=settings,
        geometry_frame_ids=tuple(frame.frame_id for frame in frames),
        provisional_size_lwh=tuple(float(value) for value in candidate.size),
        provisional_center_in_registration_xyz=tuple(
            float(value) for value in candidate.center_in_registration
        ),
        provisional_yaw_in_registration_deg=degrees(candidate.yaw),
        normal_count=candidate.normal_count,
        normal_fourfold_coherence=candidate.normal_coherence,
        ground_frame_count=sum(frame.ground_height_m is not None for frame in frames),
        ground_median_m=candidate.ground_median,
        ground_mad_m=candidate.ground_mad,
        face_support=faces,
        leave_one_out_maximum_dimension_change_m=(
            None
            if leave_one_out is None
            else tuple(float(value) for value in leave_one_out[0])
        ),
        leave_one_out_maximum_center_change_m=(
            None if leave_one_out is None else leave_one_out[1]
        ),
        leave_one_out_maximum_yaw_change_deg=(
            None if leave_one_out is None else leave_one_out[2]
        ),
        resolution_maximum_dimension_change_m=(
            None
            if resolution is None
            else tuple(float(value) for value in resolution[0])
        ),
        resolution_maximum_yaw_change_deg=(
            None if resolution is None else resolution[1]
        ),
    )


def _insufficient_run(
    case: RefinementCase,
    trace: GeometricRefinementTrace,
    settings: CanonicalCuboidExperimentSettings,
    frames: list[_FrameGeometry],
    reasons: tuple[str, ...],
) -> CanonicalCuboidExperimentRun:
    diagnostics = CanonicalCuboidExperimentTrace(
        case_id=case.case_id,
        track_id=case.track.track_id,
        status="insufficient_evidence",
        reason_codes=reasons,
        settings=settings,
        geometry_frame_ids=tuple(frame.frame_id for frame in frames),
        provisional_size_lwh=None,
        provisional_center_in_registration_xyz=None,
        provisional_yaw_in_registration_deg=None,
        normal_count=0,
        normal_fourfold_coherence=None,
        ground_frame_count=sum(frame.ground_height_m is not None for frame in frames),
        ground_median_m=None,
        ground_mad_m=None,
        face_support=(),
        leave_one_out_maximum_dimension_change_m=None,
        leave_one_out_maximum_center_change_m=None,
        leave_one_out_maximum_yaw_change_deg=None,
        resolution_maximum_dimension_change_m=None,
        resolution_maximum_yaw_change_deg=None,
    )
    return CanonicalCuboidExperimentRun(
        _with_cuboid_status(trace, reasons, ()), diagnostics
    )


def _with_cuboid_status(
    trace: GeometricRefinementTrace,
    reasons: tuple[str, ...],
    faces: tuple[CuboidFaceSupportTrace, ...],
) -> GeometricRefinementTrace:
    counts = tuple(
        next(
            (item.supporting_point_count for item in faces if item.face == face),
            0,
        )
        for face in (
            "length_lower",
            "length_upper",
            "width_lower",
            "width_upper",
            "height_lower_ground",
            "height_upper",
        )
    )
    return replace(
        trace,
        stage=CANONICAL_CUBOID_STAGE,
        cuboid_fit=CuboidFitTrace(
            status="insufficient_evidence",
            reason_codes=reasons,
            canonical_size_lwh=None,
            center_in_registration_xyz=None,
            lower_envelope_xyz=None,
            upper_envelope_xyz=None,
            face_support_counts=counts,
            alternations=1,
            converged=False,
        ),
    )


def _materialize_candidate(
    case: RefinementCase,
    trace: GeometricRefinementTrace,
    candidate: _FitCandidate,
    faces: tuple[CuboidFaceSupportTrace, ...],
) -> GeometricRefinementTrace:
    registration_from_canonical = Pose3D(
        tuple(float(value) for value in candidate.center_in_registration),
        _yaw_quaternion(candidate.yaw),
    )
    frames = tuple(
        _materialize_frame(case, index, frame, registration_from_canonical)
        for index, frame in enumerate(trace.frames)
    )
    shape = trace.canonical_shape
    if shape is None:
        raise AssertionError("accepted Stage 4 candidate lost Stage 3 shape")
    canonical_points = inverse_transform_points(
        shape.points_xyz, registration_from_canonical
    ).astype(np.float32)
    canonical_shape = CanonicalShapeTrace(
        points_xyz=canonical_points,
        frame_support_count=shape.frame_support_count,
        registered_frame_ids=shape.registered_frame_ids,
        voxel_size_m=shape.voxel_size_m,
        iterations=shape.iterations,
        converged=shape.converged,
    )
    size = tuple(float(value) for value in candidate.size)
    half = candidate.size / 2
    counts = tuple(
        next(item.supporting_point_count for item in faces if item.face == face)
        for face in (
            "length_lower",
            "length_upper",
            "width_lower",
            "width_upper",
            "height_lower_ground",
            "height_upper",
        )
    )
    return replace(
        trace,
        stage=CANONICAL_CUBOID_STAGE,
        frames=frames,
        canonical_shape=canonical_shape,
        cuboid_fit=CuboidFitTrace(
            status="candidate",
            reason_codes=(),
            canonical_size_lwh=size,
            center_in_registration_xyz=(0.0, 0.0, 0.0),
            lower_envelope_xyz=tuple(float(value) for value in -half),
            upper_envelope_xyz=tuple(float(value) for value in half),
            face_support_counts=counts,
            alternations=1,
            converged=True,
        ),
    )


def _materialize_frame(
    case: RefinementCase,
    index: int,
    frame: FrameEvidenceTrace,
    registration_from_canonical: Pose3D,
) -> FrameEvidenceTrace:
    registration = frame.registration
    if registration is None or registration.candidate_pose_annotation is None:
        return frame
    candidate = compose_pose(
        registration.candidate_pose_annotation, registration_from_canonical
    )
    coarse = case.track.observations[index].coarse_box.pose
    world_from_coarse = compose_pose(case.frames[index].world_from_annotation, coarse)
    world_from_candidate = compose_pose(
        case.frames[index].world_from_annotation, candidate
    )
    canonical_from_coarse = compose_pose(
        inverse_pose(world_from_candidate), world_from_coarse
    )
    return replace(
        frame,
        registration=FrameRegistrationTrace(
            status="registered",
            reason_codes=(),
            canonical_from_coarse=canonical_from_coarse,
            candidate_pose_annotation=candidate,
            iterations=registration.iterations,
            correspondence_count=registration.correspondence_count,
            initial_rmse_m=registration.initial_rmse_m,
            final_rmse_m=registration.final_rmse_m,
            translation_correction_m=float(
                np.linalg.norm(canonical_from_coarse.translation_xyz[:2])
            ),
            yaw_correction_deg=abs(
                degrees(yaw_from_quaternion(canonical_from_coarse.orientation_xyzw))
            ),
        ),
    )


def _yaw_rotation(yaw: float) -> NDArray[np.float64]:
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))


def _ckdtree_type():
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError(
            "canonical cuboid estimation requires "
            "'pip install trackrefinery[geometric]'"
        ) from error
    return cKDTree
