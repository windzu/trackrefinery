"""Immutable point-evidence traces and portable sidecar serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trackrefinery.contracts import Pose3D, RefinementCase, RefinementOutcome

EVIDENCE_TRACE_CONTRACT = "trackrefinery-geometric-evidence-trace-v1"


class EvidenceState(IntEnum):
    """Point ownership state stored in compact trace arrays."""

    TARGET = 1
    AMBIGUOUS = 2
    BACKGROUND = 3
    GROUND = 4


class FrameRole(str, Enum):
    """Authority granted to one frame by the V2 component stage."""

    GEOMETRY = "geometry"
    POSE_ONLY = "pose_only"
    TRAJECTORY_ONLY = "trajectory_only"


@dataclass(frozen=True, slots=True)
class FrameComponentTrace:
    """V2 component-selection decision and measurable frame role evidence."""

    status: str
    frame_role: FrameRole
    reason_codes: tuple[str, ...]
    component_count: int
    candidate_component_count: int
    selected_component_id: int | None
    selected_point_count: int
    selected_voxel_count: int
    seed_point_count: int
    component_dominance: float | None
    nearest_competing_distance_m: float | None
    robust_spread_xyz_m: tuple[float, float, float] | None
    resolution_stability_iou: float | None
    outside_coarse_envelope_fraction: float | None

    def __post_init__(self) -> None:
        if self.status not in {"selected", "ambiguous", "insufficient_evidence"}:
            raise ValueError("component status is unsupported")
        object.__setattr__(self, "frame_role", FrameRole(self.frame_role))
        reasons = tuple(self.reason_codes)
        if any(not isinstance(value, str) or not value for value in reasons):
            raise ValueError("component reason_codes must be non-empty strings")
        counts = (
            self.component_count,
            self.candidate_component_count,
            self.selected_point_count,
            self.selected_voxel_count,
            self.seed_point_count,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ):
            raise ValueError("component counts must be non-negative integers")
        if self.candidate_component_count > self.component_count:
            raise ValueError("candidate component count exceeds total components")
        if self.status in {"selected", "ambiguous"}:
            if (
                self.selected_component_id is None
                or self.selected_component_id < 0
                or self.selected_component_id >= self.component_count
                or self.selected_point_count < 1
                or self.selected_voxel_count < 1
                or self.seed_point_count < 1
                or self.component_dominance is None
                or self.robust_spread_xyz_m is None
                or self.resolution_stability_iou is None
            ):
                raise ValueError("selected or ambiguous component requires metrics")
            if self.status == "ambiguous" and (
                self.frame_role is not FrameRole.TRAJECTORY_ONLY or not reasons
            ):
                raise ValueError(
                    "ambiguous component requires trajectory-only role and reasons"
                )
        else:
            if self.frame_role is not FrameRole.TRAJECTORY_ONLY or not reasons:
                raise ValueError(
                    "insufficient component requires trajectory-only role and reasons"
                )
            if any(
                value is not None
                for value in (
                    self.selected_component_id,
                    self.component_dominance,
                    self.robust_spread_xyz_m,
                    self.resolution_stability_iou,
                    self.outside_coarse_envelope_fraction,
                )
            ):
                raise ValueError("insufficient component cannot carry selected metrics")
            if any(
                value != 0
                for value in (
                    self.selected_point_count,
                    self.selected_voxel_count,
                    self.seed_point_count,
                )
            ):
                raise ValueError("insufficient component cannot carry selected counts")
        for name, value in (
            ("component_dominance", self.component_dominance),
            ("resolution_stability_iou", self.resolution_stability_iou),
            (
                "outside_coarse_envelope_fraction",
                self.outside_coarse_envelope_fraction,
            ),
        ):
            if value is not None and (
                not np.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.nearest_competing_distance_m is not None and (
            not np.isfinite(self.nearest_competing_distance_m)
            or self.nearest_competing_distance_m < 0
        ):
            raise ValueError(
                "nearest_competing_distance_m must be finite and non-negative"
            )
        if self.robust_spread_xyz_m is not None:
            spread = tuple(float(value) for value in self.robust_spread_xyz_m)
            if (
                len(spread) != 3
                or not np.isfinite(spread).all()
                or any(value < 0 for value in spread)
            ):
                raise ValueError("robust_spread_xyz_m must be a non-negative triplet")
            object.__setattr__(self, "robust_spread_xyz_m", spread)
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "frame_role": self.frame_role.value,
            "reason_codes": list(self.reason_codes),
            "component_count": self.component_count,
            "candidate_component_count": self.candidate_component_count,
            "selected_component_id": self.selected_component_id,
            "selected_point_count": self.selected_point_count,
            "selected_voxel_count": self.selected_voxel_count,
            "seed_point_count": self.seed_point_count,
            "component_dominance": self.component_dominance,
            "nearest_competing_distance_m": self.nearest_competing_distance_m,
            "robust_spread_xyz_m": (
                None
                if self.robust_spread_xyz_m is None
                else list(self.robust_spread_xyz_m)
            ),
            "resolution_stability_iou": self.resolution_stability_iou,
            "outside_coarse_envelope_fraction": (self.outside_coarse_envelope_fraction),
        }

    @classmethod
    def from_dict(cls, value: object) -> FrameComponentTrace:
        if not isinstance(value, dict):
            raise ValueError("component must be an object")
        reasons = value.get("reason_codes")
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) for reason in reasons
        ):
            raise ValueError("component reason_codes must be a string list")
        selected_id = value.get("selected_component_id")
        if selected_id is not None:
            selected_id = _integer(selected_id, "selected_component_id")
        spread = value.get("robust_spread_xyz_m")
        return cls(
            status=_string(value.get("status"), "component.status"),
            frame_role=FrameRole(
                _string(value.get("frame_role"), "component.frame_role")
            ),
            reason_codes=tuple(reasons),
            component_count=_integer(value.get("component_count"), "component_count"),
            candidate_component_count=_integer(
                value.get("candidate_component_count"),
                "candidate_component_count",
            ),
            selected_component_id=selected_id,
            selected_point_count=_integer(
                value.get("selected_point_count"), "selected_point_count"
            ),
            selected_voxel_count=_integer(
                value.get("selected_voxel_count"), "selected_voxel_count"
            ),
            seed_point_count=_integer(
                value.get("seed_point_count"), "seed_point_count"
            ),
            component_dominance=_optional_number(
                value.get("component_dominance"), "component_dominance"
            ),
            nearest_competing_distance_m=_optional_number(
                value.get("nearest_competing_distance_m"),
                "nearest_competing_distance_m",
            ),
            robust_spread_xyz_m=(
                None
                if spread is None
                else _float_triplet(spread, "robust_spread_xyz_m")
            ),
            resolution_stability_iou=_optional_number(
                value.get("resolution_stability_iou"),
                "resolution_stability_iou",
            ),
            outside_coarse_envelope_fraction=_optional_number(
                value.get("outside_coarse_envelope_fraction"),
                "outside_coarse_envelope_fraction",
            ),
        )


@dataclass(frozen=True, slots=True)
class GroundPlaneEstimate:
    """Robust plane represented as ``z = a*x + b*y + c``."""

    z_from_xyc: tuple[float, float, float]
    normal_xyz: tuple[float, float, float]
    candidate_count: int
    inlier_count: int
    rmse_m: float
    tilt_deg: float

    def __post_init__(self) -> None:
        coefficients = tuple(float(value) for value in self.z_from_xyc)
        normal = tuple(float(value) for value in self.normal_xyz)
        numeric = (*coefficients, *normal, float(self.rmse_m), float(self.tilt_deg))
        if len(coefficients) != 3 or len(normal) != 3 or not np.isfinite(numeric).all():
            raise ValueError("ground plane values must be finite triplets")
        if self.candidate_count < 3:
            raise ValueError("ground plane requires at least three candidates")
        if not 3 <= self.inlier_count <= self.candidate_count:
            raise ValueError("ground plane inlier count is invalid")
        if self.rmse_m < 0 or not 0 <= self.tilt_deg < 90:
            raise ValueError("ground plane residual or tilt is invalid")
        norm = float(np.linalg.norm(normal))
        if not np.isclose(norm, 1.0, atol=1e-6) or normal[2] <= 0:
            raise ValueError("ground plane normal must be unit length with positive Z")
        object.__setattr__(self, "z_from_xyc", coefficients)
        object.__setattr__(self, "normal_xyz", normal)

    def to_dict(self) -> dict[str, object]:
        return {
            "z_from_xyc": list(self.z_from_xyc),
            "normal_xyz": list(self.normal_xyz),
            "candidate_count": self.candidate_count,
            "inlier_count": self.inlier_count,
            "rmse_m": self.rmse_m,
            "tilt_deg": self.tilt_deg,
        }

    @classmethod
    def from_dict(cls, value: object) -> GroundPlaneEstimate:
        if not isinstance(value, dict):
            raise ValueError("ground_plane must be an object")
        return cls(
            z_from_xyc=_float_triplet(value.get("z_from_xyc"), "z_from_xyc"),
            normal_xyz=_float_triplet(value.get("normal_xyz"), "normal_xyz"),
            candidate_count=_integer(value.get("candidate_count"), "candidate_count"),
            inlier_count=_integer(value.get("inlier_count"), "inlier_count"),
            rmse_m=_number(value.get("rmse_m"), "rmse_m"),
            tilt_deg=_number(value.get("tilt_deg"), "tilt_deg"),
        )


@dataclass(frozen=True, slots=True)
class FrameRegistrationTrace:
    """Provisional registration state for one frame, never a released result."""

    status: str
    reason_codes: tuple[str, ...]
    canonical_from_coarse: Pose3D | None
    candidate_pose_annotation: Pose3D | None
    iterations: int
    correspondence_count: int
    initial_rmse_m: float | None
    final_rmse_m: float | None
    translation_correction_m: float | None
    yaw_correction_deg: float | None

    def __post_init__(self) -> None:
        if self.status not in {"registered", "insufficient_evidence"}:
            raise ValueError("registration status is unsupported")
        reasons = tuple(self.reason_codes)
        if self.status == "registered":
            if reasons:
                raise ValueError("registered frames cannot contain reason codes")
            if (
                self.canonical_from_coarse is None
                or self.candidate_pose_annotation is None
            ):
                raise ValueError("registered frames require both provisional poses")
            if (
                self.iterations < 1
                or self.correspondence_count < 1
                or any(
                    value is None
                    for value in (
                        self.initial_rmse_m,
                        self.final_rmse_m,
                        self.translation_correction_m,
                        self.yaw_correction_deg,
                    )
                )
            ):
                raise ValueError("registered frames require complete metrics")
        else:
            if not reasons or any(not value for value in reasons):
                raise ValueError("insufficient registration requires reason codes")
            if (
                self.canonical_from_coarse is not None
                or self.candidate_pose_annotation is not None
            ):
                raise ValueError("insufficient registration cannot contain poses")
        if self.iterations < 0 or self.correspondence_count < 0:
            raise ValueError("registration counts must be non-negative")
        metrics = (
            self.initial_rmse_m,
            self.final_rmse_m,
            self.translation_correction_m,
            self.yaw_correction_deg,
        )
        if any(
            value is not None and (not np.isfinite(value) or value < 0)
            for value in metrics
        ):
            raise ValueError("registration metrics must be finite and non-negative")
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "canonical_from_coarse": _pose_to_dict(self.canonical_from_coarse),
            "candidate_pose_annotation": _pose_to_dict(self.candidate_pose_annotation),
            "iterations": self.iterations,
            "correspondence_count": self.correspondence_count,
            "initial_rmse_m": self.initial_rmse_m,
            "final_rmse_m": self.final_rmse_m,
            "translation_correction_m": self.translation_correction_m,
            "yaw_correction_deg": self.yaw_correction_deg,
        }

    @classmethod
    def from_dict(cls, value: object) -> FrameRegistrationTrace:
        if not isinstance(value, dict):
            raise ValueError("registration must be an object")
        reasons = value.get("reason_codes")
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) for reason in reasons
        ):
            raise ValueError("registration reason_codes must be a string list")
        return cls(
            status=_string(value.get("status"), "registration.status"),
            reason_codes=tuple(reasons),
            canonical_from_coarse=_optional_pose(
                value.get("canonical_from_coarse"), "canonical_from_coarse"
            ),
            candidate_pose_annotation=_optional_pose(
                value.get("candidate_pose_annotation"), "candidate_pose_annotation"
            ),
            iterations=_integer(value.get("iterations"), "registration.iterations"),
            correspondence_count=_integer(
                value.get("correspondence_count"),
                "registration.correspondence_count",
            ),
            initial_rmse_m=_optional_number(
                value.get("initial_rmse_m"), "registration.initial_rmse_m"
            ),
            final_rmse_m=_optional_number(
                value.get("final_rmse_m"), "registration.final_rmse_m"
            ),
            translation_correction_m=_optional_number(
                value.get("translation_correction_m"),
                "registration.translation_correction_m",
            ),
            yaw_correction_deg=_optional_number(
                value.get("yaw_correction_deg"),
                "registration.yaw_correction_deg",
            ),
        )


@dataclass(frozen=True, slots=True)
class FrameEvidenceTrace:
    """Point states for one ROI, indexed back into the immutable frame cloud."""

    frame_id: str
    roi_point_indices: NDArray[np.int64]
    point_states: NDArray[np.uint8]
    ground_plane: GroundPlaneEstimate | None = None
    represented_sensor_ids: tuple[str, ...] = ()
    component: FrameComponentTrace | None = None
    registration: FrameRegistrationTrace | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be a non-empty string")
        indices = np.asarray(self.roi_point_indices)
        states = np.asarray(self.point_states)
        if indices.dtype != np.int64 or indices.ndim != 1:
            raise ValueError("roi_point_indices must be int64 with shape [R]")
        if states.dtype != np.uint8 or states.shape != indices.shape:
            raise ValueError("point_states must be uint8 and align with ROI indices")
        if len(indices) and (np.any(indices < 0) or np.any(np.diff(indices) <= 0)):
            raise ValueError("ROI indices must be unique, ordered, and non-negative")
        valid_states = np.asarray(
            [value.value for value in EvidenceState], dtype=np.uint8
        )
        if len(states) and not np.isin(states, valid_states).all():
            raise ValueError("point_states contains an unsupported evidence state")
        sensors = tuple(self.represented_sensor_ids)
        if len(sensors) != len(set(sensors)) or any(not value for value in sensors):
            raise ValueError("represented sensor IDs must be unique and non-empty")
        indices = indices.copy()
        states = states.copy()
        indices.setflags(write=False)
        states.setflags(write=False)
        object.__setattr__(self, "roi_point_indices", indices)
        object.__setattr__(self, "point_states", states)
        object.__setattr__(self, "represented_sensor_ids", sensors)

    def count(self, state: EvidenceState) -> int:
        return int(np.count_nonzero(self.point_states == state.value))

    @property
    def counts(self) -> dict[str, int]:
        return {state.name.lower(): self.count(state) for state in EvidenceState}

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "roi_point_count": len(self.roi_point_indices),
            "point_state_counts": self.counts,
            "represented_sensor_ids": list(self.represented_sensor_ids),
            "ground_plane": (
                None if self.ground_plane is None else self.ground_plane.to_dict()
            ),
            "component": None if self.component is None else self.component.to_dict(),
            "registration": (
                None if self.registration is None else self.registration.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class CanonicalShapeTrace:
    """Persistent, voxelized object evidence in the provisional object frame."""

    points_xyz: NDArray[np.float32]
    frame_support_count: NDArray[np.uint16]
    registered_frame_ids: tuple[str, ...]
    voxel_size_m: float
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xyz)
        support = np.asarray(self.frame_support_count)
        if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("canonical points_xyz must be float32 with shape [C, 3]")
        if not len(points):
            raise ValueError("canonical points_xyz must be non-empty")
        if not np.isfinite(points).all():
            raise ValueError("canonical points_xyz must be finite")
        if support.dtype != np.uint16 or support.shape != (len(points),):
            raise ValueError("frame_support_count must be uint16 with shape [C]")
        frame_ids = tuple(self.registered_frame_ids)
        if len(frame_ids) < 2 or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("canonical shape requires at least two registered frames")
        if len(support) and (np.any(support < 2) or np.any(support > len(frame_ids))):
            raise ValueError("canonical support counts are outside frame bounds")
        if not np.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0:
            raise ValueError("canonical voxel_size_m must be finite and positive")
        if self.iterations < 1:
            raise ValueError("canonical iterations must be positive")
        if not isinstance(self.converged, bool):
            raise ValueError("canonical converged must be boolean")
        points = points.copy()
        support = support.copy()
        points.setflags(write=False)
        support.setflags(write=False)
        object.__setattr__(self, "points_xyz", points)
        object.__setattr__(self, "frame_support_count", support)
        object.__setattr__(self, "registered_frame_ids", frame_ids)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "point_count": len(self.points_xyz),
            "registered_frame_ids": list(self.registered_frame_ids),
            "voxel_size_m": self.voxel_size_m,
            "iterations": self.iterations,
            "converged": self.converged,
            "minimum_frame_support": (
                None
                if not len(self.frame_support_count)
                else int(self.frame_support_count.min())
            ),
            "maximum_frame_support": (
                None
                if not len(self.frame_support_count)
                else int(self.frame_support_count.max())
            ),
        }


@dataclass(frozen=True, slots=True)
class CuboidFitTrace:
    """Trace-only visible-envelope cuboid candidate in registration coordinates."""

    status: str
    reason_codes: tuple[str, ...]
    canonical_size_lwh: tuple[float, float, float] | None
    center_in_registration_xyz: tuple[float, float, float] | None
    lower_envelope_xyz: tuple[float, float, float] | None
    upper_envelope_xyz: tuple[float, float, float] | None
    face_support_counts: tuple[int, int, int, int, int, int]
    alternations: int
    converged: bool

    def __post_init__(self) -> None:
        if self.status not in {"candidate", "insufficient_evidence"}:
            raise ValueError("cuboid fit status is unsupported")
        reasons = tuple(self.reason_codes)
        tuples = (
            self.canonical_size_lwh,
            self.center_in_registration_xyz,
            self.lower_envelope_xyz,
            self.upper_envelope_xyz,
        )
        if self.status == "candidate":
            if reasons or any(value is None for value in tuples):
                raise ValueError("cuboid candidate requires complete geometry")
        elif not reasons or any(not value for value in reasons):
            raise ValueError("insufficient cuboid fit requires reason codes")
        for name, value in zip(
            ("canonical_size_lwh", "center", "lower_envelope", "upper_envelope"),
            tuples,
            strict=True,
        ):
            if value is not None:
                parsed = tuple(float(item) for item in value)
                if len(parsed) != 3 or not np.isfinite(parsed).all():
                    raise ValueError(f"cuboid {name} must be a finite triplet")
                object.__setattr__(
                    self,
                    {
                        "center": "center_in_registration_xyz",
                        "lower_envelope": "lower_envelope_xyz",
                        "upper_envelope": "upper_envelope_xyz",
                    }.get(name, name),
                    parsed,
                )
        if self.canonical_size_lwh is not None and any(
            value <= 0 for value in self.canonical_size_lwh
        ):
            raise ValueError("cuboid dimensions must be positive")
        counts = tuple(int(value) for value in self.face_support_counts)
        if len(counts) != 6 or any(value < 0 for value in counts):
            raise ValueError("face_support_counts must contain six non-negative values")
        if self.alternations < 1:
            raise ValueError("cuboid alternations must be positive")
        if not isinstance(self.converged, bool):
            raise ValueError("cuboid converged must be boolean")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "face_support_counts", counts)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "canonical_size_lwh": _optional_triplet_list(self.canonical_size_lwh),
            "center_in_registration_xyz": _optional_triplet_list(
                self.center_in_registration_xyz
            ),
            "lower_envelope_xyz": _optional_triplet_list(self.lower_envelope_xyz),
            "upper_envelope_xyz": _optional_triplet_list(self.upper_envelope_xyz),
            "face_support_counts": list(self.face_support_counts),
            "alternations": self.alternations,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, value: object) -> CuboidFitTrace:
        if not isinstance(value, dict):
            raise ValueError("cuboid_fit must be an object")
        reasons = value.get("reason_codes")
        counts = value.get("face_support_counts")
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) for reason in reasons
        ):
            raise ValueError("cuboid reason_codes must be a string list")
        if not isinstance(counts, list) or len(counts) != 6:
            raise ValueError("cuboid face_support_counts must contain six values")
        return cls(
            status=_string(value.get("status"), "cuboid status"),
            reason_codes=tuple(reasons),
            canonical_size_lwh=_optional_float_triplet(
                value.get("canonical_size_lwh"), "canonical_size_lwh"
            ),
            center_in_registration_xyz=_optional_float_triplet(
                value.get("center_in_registration_xyz"),
                "center_in_registration_xyz",
            ),
            lower_envelope_xyz=_optional_float_triplet(
                value.get("lower_envelope_xyz"), "lower_envelope_xyz"
            ),
            upper_envelope_xyz=_optional_float_triplet(
                value.get("upper_envelope_xyz"), "upper_envelope_xyz"
            ),
            face_support_counts=tuple(
                _integer(item, "face_support_counts[]") for item in counts
            ),  # type: ignore[arg-type]
            alternations=_integer(value.get("alternations"), "alternations"),
            converged=_boolean(value.get("converged"), "cuboid converged"),
        )


@dataclass(frozen=True, slots=True)
class GeometricRefinementTrace:
    """One deterministic algorithm-stage trace for a refinement case."""

    case_id: str
    track_id: str
    algorithm_version: str
    config_schema_version: str
    config_sha256: str
    settings_json: str
    stage: str
    frames: tuple[FrameEvidenceTrace, ...]
    canonical_shape: CanonicalShapeTrace | None = None
    cuboid_fit: CuboidFitTrace | None = None

    def __post_init__(self) -> None:
        string_values = (
            self.case_id,
            self.track_id,
            self.algorithm_version,
            self.config_schema_version,
            self.config_sha256,
            self.stage,
        )
        if any(not isinstance(value, str) or not value for value in string_values):
            raise ValueError("trace identifiers must be non-empty strings")
        if len(self.config_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.config_sha256
        ):
            raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
        try:
            settings = json.loads(self.settings_json)
        except json.JSONDecodeError as error:
            raise ValueError("settings_json must contain canonical JSON") from error
        if not isinstance(settings, dict):
            raise ValueError("settings_json must contain an object")
        canonical = json.dumps(
            settings, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if canonical != self.settings_json:
            raise ValueError("settings_json must use canonical JSON encoding")
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if digest != self.config_sha256:
            raise ValueError("config_sha256 does not match settings_json")
        frames = tuple(self.frames)
        frame_ids = [frame.frame_id for frame in frames]
        if not frames or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("trace frames must be non-empty and unique")
        if self.canonical_shape is not None:
            unknown = set(self.canonical_shape.registered_frame_ids) - set(frame_ids)
            if unknown:
                raise ValueError("canonical shape references unknown registered frames")
        if (
            self.cuboid_fit is not None
            and self.cuboid_fit.status == "candidate"
            and self.canonical_shape is None
        ):
            raise ValueError("cuboid candidate requires a canonical shape")
        object.__setattr__(self, "frames", frames)

    @property
    def settings(self) -> dict[str, object]:
        value = json.loads(self.settings_json)
        if not isinstance(value, dict):
            raise AssertionError("validated trace settings must be an object")
        return value

    @property
    def total_counts(self) -> dict[str, int]:
        return {
            state.name.lower(): sum(frame.count(state) for frame in self.frames)
            for state in EvidenceState
        }

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "contract_version": EVIDENCE_TRACE_CONTRACT,
            "case_id": self.case_id,
            "track_id": self.track_id,
            "algorithm_version": self.algorithm_version,
            "config_schema_version": self.config_schema_version,
            "config_sha256": self.config_sha256,
            "settings": self.settings,
            "stage": self.stage,
            "total_point_state_counts": self.total_counts,
            "frames": [frame.to_summary_dict() for frame in self.frames],
            "canonical_shape": (
                None
                if self.canonical_shape is None
                else self.canonical_shape.to_summary_dict()
            ),
            "cuboid_fit": None
            if self.cuboid_fit is None
            else self.cuboid_fit.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GeometricRefinementRun:
    """Development result containing the public outcome and its point trace."""

    outcome: RefinementOutcome
    trace: GeometricRefinementTrace

    def __post_init__(self) -> None:
        if self.outcome.track_id != self.trace.track_id:
            raise ValueError("run outcome and trace track IDs do not match")


def validate_geometric_trace(
    case: RefinementCase, trace: GeometricRefinementTrace
) -> None:
    """Validate trace ownership and point-index alignment with one case."""

    if trace.case_id != case.case_id or trace.track_id != case.track.track_id:
        raise ValueError("evidence trace does not belong to the refinement case")
    expected = [frame.frame_id for frame in case.frames]
    actual = [frame.frame_id for frame in trace.frames]
    if actual != expected:
        raise ValueError("evidence trace must preserve every input frame")
    for frame, frame_trace in zip(case.frames, trace.frames, strict=True):
        if len(frame_trace.roi_point_indices) and int(
            frame_trace.roi_point_indices[-1]
        ) >= len(frame.points_xyz):
            raise ValueError("evidence trace references a point outside its frame")
        if frame.point_sensor_index is None and frame_trace.represented_sensor_ids:
            raise ValueError("trace cannot claim sensors without point provenance")
        if frame.point_sensor_index is not None:
            represented = tuple(
                frame.sensor_ids[index]
                for index in sorted(
                    set(
                        int(value)
                        for value in frame.point_sensor_index[
                            frame_trace.roi_point_indices
                        ]
                    )
                )
            )
            if represented != frame_trace.represented_sensor_ids:
                raise ValueError("trace represented sensors do not match ROI points")
    if trace.canonical_shape is not None:
        registered = tuple(
            frame.frame_id
            for frame in trace.frames
            if frame.registration is not None
            and frame.registration.status == "registered"
        )
        if trace.canonical_shape.registered_frame_ids != registered:
            raise ValueError(
                "canonical shape registered frames do not match frame traces"
            )


def write_geometric_trace(
    output_dir: str | Path, trace: GeometricRefinementTrace
) -> tuple[Path, Path]:
    """Write JSON metadata and compact point masks without embedding frame points."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray[np.generic]] = {}
    manifest = trace.to_summary_dict()
    frame_rows = manifest["frames"]
    if not isinstance(frame_rows, list):
        raise AssertionError("trace frame summaries must be a list")
    for index, (frame, row) in enumerate(zip(trace.frames, frame_rows, strict=True)):
        index_key = f"frame_{index:06d}_roi_indices"
        state_key = f"frame_{index:06d}_point_states"
        arrays[index_key] = frame.roi_point_indices
        arrays[state_key] = frame.point_states
        if not isinstance(row, dict):
            raise AssertionError("trace frame summary must be an object")
        row["roi_indices_key"] = index_key
        row["point_states_key"] = state_key
    canonical_row = manifest.get("canonical_shape")
    if trace.canonical_shape is not None:
        if not isinstance(canonical_row, dict):
            raise AssertionError("canonical shape summary must be an object")
        point_key = "canonical_points_xyz"
        support_key = "canonical_frame_support_count"
        arrays[point_key] = trace.canonical_shape.points_xyz
        arrays[support_key] = trace.canonical_shape.frame_support_count
        canonical_row["points_key"] = point_key
        canonical_row["frame_support_count_key"] = support_key
    manifest["arrays_path"] = "evidence_masks.npz"

    arrays_path = output / "evidence_masks.npz"
    manifest_path = output / "evidence_trace.json"
    np.savez_compressed(arrays_path, **arrays)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path, arrays_path


def read_geometric_trace(path: str | Path) -> GeometricRefinementTrace:
    """Read a trace from its directory or ``evidence_trace.json`` path."""

    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "evidence_trace.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("contract_version") != EVIDENCE_TRACE_CONTRACT
    ):
        raise ValueError("unsupported geometric evidence trace contract")
    arrays_name = payload.get("arrays_path")
    if not isinstance(arrays_name, str) or not arrays_name:
        raise ValueError("trace arrays_path must be a non-empty string")
    arrays_path = (manifest_path.parent / arrays_name).resolve()
    if arrays_path.parent != manifest_path.parent:
        raise ValueError("trace arrays_path must remain beside its manifest")
    frame_rows = payload.get("frames")
    if not isinstance(frame_rows, list) or not frame_rows:
        raise ValueError("trace frames must be a non-empty list")
    frames: list[FrameEvidenceTrace] = []
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for row in frame_rows:
            if not isinstance(row, dict):
                raise ValueError("trace frame row must be an object")
            index_key = row.get("roi_indices_key")
            state_key = row.get("point_states_key")
            if not isinstance(index_key, str) or not isinstance(state_key, str):
                raise ValueError("trace frame array keys must be strings")
            ground = row.get("ground_plane")
            component = row.get("component")
            registration = row.get("registration")
            sensors = row.get("represented_sensor_ids")
            if not isinstance(sensors, list) or any(
                not isinstance(value, str) for value in sensors
            ):
                raise ValueError("represented_sensor_ids must be a string list")
            frames.append(
                FrameEvidenceTrace(
                    frame_id=_string(row.get("frame_id"), "frame_id"),
                    roi_point_indices=np.asarray(arrays[index_key], dtype=np.int64),
                    point_states=np.asarray(arrays[state_key], dtype=np.uint8),
                    ground_plane=(
                        None
                        if ground is None
                        else GroundPlaneEstimate.from_dict(ground)
                    ),
                    represented_sensor_ids=tuple(sensors),
                    component=(
                        None
                        if component is None
                        else FrameComponentTrace.from_dict(component)
                    ),
                    registration=(
                        None
                        if registration is None
                        else FrameRegistrationTrace.from_dict(registration)
                    ),
                )
            )
        canonical_row = payload.get("canonical_shape")
        canonical_shape = None
        if canonical_row is not None:
            if not isinstance(canonical_row, dict):
                raise ValueError("canonical_shape must be an object")
            point_key = _string(canonical_row.get("points_key"), "points_key")
            support_key = _string(
                canonical_row.get("frame_support_count_key"),
                "frame_support_count_key",
            )
            registered_ids = canonical_row.get("registered_frame_ids")
            if not isinstance(registered_ids, list) or any(
                not isinstance(value, str) for value in registered_ids
            ):
                raise ValueError("registered_frame_ids must be a string list")
            canonical_shape = CanonicalShapeTrace(
                points_xyz=np.asarray(arrays[point_key], dtype=np.float32),
                frame_support_count=np.asarray(arrays[support_key], dtype=np.uint16),
                registered_frame_ids=tuple(registered_ids),
                voxel_size_m=_number(
                    canonical_row.get("voxel_size_m"), "canonical voxel_size_m"
                ),
                iterations=_integer(
                    canonical_row.get("iterations"), "canonical iterations"
                ),
                converged=_boolean(
                    canonical_row.get("converged"), "canonical converged"
                ),
            )
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("trace settings must be an object")
    settings_json = json.dumps(
        settings, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    cuboid_row = payload.get("cuboid_fit")
    cuboid_fit = None if cuboid_row is None else CuboidFitTrace.from_dict(cuboid_row)
    return GeometricRefinementTrace(
        case_id=_string(payload.get("case_id"), "case_id"),
        track_id=_string(payload.get("track_id"), "track_id"),
        algorithm_version=_string(
            payload.get("algorithm_version"), "algorithm_version"
        ),
        config_schema_version=_string(
            payload.get("config_schema_version"), "config_schema_version"
        ),
        config_sha256=_string(payload.get("config_sha256"), "config_sha256"),
        settings_json=settings_json,
        stage=_string(payload.get("stage"), "stage"),
        frames=tuple(frames),
        canonical_shape=canonical_shape,
        cuboid_fit=cuboid_fit,
    )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_number(value: object, name: str) -> float | None:
    return None if value is None else _number(value, name)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _float_triplet(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-value list")
    return tuple(_number(item, f"{name}[]") for item in value)  # type: ignore[return-value]


def _optional_float_triplet(
    value: object, name: str
) -> tuple[float, float, float] | None:
    return None if value is None else _float_triplet(value, name)


def _optional_triplet_list(
    value: tuple[float, float, float] | None,
) -> list[float] | None:
    return None if value is None else list(value)


def _pose_to_dict(pose: Pose3D | None) -> dict[str, list[float]] | None:
    if pose is None:
        return None
    return {
        "translation_xyz": list(pose.translation_xyz),
        "orientation_xyzw": list(pose.orientation_xyzw),
    }


def _optional_pose(value: object, name: str) -> Pose3D | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    orientation = value.get("orientation_xyzw")
    if not isinstance(orientation, list) or len(orientation) != 4:
        raise ValueError(f"{name}.orientation_xyzw must contain four values")
    return Pose3D(
        translation_xyz=_float_triplet(
            value.get("translation_xyz"), f"{name}.translation_xyz"
        ),
        orientation_xyzw=tuple(
            _number(item, f"{name}.orientation_xyzw[]") for item in orientation
        ),  # type: ignore[arg-type]
    )
