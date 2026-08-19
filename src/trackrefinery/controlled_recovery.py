"""Controlled Stage 3 pose-recovery diagnostics over frozen components."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from math import cos, degrees, isfinite, radians, sin, sqrt
from pathlib import Path

import numpy as np

from trackrefinery.component_consensus.aggregation import (
    aggregate_geometry_components,
)
from trackrefinery.component_consensus.components import select_object_components
from trackrefinery.component_consensus.settings import ComponentConsensusSettings
from trackrefinery.contracts import Box3D, Pose3D, RefinementCase, TrackObservation
from trackrefinery.geometric.trace import (
    FrameRole,
    GeometricRefinementTrace,
    validate_geometric_trace,
)
from trackrefinery.geometry import (
    angle_difference,
    compose_pose,
    inverse_pose,
    yaw_from_quaternion,
)

CONTROLLED_RECOVERY_CONTRACT = "trackrefinery-controlled-recovery-v2"

AggregationBackend = Callable[
    [RefinementCase, GeometricRefinementTrace, ComponentConsensusSettings],
    GeometricRefinementTrace,
]


@dataclass(frozen=True, slots=True)
class ControlledPerturbationProfile:
    """Maximum deterministic drift injected on either side of the anchor."""

    name: str
    maximum_translation_m: float
    maximum_yaw_deg: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("profile name must be a non-empty string")
        translation = float(self.maximum_translation_m)
        yaw = float(self.maximum_yaw_deg)
        if not isfinite(translation) or translation <= 0:
            raise ValueError("maximum_translation_m must be finite and positive")
        if not isfinite(yaw) or yaw <= 0:
            raise ValueError("maximum_yaw_deg must be finite and positive")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "maximum_translation_m", translation)
        object.__setattr__(self, "maximum_yaw_deg", yaw)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "maximum_translation_m": self.maximum_translation_m,
            "maximum_yaw_deg": self.maximum_yaw_deg,
        }


DEFAULT_CONTROLLED_PERTURBATION_PROFILES = (
    ControlledPerturbationProfile("mild", 0.05, 0.5),
    ControlledPerturbationProfile("medium", 0.10, 1.0),
    ControlledPerturbationProfile("strong", 0.15, 2.0),
)


@dataclass(frozen=True, slots=True)
class ControlledFramePerturbation:
    frame_id: str
    phase: float
    translation_xy_m: tuple[float, float]
    yaw_deg: float
    perturbed_pose_annotation: Pose3D

    @property
    def translation_m(self) -> float:
        return float(np.linalg.norm(self.translation_xy_m))

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "phase": self.phase,
            "translation_xy_m": list(self.translation_xy_m),
            "translation_m": self.translation_m,
            "yaw_deg": self.yaw_deg,
        }


@dataclass(frozen=True, slots=True)
class ControlledFrameRecovery:
    frame_id: str
    phase: float
    injected_translation_m: float
    injected_yaw_deg: float
    output_status: str
    output_translation_error_m: float
    output_yaw_error_deg: float
    translation_recovery_fraction: float | None
    yaw_recovery_fraction: float | None
    equivariant_output_translation_error_m: float
    equivariant_output_yaw_error_deg: float
    equivariant_translation_recovery_fraction: float | None
    equivariant_yaw_recovery_fraction: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "phase": self.phase,
            "injected_translation_m": self.injected_translation_m,
            "injected_yaw_deg": self.injected_yaw_deg,
            "output_status": self.output_status,
            "output_translation_error_m": self.output_translation_error_m,
            "output_yaw_error_deg": self.output_yaw_error_deg,
            "translation_recovery_fraction": self.translation_recovery_fraction,
            "yaw_recovery_fraction": self.yaw_recovery_fraction,
            "equivariant_output_translation_error_m": (
                self.equivariant_output_translation_error_m
            ),
            "equivariant_output_yaw_error_deg": (self.equivariant_output_yaw_error_deg),
            "equivariant_translation_recovery_fraction": (
                self.equivariant_translation_recovery_fraction
            ),
            "equivariant_yaw_recovery_fraction": (
                self.equivariant_yaw_recovery_fraction
            ),
        }


@dataclass(frozen=True, slots=True)
class ControlledRecoveryReport:
    case_id: str
    track_id: str
    algorithm_variant: str
    profile: ControlledPerturbationProfile
    anchor_frame_id: str
    geometry_frame_count: int
    perturbed_frame_count: int
    registered_frame_count: int
    retained_coarse_frame_count: int
    unavailable_frame_count: int
    input_translation_rms_m: float
    output_translation_rms_m: float
    input_yaw_rms_deg: float
    output_yaw_rms_deg: float
    translation_rms_reduction_fraction: float
    yaw_rms_reduction_fraction: float
    improved_translation_frame_fraction: float
    improved_yaw_frame_fraction: float
    equivariant_output_translation_rms_m: float
    equivariant_output_yaw_rms_deg: float
    equivariant_translation_rms_reduction_fraction: float
    equivariant_yaw_rms_reduction_fraction: float
    equivariant_improved_translation_frame_fraction: float
    equivariant_improved_yaw_frame_fraction: float
    frames: tuple[ControlledFrameRecovery, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": CONTROLLED_RECOVERY_CONTRACT,
            "reference_semantics": "frozen_model_track_proxy_not_gold",
            "equivariant_reference_semantics": (
                "same_algorithm_unperturbed_output_not_gold"
            ),
            "case_id": self.case_id,
            "track_id": self.track_id,
            "algorithm_variant": self.algorithm_variant,
            "profile": self.profile.to_dict(),
            "anchor_frame_id": self.anchor_frame_id,
            "geometry_frame_count": self.geometry_frame_count,
            "perturbed_frame_count": self.perturbed_frame_count,
            "registered_frame_count": self.registered_frame_count,
            "retained_coarse_frame_count": self.retained_coarse_frame_count,
            "unavailable_frame_count": self.unavailable_frame_count,
            "input_translation_rms_m": self.input_translation_rms_m,
            "output_translation_rms_m": self.output_translation_rms_m,
            "input_yaw_rms_deg": self.input_yaw_rms_deg,
            "output_yaw_rms_deg": self.output_yaw_rms_deg,
            "translation_rms_reduction_fraction": (
                self.translation_rms_reduction_fraction
            ),
            "yaw_rms_reduction_fraction": self.yaw_rms_reduction_fraction,
            "improved_translation_frame_fraction": (
                self.improved_translation_frame_fraction
            ),
            "improved_yaw_frame_fraction": self.improved_yaw_frame_fraction,
            "equivariant_output_translation_rms_m": (
                self.equivariant_output_translation_rms_m
            ),
            "equivariant_output_yaw_rms_deg": self.equivariant_output_yaw_rms_deg,
            "equivariant_translation_rms_reduction_fraction": (
                self.equivariant_translation_rms_reduction_fraction
            ),
            "equivariant_yaw_rms_reduction_fraction": (
                self.equivariant_yaw_rms_reduction_fraction
            ),
            "equivariant_improved_translation_frame_fraction": (
                self.equivariant_improved_translation_frame_fraction
            ),
            "equivariant_improved_yaw_frame_fraction": (
                self.equivariant_improved_yaw_frame_fraction
            ),
            "frames": [frame.to_dict() for frame in self.frames],
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
class ControlledRecoveryRun:
    """Input/output material retained for numeric and visual review."""

    reference_case: RefinementCase
    perturbed_case: RefinementCase
    component_trace: GeometricRefinementTrace
    reference_output_trace: GeometricRefinementTrace
    output_trace: GeometricRefinementTrace
    perturbations: tuple[ControlledFramePerturbation, ...]
    report: ControlledRecoveryReport


def run_controlled_recovery(
    case: RefinementCase,
    *,
    profile: ControlledPerturbationProfile | None = None,
    settings: ComponentConsensusSettings | None = None,
    component_trace: GeometricRefinementTrace | None = None,
    aggregation_backend: AggregationBackend | None = None,
    algorithm_variant: str = "sequential_v2_1",
) -> ControlledRecoveryRun:
    """Measure Stage 3 recovery around a frozen model-track proxy reference.

    Component selection is performed before perturbation and remains fixed. The
    original coarse poses are used only by this evaluator and are not present in
    the perturbed case passed to anchored aggregation.
    """

    resolved_profile = profile or DEFAULT_CONTROLLED_PERTURBATION_PROFILES[-1]
    resolved_settings = settings or ComponentConsensusSettings()
    if not isinstance(algorithm_variant, str) or not algorithm_variant.strip():
        raise ValueError("algorithm_variant must be a non-empty string")
    variant = algorithm_variant.strip()
    supplied = component_trace or select_object_components(case, resolved_settings)
    validate_geometric_trace(case, supplied)
    if supplied.config_sha256 != resolved_settings.sha256:
        raise ValueError("component trace and recovery settings do not match")
    selected = _component_only_trace(supplied)
    backend = aggregation_backend or aggregate_geometry_components
    reference_output_trace = backend(case, selected, resolved_settings)
    aggregation = reference_output_trace.anchored_aggregation
    if aggregation is None or aggregation.anchor_frame_id is None:
        raise ValueError("controlled recovery requires an anchored geometry track")
    perturbations = _build_perturbations(
        case,
        selected,
        aggregation.anchor_frame_id,
        resolved_profile,
        reference_output_trace,
    )
    perturbed_case = _perturb_case(case, perturbations)
    output_trace = backend(perturbed_case, selected, resolved_settings)
    report = _evaluate_recovery(
        case,
        perturbed_case,
        reference_output_trace,
        output_trace,
        perturbations,
        resolved_profile,
        aggregation.anchor_frame_id,
        variant,
    )
    return ControlledRecoveryRun(
        reference_case=case,
        perturbed_case=perturbed_case,
        component_trace=selected,
        reference_output_trace=reference_output_trace,
        output_trace=output_trace,
        perturbations=perturbations,
        report=report,
    )


def _component_only_trace(
    trace: GeometricRefinementTrace,
) -> GeometricRefinementTrace:
    """Remove all original-pose Stage 3 products before the perturbed run."""

    return replace(
        trace,
        stage="component_selection_v2",
        frames=tuple(replace(frame, registration=None) for frame in trace.frames),
        anchored_aggregation=None,
        canonical_shape=None,
        cuboid_fit=None,
    )


def _build_perturbations(
    case: RefinementCase,
    trace: GeometricRefinementTrace,
    anchor_frame_id: str,
    profile: ControlledPerturbationProfile,
    reference_output_trace: GeometricRefinementTrace,
) -> tuple[ControlledFramePerturbation, ...]:
    geometry_indices = [
        index
        for index, frame in enumerate(trace.frames)
        if frame.component is not None
        and frame.component.frame_role is FrameRole.GEOMETRY
    ]
    frame_indices = {frame.frame_id: index for index, frame in enumerate(case.frames)}
    anchor_index = frame_indices[anchor_frame_id]
    if anchor_index not in geometry_indices:
        raise ValueError("aggregation anchor is not a geometry frame")
    timestamps = {index: case.frames[index].timestamp_ns for index in geometry_indices}
    anchor_timestamp = timestamps[anchor_index]
    left_span = anchor_timestamp - min(timestamps.values())
    right_span = max(timestamps.values()) - anchor_timestamp
    direction = (cos(radians(30.0)), sin(radians(30.0)))
    values: list[ControlledFramePerturbation] = []
    for index in geometry_indices:
        timestamp = timestamps[index]
        if timestamp < anchor_timestamp:
            phase = (timestamp - anchor_timestamp) / max(1, left_span)
        elif timestamp > anchor_timestamp:
            phase = (timestamp - anchor_timestamp) / max(1, right_span)
        else:
            phase = 0.0
        translation_xy = (
            profile.maximum_translation_m * phase * direction[0],
            profile.maximum_translation_m * phase * direction[1],
        )
        yaw_deg = profile.maximum_yaw_deg * phase
        frame = case.frames[index]
        input_reference = case.track.observations[index].coarse_box.pose
        registration = reference_output_trace.frames[index].registration
        reference = (
            registration.candidate_pose_annotation
            if registration is not None
            and registration.candidate_pose_annotation is not None
            else input_reference
        )
        if phase == 0.0:
            perturbed_annotation = reference
        else:
            reference_world = compose_pose(frame.world_from_annotation, reference)
            delta = Pose3D(
                (translation_xy[0], translation_xy[1], 0.0),
                _yaw_quaternion(radians(yaw_deg)),
            )
            perturbed_world = compose_pose(reference_world, delta)
            perturbed_annotation = compose_pose(
                inverse_pose(frame.world_from_annotation), perturbed_world
            )
        values.append(
            ControlledFramePerturbation(
                frame_id=frame.frame_id,
                phase=float(phase),
                translation_xy_m=translation_xy,
                yaw_deg=float(yaw_deg),
                perturbed_pose_annotation=perturbed_annotation,
            )
        )
    return tuple(values)


def _perturb_case(
    case: RefinementCase,
    perturbations: tuple[ControlledFramePerturbation, ...],
) -> RefinementCase:
    by_frame = {value.frame_id: value for value in perturbations}
    observations = tuple(
        _perturb_observation(observation, by_frame.get(observation.frame_id))
        for observation in case.track.observations
    )
    return replace(case, track=replace(case.track, observations=observations))


def _perturb_observation(
    observation: TrackObservation,
    perturbation: ControlledFramePerturbation | None,
) -> TrackObservation:
    if perturbation is None:
        return observation
    box = observation.coarse_box
    pose = perturbation.perturbed_pose_annotation
    return replace(
        observation,
        coarse_box=Box3D(pose.translation_xyz, box.size_lwh, pose.orientation_xyzw),
    )


def _evaluate_recovery(
    reference_case: RefinementCase,
    perturbed_case: RefinementCase,
    reference_output_trace: GeometricRefinementTrace,
    output_trace: GeometricRefinementTrace,
    perturbations: tuple[ControlledFramePerturbation, ...],
    profile: ControlledPerturbationProfile,
    anchor_frame_id: str,
    algorithm_variant: str,
) -> ControlledRecoveryReport:
    indices = {
        frame.frame_id: index for index, frame in enumerate(reference_case.frames)
    }
    frame_results: list[ControlledFrameRecovery] = []
    for perturbation in perturbations:
        if perturbation.frame_id == anchor_frame_id:
            continue
        index = indices[perturbation.frame_id]
        frame = reference_case.frames[index]
        reference_pose = reference_case.track.observations[index].coarse_box.pose
        input_pose = perturbed_case.track.observations[index].coarse_box.pose
        reference_registration = reference_output_trace.frames[index].registration
        algorithm_reference_pose = (
            reference_registration.candidate_pose_annotation
            if reference_registration is not None
            and reference_registration.candidate_pose_annotation is not None
            else reference_pose
        )
        registration = output_trace.frames[index].registration
        output_pose = (
            registration.candidate_pose_annotation
            if registration is not None
            and registration.candidate_pose_annotation is not None
            else input_pose
        )
        input_translation = perturbation.translation_m
        input_yaw = abs(perturbation.yaw_deg)
        output_translation, output_yaw = _pose_error(
            frame.world_from_annotation, reference_pose, output_pose
        )
        equivariant_translation, equivariant_yaw = _pose_error(
            frame.world_from_annotation, algorithm_reference_pose, output_pose
        )
        frame_results.append(
            ControlledFrameRecovery(
                frame_id=frame.frame_id,
                phase=perturbation.phase,
                injected_translation_m=input_translation,
                injected_yaw_deg=input_yaw,
                output_status=(
                    "unavailable" if registration is None else registration.status
                ),
                output_translation_error_m=output_translation,
                output_yaw_error_deg=output_yaw,
                translation_recovery_fraction=_recovery_fraction(
                    input_translation, output_translation
                ),
                yaw_recovery_fraction=_recovery_fraction(input_yaw, output_yaw),
                equivariant_output_translation_error_m=equivariant_translation,
                equivariant_output_yaw_error_deg=equivariant_yaw,
                equivariant_translation_recovery_fraction=_recovery_fraction(
                    input_translation, equivariant_translation
                ),
                equivariant_yaw_recovery_fraction=_recovery_fraction(
                    input_yaw, equivariant_yaw
                ),
            )
        )
    if not frame_results:
        raise ValueError("controlled recovery requires a non-anchor geometry frame")

    input_translation_rms = _rms(
        [frame.injected_translation_m for frame in frame_results]
    )
    output_translation_rms = _rms(
        [frame.output_translation_error_m for frame in frame_results]
    )
    input_yaw_rms = _rms([frame.injected_yaw_deg for frame in frame_results])
    output_yaw_rms = _rms([frame.output_yaw_error_deg for frame in frame_results])
    equivariant_translation_rms = _rms(
        [frame.equivariant_output_translation_error_m for frame in frame_results]
    )
    equivariant_yaw_rms = _rms(
        [frame.equivariant_output_yaw_error_deg for frame in frame_results]
    )
    statuses = [frame.output_status for frame in frame_results]
    return ControlledRecoveryReport(
        case_id=reference_case.case_id,
        track_id=reference_case.track.track_id,
        algorithm_variant=algorithm_variant,
        profile=profile,
        anchor_frame_id=anchor_frame_id,
        geometry_frame_count=len(perturbations),
        perturbed_frame_count=len(frame_results),
        registered_frame_count=statuses.count("registered"),
        retained_coarse_frame_count=statuses.count("retained_coarse"),
        unavailable_frame_count=sum(
            status not in {"registered", "retained_coarse"} for status in statuses
        ),
        input_translation_rms_m=input_translation_rms,
        output_translation_rms_m=output_translation_rms,
        input_yaw_rms_deg=input_yaw_rms,
        output_yaw_rms_deg=output_yaw_rms,
        translation_rms_reduction_fraction=_recovery_fraction(
            input_translation_rms, output_translation_rms
        )
        or 0.0,
        yaw_rms_reduction_fraction=(
            _recovery_fraction(input_yaw_rms, output_yaw_rms) or 0.0
        ),
        improved_translation_frame_fraction=sum(
            frame.output_translation_error_m < frame.injected_translation_m - 1e-9
            for frame in frame_results
        )
        / len(frame_results),
        improved_yaw_frame_fraction=sum(
            frame.output_yaw_error_deg < frame.injected_yaw_deg - 1e-9
            for frame in frame_results
        )
        / len(frame_results),
        equivariant_output_translation_rms_m=equivariant_translation_rms,
        equivariant_output_yaw_rms_deg=equivariant_yaw_rms,
        equivariant_translation_rms_reduction_fraction=(
            _recovery_fraction(input_translation_rms, equivariant_translation_rms)
            or 0.0
        ),
        equivariant_yaw_rms_reduction_fraction=(
            _recovery_fraction(input_yaw_rms, equivariant_yaw_rms) or 0.0
        ),
        equivariant_improved_translation_frame_fraction=sum(
            frame.equivariant_output_translation_error_m
            < frame.injected_translation_m - 1e-9
            for frame in frame_results
        )
        / len(frame_results),
        equivariant_improved_yaw_frame_fraction=sum(
            frame.equivariant_output_yaw_error_deg < frame.injected_yaw_deg - 1e-9
            for frame in frame_results
        )
        / len(frame_results),
        frames=tuple(frame_results),
    )


def _pose_error(
    world_from_annotation: Pose3D,
    reference_annotation: Pose3D,
    candidate_annotation: Pose3D,
) -> tuple[float, float]:
    reference_world = compose_pose(world_from_annotation, reference_annotation)
    candidate_world = compose_pose(world_from_annotation, candidate_annotation)
    translation = np.asarray(candidate_world.translation_xyz) - np.asarray(
        reference_world.translation_xyz
    )
    yaw = abs(
        degrees(
            angle_difference(
                yaw_from_quaternion(candidate_world.orientation_xyzw),
                yaw_from_quaternion(reference_world.orientation_xyzw),
            )
        )
    )
    return float(np.linalg.norm(translation[:2])), float(yaw)


def _recovery_fraction(input_error: float, output_error: float) -> float | None:
    if input_error <= 1e-12:
        return None
    return float(1.0 - output_error / input_error)


def _rms(values: list[float]) -> float:
    return float(sqrt(sum(value * value for value in values) / len(values)))


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, sin(yaw / 2), cos(yaw / 2))
