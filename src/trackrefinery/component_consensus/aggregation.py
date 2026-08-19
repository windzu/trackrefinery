"""Deterministic Stage 3 anchored aggregation for dense clean components."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, degrees, radians

import numpy as np
from numpy.typing import NDArray

from trackrefinery.component_consensus.settings import ComponentConsensusSettings
from trackrefinery.contracts import Pose3D, RefinementCase
from trackrefinery.geometric.trace import (
    AggregateSharpnessTrace,
    AnchoredAggregationTrace,
    CanonicalShapeTrace,
    EvidenceState,
    FrameEvidenceTrace,
    FrameRegistrationTrace,
    FrameRole,
    GeometricRefinementTrace,
    validate_geometric_trace,
)
from trackrefinery.geometry import compose_pose, inverse_pose, inverse_transform_points


@dataclass(slots=True)
class _FrameState:
    index: int
    points: NDArray[np.float64]
    reduced_points: NDArray[np.float64]
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    accepted: bool = False
    retained_reason: str | None = None
    failure_reason: str | None = None
    iterations: int = 0
    correspondence_count: int = 0
    initial_rmse_m: float | None = None
    final_rmse_m: float | None = None

    def aligned(self, *, reduced: bool = False) -> NDArray[np.float64]:
        points = self.reduced_points if reduced else self.points
        return points @ self.rotation.T + self.translation


@dataclass(frozen=True, slots=True)
class _Correspondences:
    source_positions: NDArray[np.int64]
    target_positions: NDArray[np.int64]
    distances: NDArray[np.float64]
    overlap_fraction: float
    rmse_m: float


def aggregate_geometry_components(
    case: RefinementCase,
    component_trace: GeometricRefinementTrace,
    settings: ComponentConsensusSettings | None = None,
) -> GeometricRefinementTrace:
    """Align reliable components one at a time without moving accepted frames."""

    validate_geometric_trace(case, component_trace)
    resolved = settings or ComponentConsensusSettings()
    if component_trace.config_sha256 != resolved.sha256:
        raise ValueError("component trace and aggregation settings do not match")
    states = _geometry_states(case, component_trace, resolved)
    if len(states) < resolved.track_minimum_geometry_frames:
        attempted = tuple(case.frames[state.index].frame_id for state in states)
        frames = _with_rejected_geometry_frames(
            component_trace.frames, states, "insufficient_geometry_frames"
        )
        return replace(
            component_trace,
            stage="anchored_component_aggregation_v2",
            frames=frames,
            anchored_aggregation=AnchoredAggregationTrace(
                status="insufficient_evidence",
                reason_codes=("insufficient_geometry_frames",),
                anchor_frame_id=None,
                attempted_frame_ids=attempted,
                accepted_frame_ids=(),
                rejected_frame_ids=attempted,
                baseline_sharpness=None,
                candidate_sharpness=None,
                maximum_correction_velocity_mps=None,
                maximum_correction_acceleration_mps2=None,
                maximum_correction_yaw_rate_degps=None,
            ),
        )

    order = _aggregation_order(case, component_trace, states, resolved)
    anchor = order[0]
    anchor.accepted = True
    anchor.retained_reason = "anchor_frame"
    anchor.iterations = 1
    anchor.correspondence_count = len(anchor.reduced_points)
    anchor.initial_rmse_m = 0.0
    anchor.final_rmse_m = 0.0
    accepted = [anchor]
    for state in order[1:]:
        _register_candidate(state, accepted, resolved)
        if state.accepted:
            accepted.append(state)

    accepted = _protect_track_level_regressions(case, accepted, resolved)
    accepted_by_index = {state.index: state for state in accepted}
    rejected = [state for state in states if state.index not in accepted_by_index]
    attempted_ids = tuple(case.frames[state.index].frame_id for state in order)
    accepted_order_ids = tuple(case.frames[state.index].frame_id for state in accepted)
    rejected_ids = tuple(case.frames[state.index].frame_id for state in rejected)
    if len(accepted) < resolved.track_minimum_geometry_frames:
        aggregation = AnchoredAggregationTrace(
            status="insufficient_evidence",
            reason_codes=("insufficient_accepted_geometry_frames",),
            anchor_frame_id=case.frames[anchor.index].frame_id,
            attempted_frame_ids=attempted_ids,
            accepted_frame_ids=accepted_order_ids,
            rejected_frame_ids=rejected_ids,
            baseline_sharpness=None,
            candidate_sharpness=None,
            maximum_correction_velocity_mps=None,
            maximum_correction_acceleration_mps2=None,
            maximum_correction_yaw_rate_degps=None,
        )
        canonical = None
    else:
        baseline = _aggregate_sharpness(
            [state.reduced_points for state in accepted], resolved
        )
        candidate = _aggregate_sharpness(
            [state.aligned(reduced=True) for state in accepted], resolved
        )
        velocity, acceleration, yaw_rate = _trajectory_metrics(case, accepted)
        aggregation = AnchoredAggregationTrace(
            status="candidate",
            reason_codes=(),
            anchor_frame_id=case.frames[anchor.index].frame_id,
            attempted_frame_ids=attempted_ids,
            accepted_frame_ids=accepted_order_ids,
            rejected_frame_ids=rejected_ids,
            baseline_sharpness=baseline,
            candidate_sharpness=candidate,
            maximum_correction_velocity_mps=velocity,
            maximum_correction_acceleration_mps2=acceleration,
            maximum_correction_yaw_rate_degps=yaw_rate,
        )
        canonical = _canonical_shape(case, accepted, resolved)
        if canonical is None:
            aggregation = replace(
                aggregation,
                status="insufficient_evidence",
                reason_codes=("insufficient_persistent_component_support",),
            )

    frames = tuple(
        replace(
            frame_trace,
            registration=(
                _registration_trace(case, accepted_by_index[index])
                if index in accepted_by_index
                else _rejected_registration(
                    next(state for state in rejected if state.index == index)
                )
                if any(state.index == index for state in rejected)
                else None
            ),
        )
        for index, frame_trace in enumerate(component_trace.frames)
    )
    return replace(
        component_trace,
        stage="anchored_component_aggregation_v2",
        frames=frames,
        anchored_aggregation=aggregation,
        canonical_shape=canonical,
    )


def _geometry_states(
    case: RefinementCase,
    trace: GeometricRefinementTrace,
    settings: ComponentConsensusSettings,
) -> list[_FrameState]:
    states: list[_FrameState] = []
    for index, (frame, observation, frame_trace) in enumerate(
        zip(case.frames, case.track.observations, trace.frames, strict=True)
    ):
        component = frame_trace.component
        if component is None or component.frame_role is not FrameRole.GEOMETRY:
            continue
        positions = np.flatnonzero(
            frame_trace.point_states == EvidenceState.TARGET.value
        )
        point_indices = frame_trace.roi_point_indices[positions]
        points = inverse_transform_points(
            frame.points_xyz[point_indices], observation.coarse_box.pose
        )
        reduced = _voxel_reduce(points, settings.aggregation_voxel_size_m)
        states.append(
            _FrameState(
                index=index,
                points=points,
                reduced_points=reduced,
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
            )
        )
    return states


def _aggregation_order(
    case: RefinementCase,
    trace: GeometricRefinementTrace,
    states: list[_FrameState],
    settings: ComponentConsensusSettings,
) -> list[_FrameState]:
    quality = {
        state.index: _frame_quality(trace.frames[state.index]) for state in states
    }
    maximum = max(quality.values())
    anchor_candidates = [
        state
        for state in states
        if quality[state.index]
        >= maximum * settings.aggregation_anchor_minimum_relative_quality
    ]
    timestamps = [case.frames[state.index].timestamp_ns for state in states]
    midpoint = (min(timestamps) + max(timestamps)) / 2
    anchor = min(
        anchor_candidates,
        key=lambda state: (
            abs(case.frames[state.index].timestamp_ns - midpoint),
            -quality[state.index],
            case.frames[state.index].frame_id,
        ),
    )
    remaining = sorted(
        (state for state in states if state is not anchor),
        key=lambda state: (
            -quality[state.index],
            abs(case.frames[state.index].timestamp_ns - midpoint),
            case.frames[state.index].frame_id,
        ),
    )
    return [anchor, *remaining]


def _frame_quality(frame_trace: FrameEvidenceTrace) -> float:
    component = frame_trace.component
    if component is None:
        raise AssertionError("geometry frame is missing component diagnostics")
    return float(
        np.sqrt(component.selected_point_count * component.selected_voxel_count)
        * (component.component_dominance or 0.0)
        * (component.resolution_stability_iou or 0.0)
    )


def _register_candidate(
    state: _FrameState,
    accepted: list[_FrameState],
    settings: ComponentConsensusSettings,
) -> None:
    reference = _voxel_reduce(
        np.concatenate([item.aligned(reduced=True) for item in accepted]),
        settings.aggregation_voxel_size_m,
    )
    tree = _ckdtree_type()(reference)
    initial = _correspondences(state.reduced_points, reference, tree, settings)
    if initial is None:
        state.failure_reason = "insufficient_component_overlap"
        return
    state.initial_rmse_m = initial.rmse_m
    for iteration in range(1, settings.aggregation_maximum_iterations + 1):
        state.iterations = iteration
        aligned = state.aligned(reduced=True)
        correspondences = _correspondences(aligned, reference, tree, settings)
        if correspondences is None:
            state.failure_reason = "component_alignment_underconstrained"
            return
        step = _upright_step(aligned, reference, correspondences, settings)
        if step is None:
            state.failure_reason = "component_alignment_underconstrained"
            return
        rotation, translation, yaw = step
        state.rotation = rotation @ state.rotation
        state.translation = rotation @ state.translation + translation
        total_xy = float(np.linalg.norm(state.translation[:2]))
        total_yaw = abs(atan2(state.rotation[1, 0], state.rotation[0, 0]))
        if (
            total_xy > settings.aggregation_maximum_xy_correction_m + 1e-9
            or total_yaw
            > radians(settings.aggregation_maximum_yaw_correction_deg) + 1e-9
        ):
            state.failure_reason = "component_alignment_exceeds_correction_bound"
            return
        if float(
            np.linalg.norm(translation[:2])
        ) <= settings.aggregation_convergence_translation_m and abs(yaw) <= radians(
            settings.aggregation_convergence_yaw_deg
        ):
            break

    final = _correspondences(state.aligned(reduced=True), reference, tree, settings)
    if final is None:
        state.failure_reason = "insufficient_component_overlap"
        return
    state.correspondence_count = len(final.distances)
    state.final_rmse_m = final.rmse_m
    translation_m = float(np.linalg.norm(state.translation[:2]))
    yaw_deg = abs(degrees(atan2(state.rotation[1, 0], state.rotation[0, 0])))
    no_op = (
        translation_m <= settings.aggregation_noop_translation_m
        and yaw_deg <= settings.aggregation_noop_yaw_deg
    )
    absolute_improvement = initial.rmse_m - final.rmse_m
    relative_improvement = absolute_improvement / max(initial.rmse_m, 1e-12)
    before = _aggregate_sharpness(
        [*[item.aligned(reduced=True) for item in accepted], state.reduced_points],
        settings,
    )
    after = _aggregate_sharpness(
        [
            *[item.aligned(reduced=True) for item in accepted],
            state.aligned(reduced=True),
        ],
        settings,
    )
    sharpness_safe = _sharpness_non_regressing(before, after, settings)
    improvement = (
        absolute_improvement >= settings.aggregation_minimum_rmse_improvement_m
        and relative_improvement
        >= settings.aggregation_minimum_relative_rmse_improvement
    )
    if no_op:
        _retain_coarse(state, "alignment_not_needed", initial)
    elif not improvement or not sharpness_safe:
        reason = (
            "candidate_regressed" if not sharpness_safe else "no_material_improvement"
        )
        _retain_coarse(state, reason, initial)
    else:
        state.accepted = True


def _retain_coarse(state: _FrameState, reason: str, baseline: _Correspondences) -> None:
    state.rotation = np.eye(3, dtype=np.float64)
    state.translation = np.zeros(3, dtype=np.float64)
    state.accepted = True
    state.retained_reason = reason
    state.correspondence_count = len(baseline.distances)
    state.final_rmse_m = baseline.rmse_m
    state.iterations = max(1, state.iterations)


def _correspondences(
    source: NDArray[np.float64],
    reference: NDArray[np.float64],
    tree: object,
    settings: ComponentConsensusSettings,
) -> _Correspondences | None:
    distances, nearest = tree.query(source, k=1, workers=1)
    positions = np.flatnonzero(
        distances <= settings.aggregation_maximum_correspondence_distance_m
    )
    overlap = len(positions) / max(1, len(source))
    if (
        len(positions) < settings.aggregation_minimum_correspondences
        or overlap < settings.aggregation_minimum_overlap_fraction
    ):
        return None
    keep_count = max(
        settings.aggregation_minimum_correspondences,
        int(len(positions) * settings.aggregation_correspondence_trim_fraction),
    )
    ordering = np.argsort(distances[positions], kind="stable")
    positions = positions[ordering[:keep_count]]
    kept_distances = np.asarray(distances[positions], dtype=np.float64)
    return _Correspondences(
        source_positions=positions.astype(np.int64, copy=False),
        target_positions=np.asarray(nearest[positions], dtype=np.int64),
        distances=kept_distances,
        overlap_fraction=overlap,
        rmse_m=float(np.sqrt(np.mean(np.square(kept_distances)))),
    )


def _upright_step(
    aligned: NDArray[np.float64],
    reference: NDArray[np.float64],
    correspondences: _Correspondences,
    settings: ComponentConsensusSettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float] | None:
    source = aligned[correspondences.source_positions, :2]
    target = reference[correspondences.target_positions, :2]
    weights = np.minimum(
        1.0,
        settings.aggregation_huber_delta_m
        / np.maximum(correspondences.distances, 1e-12),
    )
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return None
    source_center = np.sum(source * weights[:, None], axis=0) / weight_sum
    target_center = np.sum(target * weights[:, None], axis=0) / weight_sum
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = source_centered.T @ (weights[:, None] * target_centered)
    try:
        left, singular, right_t = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    if singular[-1] <= 1e-12 or singular[0] / singular[-1] > 1e6:
        return None
    planar_rotation = right_t.T @ left.T
    if np.linalg.det(planar_rotation) < 0:
        right_t[-1, :] *= -1
        planar_rotation = right_t.T @ left.T
    raw_yaw = atan2(planar_rotation[1, 0], planar_rotation[0, 0])
    yaw_limit = radians(settings.aggregation_maximum_yaw_step_deg)
    yaw = float(
        np.clip(raw_yaw * settings.aggregation_step_gain, -yaw_limit, yaw_limit)
    )
    rotation = _yaw_rotation(yaw)
    desired_translation = target_center - source_center @ rotation[:2, :2].T
    xy = desired_translation * settings.aggregation_step_gain
    norm = float(np.linalg.norm(xy))
    if norm > settings.aggregation_maximum_xy_step_m:
        xy *= settings.aggregation_maximum_xy_step_m / norm
    translation = np.asarray((xy[0], xy[1], 0.0), dtype=np.float64)
    return rotation, translation, yaw


def _protect_track_level_regressions(
    case: RefinementCase,
    accepted: list[_FrameState],
    settings: ComponentConsensusSettings,
) -> list[_FrameState]:
    if len(accepted) < 2:
        return accepted
    baseline = _aggregate_sharpness(
        [state.reduced_points for state in accepted], settings
    )
    candidate = _aggregate_sharpness(
        [state.aligned(reduced=True) for state in accepted], settings
    )
    velocity, acceleration, yaw_rate = _trajectory_metrics(case, accepted)
    trajectory_safe = (
        velocity <= settings.aggregation_maximum_correction_velocity_mps
        and acceleration <= settings.aggregation_maximum_correction_acceleration_mps2
        and yaw_rate <= settings.aggregation_maximum_correction_yaw_rate_degps
    )
    if _sharpness_non_regressing(baseline, candidate, settings) and trajectory_safe:
        return accepted
    reason = "trajectory_regression" if not trajectory_safe else "aggregate_regression"
    for state in accepted:
        if state.retained_reason == "anchor_frame":
            continue
        _force_retain_coarse(state, reason, settings)
    return accepted


def _force_retain_coarse(
    state: _FrameState,
    reason: str,
    settings: ComponentConsensusSettings,
) -> None:
    state.rotation = np.eye(3, dtype=np.float64)
    state.translation = np.zeros(3, dtype=np.float64)
    state.accepted = True
    state.retained_reason = reason
    state.correspondence_count = max(
        state.correspondence_count,
        settings.aggregation_minimum_correspondences,
    )
    state.final_rmse_m = state.initial_rmse_m
    state.iterations = max(1, state.iterations)


def _sharpness_non_regressing(
    baseline: AggregateSharpnessTrace,
    candidate: AggregateSharpnessTrace,
    settings: ComponentConsensusSettings,
) -> bool:
    axis_safe = all(
        after <= before + settings.aggregation_maximum_axis_spread_regression_m
        for before, after in zip(
            baseline.robust_spread_xyz_m,
            candidate.robust_spread_xyz_m,
            strict=True,
        )
    )
    concentration_safe = (
        candidate.voxel_concentration
        >= baseline.voxel_concentration
        - settings.aggregation_maximum_concentration_regression
    )
    return axis_safe and concentration_safe


def _aggregate_sharpness(
    groups: list[NDArray[np.float64]], settings: ComponentConsensusSettings
) -> AggregateSharpnessTrace:
    points = np.concatenate(groups)
    quantile = settings.aggregation_sharpness_quantile
    lower = np.quantile(points, quantile, axis=0)
    upper = np.quantile(points, 1.0 - quantile, axis=0)
    spread = upper - lower
    voxel_sets = [
        {
            tuple(int(value) for value in cell)
            for cell in np.floor(
                group / settings.aggregation_sharpness_voxel_size_m
            ).astype(np.int64)
        }
        for group in groups
    ]
    support: dict[tuple[int, int, int], int] = {}
    for cells in voxel_sets:
        for cell in cells:
            support[cell] = support.get(cell, 0) + 1
    incidence_count = sum(len(cells) for cells in voxel_sets)
    supported_incidence_count = sum(
        support[cell] for cell in support if support[cell] >= 2
    )
    concentration = supported_incidence_count / max(1, incidence_count)
    rmse = _cross_frame_rmse(groups, settings)
    return AggregateSharpnessTrace(
        robust_spread_xyz_m=tuple(float(value) for value in spread),
        xy_area_m2=float(spread[0] * spread[1]),
        voxel_concentration=float(concentration),
        cross_frame_rmse_m=rmse,
        point_count=len(points),
    )


def _cross_frame_rmse(
    groups: list[NDArray[np.float64]], settings: ComponentConsensusSettings
) -> float:
    if len(groups) < 2:
        return 0.0
    distances: list[NDArray[np.float64]] = []
    tree_type = _ckdtree_type()
    for index, group in enumerate(groups):
        reference = np.concatenate(
            [other for other_index, other in enumerate(groups) if other_index != index]
        )
        values, _ = tree_type(reference).query(group, k=1, workers=1)
        kept = values[values <= settings.aggregation_maximum_correspondence_distance_m]
        if len(kept):
            distances.append(kept)
    if not distances:
        return settings.aggregation_maximum_correspondence_distance_m
    combined = np.concatenate(distances)
    keep_count = max(
        1, int(len(combined) * settings.aggregation_correspondence_trim_fraction)
    )
    combined = np.sort(combined, kind="stable")[:keep_count]
    return float(np.sqrt(np.mean(np.square(combined))))


def _trajectory_metrics(
    case: RefinementCase, states: list[_FrameState]
) -> tuple[float, float, float]:
    ordered = sorted(states, key=lambda state: case.frames[state.index].timestamp_ns)
    origin_timestamp_ns = case.frames[ordered[0].index].timestamp_ns
    corrections: list[NDArray[np.float64]] = []
    yaws: list[float] = []
    timestamps: list[float] = []
    for state in ordered:
        frame = case.frames[state.index]
        coarse = case.track.observations[state.index].coarse_box.pose
        candidate = _candidate_pose(case, state)
        coarse_world = compose_pose(frame.world_from_annotation, coarse)
        candidate_world = compose_pose(frame.world_from_annotation, candidate)
        corrections.append(
            np.asarray(candidate_world.translation_xyz)
            - np.asarray(coarse_world.translation_xyz)
        )
        correction_pose = inverse_pose(coarse_world)
        correction_pose = compose_pose(correction_pose, candidate_world)
        yaws.append(
            atan2(
                2
                * (
                    correction_pose.orientation_xyzw[3]
                    * correction_pose.orientation_xyzw[2]
                    + correction_pose.orientation_xyzw[0]
                    * correction_pose.orientation_xyzw[1]
                ),
                1
                - 2
                * (
                    correction_pose.orientation_xyzw[1] ** 2
                    + correction_pose.orientation_xyzw[2] ** 2
                ),
            )
        )
        timestamps.append((frame.timestamp_ns - origin_timestamp_ns) * 1e-9)
    velocities: list[NDArray[np.float64]] = []
    yaw_rates: list[float] = []
    intervals: list[float] = []
    for index in range(1, len(ordered)):
        delta_t = timestamps[index] - timestamps[index - 1]
        if delta_t <= 0:
            raise ValueError("frame timestamps must be strictly increasing")
        intervals.append(delta_t)
        velocities.append((corrections[index] - corrections[index - 1]) / delta_t)
        yaw_delta = atan2(
            np.sin(yaws[index] - yaws[index - 1]),
            np.cos(yaws[index] - yaws[index - 1]),
        )
        yaw_rates.append(degrees(yaw_delta) / delta_t)
    accelerations = [
        (velocities[index] - velocities[index - 1])
        / ((intervals[index] + intervals[index - 1]) / 2)
        for index in range(1, len(velocities))
    ]
    maximum_velocity = max(
        (float(np.linalg.norm(value[:2])) for value in velocities), default=0.0
    )
    maximum_acceleration = max(
        (float(np.linalg.norm(value[:2])) for value in accelerations), default=0.0
    )
    maximum_yaw_rate = max((abs(value) for value in yaw_rates), default=0.0)
    return maximum_velocity, maximum_acceleration, maximum_yaw_rate


def _canonical_shape(
    case: RefinementCase,
    accepted: list[_FrameState],
    settings: ComponentConsensusSettings,
) -> CanonicalShapeTrace | None:
    cells: dict[tuple[int, int, int], list[NDArray[np.float64]]] = {}
    support: dict[tuple[int, int, int], set[int]] = {}
    for state in accepted:
        points = state.aligned(reduced=True)
        voxel_indices = np.floor(points / settings.aggregation_voxel_size_m).astype(
            np.int64
        )
        for point, voxel in zip(points, voxel_indices, strict=True):
            key = tuple(int(value) for value in voxel)
            cells.setdefault(key, []).append(point)
            support.setdefault(key, set()).add(state.index)
    persistent = sorted(
        key for key, frame_ids in support.items() if len(frame_ids) >= 2
    )
    if not persistent:
        return None
    points = np.asarray([np.median(cells[key], axis=0) for key in persistent])
    counts = np.asarray([len(support[key]) for key in persistent], dtype=np.uint16)
    accepted_indices = {state.index for state in accepted}
    frame_ids = tuple(
        frame.frame_id
        for index, frame in enumerate(case.frames)
        if index in accepted_indices
    )
    return CanonicalShapeTrace(
        points_xyz=points.astype(np.float32),
        frame_support_count=counts,
        registered_frame_ids=frame_ids,
        voxel_size_m=settings.aggregation_voxel_size_m,
        iterations=max(1, max(state.iterations for state in accepted)),
        converged=True,
    )


def _registration_trace(
    case: RefinementCase, state: _FrameState
) -> FrameRegistrationTrace:
    yaw = atan2(state.rotation[1, 0], state.rotation[0, 0])
    canonical_from_coarse = Pose3D(
        tuple(float(value) for value in state.translation), _yaw_quaternion(yaw)
    )
    candidate = _candidate_pose(case, state)
    return FrameRegistrationTrace(
        status="retained_coarse" if state.retained_reason is not None else "registered",
        reason_codes=(state.retained_reason,) if state.retained_reason else (),
        canonical_from_coarse=canonical_from_coarse,
        candidate_pose_annotation=candidate,
        iterations=max(1, state.iterations),
        correspondence_count=max(1, state.correspondence_count),
        initial_rmse_m=state.initial_rmse_m or 0.0,
        final_rmse_m=state.final_rmse_m or 0.0,
        translation_correction_m=float(np.linalg.norm(state.translation[:2])),
        yaw_correction_deg=abs(degrees(yaw)),
    )


def _candidate_pose(case: RefinementCase, state: _FrameState) -> Pose3D:
    frame = case.frames[state.index]
    coarse = case.track.observations[state.index].coarse_box.pose
    if np.array_equal(state.rotation, np.eye(3)) and not np.any(state.translation):
        return coarse
    correction = Pose3D(
        tuple(float(value) for value in state.translation),
        _yaw_quaternion(atan2(state.rotation[1, 0], state.rotation[0, 0])),
    )
    world_from_coarse = compose_pose(frame.world_from_annotation, coarse)
    world_from_candidate = compose_pose(world_from_coarse, inverse_pose(correction))
    return compose_pose(inverse_pose(frame.world_from_annotation), world_from_candidate)


def _rejected_registration(state: _FrameState) -> FrameRegistrationTrace:
    return FrameRegistrationTrace(
        status="insufficient_evidence",
        reason_codes=(state.failure_reason or "component_alignment_failed",),
        canonical_from_coarse=None,
        candidate_pose_annotation=None,
        iterations=state.iterations,
        correspondence_count=state.correspondence_count,
        initial_rmse_m=state.initial_rmse_m,
        final_rmse_m=state.final_rmse_m,
        translation_correction_m=None,
        yaw_correction_deg=None,
    )


def _with_rejected_geometry_frames(
    frames: tuple[FrameEvidenceTrace, ...],
    states: list[_FrameState],
    reason: str,
) -> tuple[FrameEvidenceTrace, ...]:
    indices = {state.index for state in states}
    return tuple(
        replace(
            frame,
            registration=(
                FrameRegistrationTrace(
                    status="insufficient_evidence",
                    reason_codes=(reason,),
                    canonical_from_coarse=None,
                    candidate_pose_annotation=None,
                    iterations=0,
                    correspondence_count=0,
                    initial_rmse_m=None,
                    final_rmse_m=None,
                    translation_correction_m=None,
                    yaw_correction_deg=None,
                )
                if index in indices
                else None
            ),
        )
        for index, frame in enumerate(frames)
    )


def _voxel_reduce(
    points: NDArray[np.float64], voxel_size_m: float
) -> NDArray[np.float64]:
    values = np.asarray(points, dtype=np.float64)
    if not len(values):
        return values.reshape(0, 3)
    voxels = np.floor(values / voxel_size_m).astype(np.int64)
    order = np.lexsort(
        (
            values[:, 2],
            values[:, 1],
            values[:, 0],
            voxels[:, 2],
            voxels[:, 1],
            voxels[:, 0],
        )
    )
    values = values[order]
    voxels = voxels[order]
    starts = np.concatenate(
        (
            np.asarray([0]),
            np.flatnonzero(np.any(voxels[1:] != voxels[:-1], axis=1)) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.asarray([len(values)])))
    return np.asarray(
        [
            np.median(values[start:end], axis=0)
            for start, end in zip(starts, ends, strict=True)
        ]
    )


def _yaw_rotation(yaw: float) -> NDArray[np.float64]:
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))


def _ckdtree_type() -> type:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "ComponentConsensusRefiner aggregation requires trackrefinery[geometric]"
        ) from error
    return cKDTree
