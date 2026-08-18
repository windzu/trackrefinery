"""Deterministic upright registration and canonical evidence aggregation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, degrees, pi, radians, sin

import numpy as np
from numpy.typing import NDArray

from trackrefinery.contracts import Pose3D, RefinementCase
from trackrefinery.geometric.settings import (
    GeometricRefinementSettings,
    RegistrationSettings,
)
from trackrefinery.geometric.trace import (
    CanonicalShapeTrace,
    EvidenceState,
    FrameEvidenceTrace,
    FrameRegistrationTrace,
    GeometricRefinementTrace,
    validate_geometric_trace,
)
from trackrefinery.geometry import (
    compose_pose,
    inverse_pose,
    inverse_transform_points,
)


@dataclass(slots=True)
class _RegistrationState:
    frame_index: int
    target_points: NDArray[np.float64]
    rotation: NDArray[np.float64] | None
    translation: NDArray[np.float64] | None
    failure_reason: str | None = None
    initial_rmse_m: float | None = None
    final_rmse_m: float | None = None
    correspondence_count: int = 0

    @property
    def active(self) -> bool:
        return (
            self.failure_reason is None
            and self.rotation is not None
            and self.translation is not None
        )

    def aligned_points(self) -> NDArray[np.float64]:
        if self.rotation is None or self.translation is None:
            raise AssertionError("inactive registration state has no aligned points")
        return self.target_points @ self.rotation.T + self.translation


@dataclass(frozen=True, slots=True)
class _SurfaceReference:
    points: NDArray[np.float64]
    frame_index: NDArray[np.int32]
    normals: NDArray[np.float64]
    normal_quality: NDArray[np.float64]
    tree: object


def register_canonical_shape(
    case: RefinementCase,
    evidence_trace: GeometricRefinementTrace,
    settings: GeometricRefinementSettings | None = None,
) -> GeometricRefinementTrace:
    """Register initial target evidence and build a persistent canonical shape.

    Returned poses are provisional development trace data. They are not a
    successful refinement and must still pass cuboid fitting and observability.
    """

    validate_geometric_trace(case, evidence_trace)
    resolved = settings or GeometricRefinementSettings()
    if evidence_trace.config_sha256 != resolved.sha256:
        raise ValueError("evidence trace and registration settings do not match")
    policy = resolved.registration
    states = [
        _initialize_frame(frame_index, case, frame_trace, policy)
        for frame_index, frame_trace in enumerate(evidence_trace.frames)
    ]
    active = [state for state in states if state.active]
    iteration_count = 0
    converged = False
    if len(active) >= 2:
        _measure_registration(active, policy, initial=True)
        for current_iteration in range(1, policy.maximum_iterations + 1):
            iteration_count = current_iteration
            updates = _registration_updates(active, policy)
            if updates is None:
                break
            maximum_translation = 0.0
            maximum_yaw = 0.0
            for state, (rotation, translation) in zip(active, updates, strict=True):
                if state.rotation is None or state.translation is None:
                    raise AssertionError("active state lost its transformation")
                state.rotation = rotation @ state.rotation
                state.translation = rotation @ state.translation + translation
                maximum_translation = max(
                    maximum_translation, float(np.linalg.norm(translation))
                )
                maximum_yaw = max(
                    maximum_yaw, abs(atan2(rotation[1, 0], rotation[0, 0]))
                )
            if (
                maximum_translation <= policy.convergence_translation_m
                and maximum_yaw <= radians(policy.convergence_yaw_deg)
            ):
                converged = True
                break
        _measure_registration(active, policy, initial=False)

    frame_traces = tuple(
        replace(
            evidence_trace.frames[index],
            registration=_frame_registration_trace(
                case, state, iteration_count, policy
            ),
        )
        for index, state in enumerate(states)
    )
    registered = [state for state in states if _state_is_registered(state, policy)]
    canonical_shape = _build_canonical_shape(
        case,
        registered,
        policy,
        iteration_count=iteration_count,
        converged=converged,
    )
    return replace(
        evidence_trace,
        stage="canonical_registration_v1",
        frames=frame_traces,
        canonical_shape=canonical_shape,
    )


def _initialize_frame(
    frame_index: int,
    case: RefinementCase,
    frame_trace: FrameEvidenceTrace,
    settings: RegistrationSettings,
) -> _RegistrationState:
    frame = case.frames[frame_index]
    coarse_box = case.track.observations[frame_index].coarse_box
    target_positions = np.flatnonzero(
        frame_trace.point_states == EvidenceState.TARGET.value
    )
    target_indices = frame_trace.roi_point_indices[target_positions]
    target_points = inverse_transform_points(
        frame.points_xyz[target_indices], coarse_box.pose
    )
    target_points = _voxel_reduce(target_points, settings.voxel_size_m)[0]
    state = _RegistrationState(frame_index, target_points, None, None)
    if len(target_points) < settings.minimum_target_points:
        state.failure_reason = "insufficient_target_points"
        return state

    initialization_positions = np.flatnonzero(
        (frame_trace.point_states == EvidenceState.TARGET.value)
        | (frame_trace.point_states == EvidenceState.AMBIGUOUS.value)
    )
    initialization_indices = frame_trace.roi_point_indices[initialization_positions]
    initialization_points = inverse_transform_points(
        frame.points_xyz[initialization_indices], coarse_box.pose
    )
    initialization_points = _voxel_reduce(initialization_points, settings.voxel_size_m)[
        0
    ]
    initialized = _shape_initialization(target_points, initialization_points, settings)
    if initialized is None:
        state.failure_reason = "pose_unobservable"
        return state
    state.rotation, state.translation = initialized
    return state


def _shape_initialization(
    target_points: NDArray[np.float64],
    initialization_points: NDArray[np.float64],
    settings: RegistrationSettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    center_xy = np.median(initialization_points[:, :2], axis=0)
    centered_xy = initialization_points[:, :2] - center_xy
    radii = np.linalg.norm(centered_xy, axis=1)
    radial_limit = np.quantile(radii, settings.initialization_radial_trim_fraction)
    planar = centered_xy[radii <= radial_limit]
    if len(planar) < settings.minimum_target_points:
        return None
    covariance = planar.T @ planar
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] <= 0 or (
        eigenvalues[1] / eigenvalues[0] < settings.minimum_planar_anisotropy
    ):
        return None
    major = eigenvectors[:, 1]
    shape_yaw = _half_turn_angle(atan2(major[1], major[0]))
    correction_yaw = float(
        np.clip(
            -shape_yaw,
            -radians(settings.maximum_initial_yaw_correction_deg),
            radians(settings.maximum_initial_yaw_correction_deg),
        )
    )
    rotation = _yaw_rotation(correction_yaw)
    rotated_target = target_points @ rotation.T
    lower = np.quantile(rotated_target, settings.initialization_quantile, axis=0)
    upper = np.quantile(rotated_target, 1.0 - settings.initialization_quantile, axis=0)
    translation = -(lower + upper) / 2.0
    return rotation, translation


def _registration_updates(
    states: list[_RegistrationState], settings: RegistrationSettings
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]] | None:
    aligned = [state.aligned_points() for state in states]
    reference = _surface_reference(states, aligned, settings)
    updates: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for state, source in zip(states, aligned, strict=True):
        step = _point_to_plane_step(source, state.frame_index, reference, settings)
        if step is None:
            state.failure_reason = "pose_unobservable"
            return None
        updates.append((step[0], step[1]))
    return updates


def _measure_registration(
    states: list[_RegistrationState],
    settings: RegistrationSettings,
    *,
    initial: bool,
) -> None:
    aligned = [state.aligned_points() for state in states]
    reference = _surface_reference(states, aligned, settings)
    for state, source in zip(states, aligned, strict=True):
        measurement = _point_to_plane_step(
            source, state.frame_index, reference, settings
        )
        if measurement is None:
            state.failure_reason = "pose_unobservable"
            continue
        _, _, count, rmse = measurement
        if initial:
            state.initial_rmse_m = rmse
        else:
            state.final_rmse_m = rmse
            state.correspondence_count = count


def _point_to_plane_step(
    source: NDArray[np.float64],
    source_frame_index: int,
    reference: _SurfaceReference,
    settings: RegistrationSettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, float] | None:
    neighbor_count = min(
        settings.cross_frame_neighbor_candidates, len(reference.points)
    )
    distances, nearest = reference.tree.query(source, k=neighbor_count, workers=1)
    if neighbor_count == 1:
        distances = distances[:, None]
        nearest = nearest[:, None]
    cross_frame = reference.frame_index[nearest] != source_frame_index
    has_cross_frame = np.any(cross_frame, axis=1)
    choice = np.argmax(cross_frame, axis=1)
    rows = np.arange(len(source))
    distances = distances[rows, choice]
    nearest = nearest[rows, choice]
    matched_normals = reference.normals[nearest]
    residuals = np.einsum(
        "ij,ij->i", matched_normals, source - reference.points[nearest]
    )
    positions = np.flatnonzero(
        has_cross_frame
        & (distances <= settings.maximum_correspondence_distance_m)
        & (reference.normal_quality[nearest] >= settings.normal_min_planarity)
    )
    if len(positions) < settings.minimum_correspondences:
        return None
    keep_count = max(
        settings.minimum_correspondences,
        int(len(positions) * settings.correspondence_trim_fraction),
    )
    ordering = np.argsort(np.abs(residuals[positions]), kind="stable")
    positions = positions[ordering[:keep_count]]
    matched_normals = matched_normals[positions]
    residuals = residuals[positions]
    selected_source = source[positions]
    jacobian = np.column_stack(
        (
            matched_normals,
            -matched_normals[:, 0] * selected_source[:, 1]
            + matched_normals[:, 1] * selected_source[:, 0],
        )
    )
    absolute = np.abs(residuals)
    weights = np.minimum(1.0, settings.huber_delta_m / np.maximum(absolute, 1e-12))
    weights *= reference.normal_quality[nearest[positions]]
    normal_matrix = jacobian.T @ (weights[:, None] * jacobian)
    normal_matrix += np.eye(4, dtype=np.float64) * settings.step_regularization
    gradient = jacobian.T @ (-weights * residuals)
    try:
        delta = np.linalg.solve(normal_matrix, gradient)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(delta).all():
        return None
    xy_norm = float(np.linalg.norm(delta[:2]))
    if xy_norm > settings.maximum_xy_step_m:
        delta[:2] *= settings.maximum_xy_step_m / xy_norm
    delta[2] = np.clip(delta[2], -settings.maximum_z_step_m, settings.maximum_z_step_m)
    delta[3] = np.clip(
        delta[3],
        -radians(settings.maximum_yaw_step_deg),
        radians(settings.maximum_yaw_step_deg),
    )
    rotation = _yaw_rotation(float(delta[3]))
    translation = np.asarray(delta[:3], dtype=np.float64)
    rmse = float(np.sqrt(np.average(residuals * residuals, weights=weights)))
    return rotation, translation, len(positions), rmse


def _surface_reference(
    states: list[_RegistrationState],
    aligned: list[NDArray[np.float64]],
    settings: RegistrationSettings,
) -> _SurfaceReference:
    points = np.concatenate(aligned)
    frame_index = np.concatenate(
        [
            np.full(len(frame_points), state.frame_index, dtype=np.int32)
            for state, frame_points in zip(states, aligned, strict=True)
        ]
    )
    order = np.lexsort((frame_index, points[:, 2], points[:, 1], points[:, 0]))
    points = points[order]
    frame_index = frame_index[order]
    normals, quality = _surface_normals(points, settings.normal_neighbor_count)
    return _SurfaceReference(
        points=points,
        frame_index=frame_index,
        normals=normals,
        normal_quality=quality,
        tree=_ckdtree_type()(points),
    )


def _surface_normals(
    points: NDArray[np.float64], neighbor_count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    tree_type = _ckdtree_type()
    count = min(neighbor_count, len(points))
    if count < 4:
        return np.zeros_like(points), np.zeros(len(points), dtype=np.float64)
    tree = tree_type(points)
    _, neighborhoods = tree.query(points, k=count, workers=1)
    neighborhoods = np.atleast_2d(neighborhoods)
    local = points[neighborhoods]
    local -= local.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", local, local)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    total = eigenvalues.sum(axis=1)
    quality = np.divide(
        eigenvalues[:, 0],
        total,
        out=np.ones_like(total),
        where=total > 0,
    )
    quality = 1.0 - quality
    return normals, quality


def _build_canonical_shape(
    case: RefinementCase,
    states: list[_RegistrationState],
    settings: RegistrationSettings,
    *,
    iteration_count: int,
    converged: bool,
) -> CanonicalShapeTrace | None:
    if len(states) < max(2, settings.canonical_minimum_frame_support):
        return None
    aligned = [state.aligned_points() for state in states]
    tree_type = _ckdtree_type()
    trees = [tree_type(points) for points in aligned]
    persistent_groups: list[NDArray[np.float64]] = []
    support_groups: list[NDArray[np.uint16]] = []
    for index, points in enumerate(aligned):
        support = np.ones(len(points), dtype=np.uint16)
        for other, tree in enumerate(trees):
            if other == index:
                continue
            distances, _ = tree.query(points, k=1, workers=1)
            support += (distances <= settings.canonical_support_radius_m).astype(
                np.uint16
            )
        keep = support >= settings.canonical_minimum_frame_support
        persistent_groups.append(points[keep])
        support_groups.append(support[keep])
    persistent = np.concatenate(persistent_groups)
    support = np.concatenate(support_groups)
    if not len(persistent):
        return None
    points, frame_support = _voxel_reduce(persistent, settings.voxel_size_m, support)
    if frame_support is None:
        raise AssertionError("canonical voxel reduction lost support counts")
    frame_ids = tuple(case.frames[state.frame_index].frame_id for state in states)
    return CanonicalShapeTrace(
        points_xyz=points.astype(np.float32),
        frame_support_count=frame_support.astype(np.uint16),
        registered_frame_ids=frame_ids,
        voxel_size_m=settings.voxel_size_m,
        iterations=max(1, iteration_count),
        converged=converged,
    )


def _frame_registration_trace(
    case: RefinementCase,
    state: _RegistrationState,
    iteration_count: int,
    settings: RegistrationSettings,
) -> FrameRegistrationTrace:
    if not _state_is_registered(state, settings):
        reason = state.failure_reason or "pose_unobservable"
        return FrameRegistrationTrace(
            status="insufficient_evidence",
            reason_codes=(reason,),
            canonical_from_coarse=None,
            candidate_pose_annotation=None,
            iterations=iteration_count,
            correspondence_count=state.correspondence_count,
            initial_rmse_m=state.initial_rmse_m,
            final_rmse_m=state.final_rmse_m,
            translation_correction_m=None,
            yaw_correction_deg=None,
        )
    if state.rotation is None or state.translation is None:
        raise AssertionError("registered state has no transformation")
    correction_yaw = atan2(state.rotation[1, 0], state.rotation[0, 0])
    canonical_from_coarse = Pose3D(
        translation_xyz=tuple(float(value) for value in state.translation),
        orientation_xyzw=_yaw_quaternion(correction_yaw),
    )
    frame = case.frames[state.frame_index]
    coarse = case.track.observations[state.frame_index].coarse_box.pose
    world_from_coarse = compose_pose(frame.world_from_annotation, coarse)
    world_from_candidate = compose_pose(
        world_from_coarse, inverse_pose(canonical_from_coarse)
    )
    candidate_pose = compose_pose(
        inverse_pose(frame.world_from_annotation), world_from_candidate
    )
    inverse_correction = inverse_pose(canonical_from_coarse)
    return FrameRegistrationTrace(
        status="registered",
        reason_codes=(),
        canonical_from_coarse=canonical_from_coarse,
        candidate_pose_annotation=candidate_pose,
        iterations=iteration_count,
        correspondence_count=state.correspondence_count,
        initial_rmse_m=state.initial_rmse_m,
        final_rmse_m=state.final_rmse_m,
        translation_correction_m=float(
            np.linalg.norm(inverse_correction.translation_xyz)
        ),
        yaw_correction_deg=abs(degrees(correction_yaw)),
    )


def _state_is_registered(
    state: _RegistrationState, settings: RegistrationSettings
) -> bool:
    return (
        state.active
        and state.final_rmse_m is not None
        and state.correspondence_count >= settings.minimum_correspondences
    )


def _voxel_reduce(
    points: NDArray[np.float64],
    voxel_size_m: float,
    support: NDArray[np.uint16] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.uint16] | None]:
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return points.reshape(0, 3), None if support is None else support[:0]
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
    ordered_support = None if support is None else np.asarray(support)[order]
    starts = np.concatenate(
        (
            np.asarray([0]),
            np.flatnonzero(np.any(voxels[1:] != voxels[:-1], axis=1)) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.asarray([len(points)])))
    representatives = np.asarray(
        [
            np.median(points[start:end], axis=0)
            for start, end in zip(starts, ends, strict=True)
        ]
    )
    reduced_support = (
        None
        if ordered_support is None
        else np.asarray(
            [
                ordered_support[start:end].max()
                for start, end in zip(starts, ends, strict=True)
            ],
            dtype=np.uint16,
        )
    )
    return representatives, reduced_support


def _ckdtree_type() -> type:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "JointCuboidRefiner registration requires trackrefinery[geometric]"
        ) from error
    return cKDTree


def _yaw_rotation(yaw: float) -> NDArray[np.float64]:
    cosine = cos(yaw)
    sine = sin(yaw)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, sin(yaw / 2.0), cos(yaw / 2.0))


def _half_turn_angle(angle: float) -> float:
    return float((angle + pi / 2.0) % pi - pi / 2.0)
