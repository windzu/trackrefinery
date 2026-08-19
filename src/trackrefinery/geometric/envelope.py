"""Visible-envelope cuboid fitting with deterministic evidence alternation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from trackrefinery.contracts import RefinementCase
from trackrefinery.geometric.registration import register_canonical_shape
from trackrefinery.geometric.settings import (
    EnvelopeFittingSettings,
    GeometricRefinementSettings,
)
from trackrefinery.geometric.trace import (
    CuboidFitTrace,
    EvidenceState,
    GeometricRefinementTrace,
    validate_geometric_trace,
)
from trackrefinery.geometry import inverse_transform_points


def fit_alternating_envelope(
    case: RefinementCase,
    registration_trace: GeometricRefinementTrace,
    settings: GeometricRefinementSettings | None = None,
) -> GeometricRefinementTrace:
    """Alternate point ownership, registration, and visible-envelope fitting."""

    validate_geometric_trace(case, registration_trace)
    resolved = settings or GeometricRefinementSettings()
    if registration_trace.config_sha256 != resolved.sha256:
        raise ValueError("registration trace and envelope settings do not match")
    policy = resolved.envelope
    current = registration_trace
    previous: CuboidFitTrace | None = None
    for alternation in range(1, policy.maximum_alternations + 1):
        reassigned = _reassign_evidence(case, current, previous, policy)
        current = register_canonical_shape(case, reassigned, resolved)
        candidate = _fit_visible_envelope(current, policy, alternation)
        if candidate.status != "candidate":
            return replace(
                current,
                stage="alternating_envelope_v1",
                cuboid_fit=candidate,
            )
        converged = _envelope_converged(previous, candidate, policy)
        candidate = replace(candidate, converged=converged)
        current = replace(
            current,
            stage="alternating_envelope_v1",
            cuboid_fit=candidate,
        )
        if converged:
            return current
        previous = candidate
    return current


def _reassign_evidence(
    case: RefinementCase,
    trace: GeometricRefinementTrace,
    cuboid: CuboidFitTrace | None,
    settings: EnvelopeFittingSettings,
) -> GeometricRefinementTrace:
    registered: list[tuple[int, NDArray[np.int64], NDArray[np.float64]]] = []
    for index, (frame, frame_trace) in enumerate(
        zip(case.frames, trace.frames, strict=True)
    ):
        registration = frame_trace.registration
        if registration is None or registration.candidate_pose_annotation is None:
            continue
        positions = np.flatnonzero(
            (frame_trace.point_states == EvidenceState.TARGET.value)
            | (frame_trace.point_states == EvidenceState.AMBIGUOUS.value)
        ).astype(np.int64)
        indices = frame_trace.roi_point_indices[positions]
        local = inverse_transform_points(
            frame.points_xyz[indices], registration.candidate_pose_annotation
        )
        registered.append((index, positions, local))

    tree_type = _ckdtree_type()
    trees = [tree_type(local) for _, _, local in registered]
    updated = list(trace.frames)
    for tree_index, (frame_index, positions, local) in enumerate(registered):
        support = np.ones(len(local), dtype=np.uint16)
        for other_index, tree in enumerate(trees):
            if tree_index == other_index:
                continue
            distances, _ = tree.query(local, k=1, workers=1)
            support += (distances <= settings.reassignment_support_radius_m).astype(
                np.uint16
            )
        persistent = support >= settings.reassignment_minimum_frame_support
        target_region, ambiguity_region = _envelope_regions(local, cuboid, settings)
        old = trace.frames[frame_index]
        states = old.point_states.copy()
        non_ground = states != EvidenceState.GROUND.value
        states[non_ground] = EvidenceState.BACKGROUND.value
        states[positions[ambiguity_region]] = EvidenceState.AMBIGUOUS.value
        states[positions[persistent & target_region]] = EvidenceState.TARGET.value
        updated[frame_index] = replace(
            old,
            point_states=states,
            registration=None,
        )
    return replace(
        trace,
        stage="reassigned_evidence_v1",
        frames=tuple(updated),
        canonical_shape=None,
        cuboid_fit=None,
    )


def _envelope_regions(
    points: NDArray[np.float64],
    cuboid: CuboidFitTrace | None,
    settings: EnvelopeFittingSettings,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    if (
        cuboid is None
        or cuboid.status != "candidate"
        or cuboid.canonical_size_lwh is None
        or cuboid.center_in_registration_xyz is None
    ):
        all_points = np.ones(len(points), dtype=bool)
        return all_points, all_points
    center = np.asarray(cuboid.center_in_registration_xyz)
    half_size = np.asarray(cuboid.canonical_size_lwh) / 2.0
    offsets = np.abs(points - center)
    target = np.all(
        offsets <= half_size + settings.target_envelope_allowance_m,
        axis=1,
    )
    ambiguous = np.all(
        offsets <= half_size + settings.ambiguity_envelope_allowance_m,
        axis=1,
    )
    return target, ambiguous


def _fit_visible_envelope(
    trace: GeometricRefinementTrace,
    settings: EnvelopeFittingSettings,
    alternation: int,
) -> CuboidFitTrace:
    if trace.canonical_shape is None:
        return _insufficient("insufficient_target_points", alternation)
    failed_frames = [
        frame.frame_id
        for frame in trace.frames
        if frame.registration is None
        or frame.registration.candidate_pose_annotation is None
    ]
    if failed_frames:
        return _insufficient("pose_unobservable", alternation)
    if any(frame.ground_plane is None for frame in trace.frames):
        return _insufficient("ground_support_unavailable", alternation)

    points = np.asarray(trace.canonical_shape.points_xyz, dtype=np.float64)
    x_lower, x_lower_count = _tail_face(points[:, 0], lower=True, settings=settings)
    x_upper, x_upper_count = _tail_face(points[:, 0], lower=False, settings=settings)
    y_lower, y_lower_count = _tail_face(points[:, 1], lower=True, settings=settings)
    y_upper, y_upper_count = _tail_face(points[:, 1], lower=False, settings=settings)
    z_upper, z_upper_count = _tail_face(points[:, 2], lower=False, settings=settings)
    ground_values: list[float] = []
    ground_support_count = 0
    for frame_trace in trace.frames:
        registration = frame_trace.registration
        ground = frame_trace.ground_plane
        if registration is None or ground is None:
            raise AssertionError("validated envelope frame lost pose or ground")
        pose = registration.candidate_pose_annotation
        if pose is None:
            raise AssertionError("registered envelope frame lost candidate pose")
        x, y, _ = pose.translation_xyz
        a, b, c = ground.z_from_xyc
        point = np.asarray([[x, y, a * x + b * y + c]], dtype=np.float64)
        ground_values.append(float(inverse_transform_points(point, pose)[0, 2]))
        ground_support_count += ground.inlier_count
    z_lower = float(np.median(ground_values))
    counts = (
        x_lower_count,
        x_upper_count,
        y_lower_count,
        y_upper_count,
        ground_support_count,
        z_upper_count,
    )
    if any(value < settings.minimum_face_points for value in counts):
        return _insufficient("insufficient_view_coverage", alternation, counts)

    lower = np.asarray([x_lower, y_lower, z_lower], dtype=np.float64)
    upper = np.asarray([x_upper, y_upper, z_upper], dtype=np.float64)
    size = upper - lower
    minimum = np.asarray(settings.minimum_size_lwh_m)
    maximum = np.asarray(settings.maximum_size_lwh_m)
    if np.any(size < minimum) or np.any(size > maximum):
        return _insufficient("unsupported_object_geometry", alternation, counts)
    center = (lower + upper) / 2.0
    return CuboidFitTrace(
        status="candidate",
        reason_codes=(),
        canonical_size_lwh=tuple(float(value) for value in size),
        center_in_registration_xyz=tuple(float(value) for value in center),
        lower_envelope_xyz=tuple(float(value) for value in lower),
        upper_envelope_xyz=tuple(float(value) for value in upper),
        face_support_counts=counts,
        alternations=alternation,
        converged=False,
    )


def _tail_face(
    values: NDArray[np.float64],
    *,
    lower: bool,
    settings: EnvelopeFittingSettings,
) -> tuple[float, int]:
    quantile = (
        settings.face_tail_fraction if lower else 1.0 - settings.face_tail_fraction
    )
    boundary = float(np.quantile(values, quantile))
    tail = values[values <= boundary] if lower else values[values >= boundary]
    location = float(np.median(tail))
    count = int(np.count_nonzero(np.abs(values - location) <= settings.face_band_m))
    return location, count


def _envelope_converged(
    previous: CuboidFitTrace | None,
    current: CuboidFitTrace,
    settings: EnvelopeFittingSettings,
) -> bool:
    if (
        previous is None
        or previous.canonical_size_lwh is None
        or previous.center_in_registration_xyz is None
        or current.canonical_size_lwh is None
        or current.center_in_registration_xyz is None
    ):
        return False
    size_change = np.max(
        np.abs(
            np.asarray(previous.canonical_size_lwh)
            - np.asarray(current.canonical_size_lwh)
        )
    )
    center_change = np.linalg.norm(
        np.asarray(previous.center_in_registration_xyz)
        - np.asarray(current.center_in_registration_xyz)
    )
    return bool(
        size_change <= settings.dimension_convergence_m
        and center_change <= settings.center_convergence_m
    )


def _insufficient(
    reason: str,
    alternation: int,
    counts: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
) -> CuboidFitTrace:
    return CuboidFitTrace(
        status="insufficient_evidence",
        reason_codes=(reason,),
        canonical_size_lwh=None,
        center_in_registration_xyz=None,
        lower_envelope_xyz=None,
        upper_envelope_xyz=None,
        face_support_counts=counts,
        alternations=alternation,
        converged=False,
    )


def _ckdtree_type() -> type:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "JointCuboidRefiner envelope fitting requires trackrefinery[geometric]"
        ) from error
    return cKDTree
