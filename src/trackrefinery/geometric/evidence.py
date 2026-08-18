"""Deterministic initial ROI, ground, and point-state selection."""

from __future__ import annotations

from math import atan, degrees

import numpy as np
from numpy.typing import NDArray

from trackrefinery.contracts import Box3D, FrameCloud, RefinementCase
from trackrefinery.geometric.settings import (
    GEOMETRIC_ALGORITHM_VERSION,
    GEOMETRIC_CONFIG_SCHEMA_VERSION,
    EvidenceSelectionSettings,
    GeometricRefinementSettings,
)
from trackrefinery.geometric.trace import (
    EvidenceState,
    FrameEvidenceTrace,
    GeometricRefinementTrace,
    GroundPlaneEstimate,
)
from trackrefinery.geometry import inverse_transform_points


def select_initial_evidence(
    case: RefinementCase,
    settings: GeometricRefinementSettings | None = None,
) -> GeometricRefinementTrace:
    """Classify a conservative initial point ROI for every observed frame.

    This is initialization evidence, not final instance segmentation. Later
    optimization rounds must recompute ownership using the refined state.
    """

    resolved = settings or GeometricRefinementSettings()
    frames = tuple(
        _select_frame_evidence(frame, observation.coarse_box, resolved.evidence)
        for frame, observation in zip(case.frames, case.track.observations, strict=True)
    )
    return GeometricRefinementTrace(
        case_id=case.case_id,
        track_id=case.track.track_id,
        algorithm_version=GEOMETRIC_ALGORITHM_VERSION,
        config_schema_version=GEOMETRIC_CONFIG_SCHEMA_VERSION,
        config_sha256=resolved.sha256,
        settings_json=resolved.canonical_json(),
        stage="initial_evidence_v1",
        frames=frames,
    )


def _select_frame_evidence(
    frame: FrameCloud,
    coarse_box: Box3D,
    settings: EvidenceSelectionSettings,
) -> FrameEvidenceTrace:
    local = inverse_transform_points(frame.points_xyz, coarse_box.pose)
    half_size = np.asarray(coarse_box.size_lwh, dtype=np.float64) / 2.0
    roi_half_size = half_size + np.asarray(settings.roi_margin_xyz_m)
    roi_mask = np.all(np.abs(local) <= roi_half_size + 1e-9, axis=1)
    roi_indices = np.flatnonzero(roi_mask).astype(np.int64, copy=False)
    roi_points = np.asarray(frame.points_xyz[roi_indices], dtype=np.float64)
    roi_local = local[roi_indices]

    ground = _estimate_ground_plane(
        roi_points,
        roi_local,
        bottom_local_z=-half_size[2],
        settings=settings,
    )
    ground_mask = _classify_ground(
        roi_points,
        roi_local,
        ground,
        bottom_local_z=-half_size[2],
        settings=settings,
    )
    target_half_size = half_size + np.asarray(settings.target_allowance_xyz_m)
    ambiguity_half_size = half_size + np.asarray(settings.ambiguity_margin_xyz_m)
    target_mask = np.all(np.abs(roi_local) <= target_half_size + 1e-9, axis=1)
    ambiguous_mask = np.all(np.abs(roi_local) <= ambiguity_half_size + 1e-9, axis=1)

    states = np.full(len(roi_indices), EvidenceState.BACKGROUND.value, dtype=np.uint8)
    states[ambiguous_mask] = EvidenceState.AMBIGUOUS.value
    states[target_mask] = EvidenceState.TARGET.value
    states[ground_mask] = EvidenceState.GROUND.value
    represented_sensors = _represented_sensors(frame, roi_indices)
    return FrameEvidenceTrace(
        frame_id=frame.frame_id,
        roi_point_indices=roi_indices,
        point_states=states,
        ground_plane=ground,
        represented_sensor_ids=represented_sensors,
    )


def _estimate_ground_plane(
    points: NDArray[np.float64],
    local: NDArray[np.float64],
    *,
    bottom_local_z: float,
    settings: EvidenceSelectionSettings,
) -> GroundPlaneEstimate | None:
    if not len(points):
        return None
    candidate_mask = (
        local[:, 2] >= bottom_local_z - settings.ground_candidate_below_bottom_m
    ) & (local[:, 2] <= bottom_local_z + settings.ground_candidate_above_bottom_m)
    candidates = points[candidate_mask]
    if len(candidates) < settings.ground_min_candidates:
        return None
    design = np.column_stack(
        (candidates[:, 0], candidates[:, 1], np.ones(len(candidates)))
    )
    coefficients = _solve_plane(design, candidates[:, 2])
    if coefficients is None:
        return None
    for _ in range(settings.ground_irls_iterations):
        residuals = candidates[:, 2] - design @ coefficients
        absolute = np.abs(residuals)
        weights = np.ones(len(candidates), dtype=np.float64)
        outside = absolute > settings.ground_huber_delta_m
        weights[outside] = settings.ground_huber_delta_m / absolute[outside]
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_z = candidates[:, 2] * np.sqrt(weights)
        updated = _solve_plane(weighted_design, weighted_z)
        if updated is None:
            return None
        coefficients = updated

    residuals = candidates[:, 2] - design @ coefficients
    inliers = np.abs(residuals) <= settings.ground_distance_tolerance_m
    inlier_count = int(np.count_nonzero(inliers))
    if (
        inlier_count < settings.ground_min_inliers
        or inlier_count / len(candidates) < settings.ground_min_inlier_fraction
    ):
        return None
    final = _solve_plane(design[inliers], candidates[inliers, 2])
    if final is None:
        return None
    final_residuals = candidates[inliers, 2] - design[inliers] @ final
    rmse = float(np.sqrt(np.mean(final_residuals * final_residuals)))
    slope = float(np.hypot(final[0], final[1]))
    tilt = degrees(atan(slope))
    if rmse > settings.ground_max_rmse_m or tilt > settings.ground_max_tilt_deg:
        return None
    normal = np.asarray([-final[0], -final[1], 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return GroundPlaneEstimate(
        z_from_xyc=tuple(float(value) for value in final),
        normal_xyz=tuple(float(value) for value in normal),
        candidate_count=len(candidates),
        inlier_count=inlier_count,
        rmse_m=rmse,
        tilt_deg=tilt,
    )


def _solve_plane(
    design: NDArray[np.float64], z: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3 or not np.isfinite(coefficients).all():
        return None
    return coefficients


def _classify_ground(
    points: NDArray[np.float64],
    local: NDArray[np.float64],
    plane: GroundPlaneEstimate | None,
    *,
    bottom_local_z: float,
    settings: EvidenceSelectionSettings,
) -> NDArray[np.bool_]:
    if plane is None or not len(points):
        return np.zeros(len(points), dtype=bool)
    a, b, c = plane.z_from_xyc
    residual = np.abs(points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c))
    below_upper_band = (
        local[:, 2] <= bottom_local_z + settings.ground_candidate_above_bottom_m
    )
    return (residual <= settings.ground_distance_tolerance_m) & below_upper_band


def _represented_sensors(
    frame: FrameCloud, roi_indices: NDArray[np.int64]
) -> tuple[str, ...]:
    if frame.point_sensor_index is None or not len(roi_indices):
        return ()
    indices = sorted(set(int(value) for value in frame.point_sensor_index[roi_indices]))
    return tuple(frame.sensor_ids[index] for index in indices)
