"""Shared deterministic local-ground estimation for geometric backends."""

from __future__ import annotations

from math import atan, degrees
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from trackrefinery.geometric.trace import GroundPlaneEstimate


class GroundSettings(Protocol):
    ground_candidate_below_bottom_m: float
    ground_candidate_above_bottom_m: float
    ground_distance_tolerance_m: float
    ground_huber_delta_m: float
    ground_max_rmse_m: float
    ground_max_tilt_deg: float
    ground_min_candidates: int
    ground_min_inliers: int
    ground_min_inlier_fraction: float
    ground_irls_iterations: int


def estimate_ground_plane(
    points: NDArray[np.float64],
    local: NDArray[np.float64],
    *,
    bottom_local_z: float,
    settings: GroundSettings,
) -> GroundPlaneEstimate | None:
    """Fit a robust local plane near the coarse object's expected bottom."""

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


def classify_ground(
    points: NDArray[np.float64],
    local: NDArray[np.float64],
    plane: GroundPlaneEstimate | None,
    *,
    bottom_local_z: float,
    settings: GroundSettings,
) -> NDArray[np.bool_]:
    """Return the ROI point mask supported by a fitted local ground plane."""

    if plane is None or not len(points):
        return np.zeros(len(points), dtype=bool)
    a, b, c = plane.z_from_xyc
    residual = np.abs(points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c))
    below_upper_band = (
        local[:, 2] <= bottom_local_z + settings.ground_candidate_above_bottom_m
    )
    return (residual <= settings.ground_distance_tolerance_m) & below_upper_band


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
