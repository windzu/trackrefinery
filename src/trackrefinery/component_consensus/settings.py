"""Versioned settings for the V2 component-consensus backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite

COMPONENT_CONSENSUS_ALGORITHM_VERSION = "component-consensus-refiner-v2.1.0"
COMPONENT_CONSENSUS_CONFIG_SCHEMA_VERSION = (
    "trackrefinery-component-consensus-settings-v3"
)


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _fraction(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _triplet(
    value: tuple[float, float, float], name: str
) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or any(not isfinite(item) or item < 0 for item in result):
        raise ValueError(f"{name} must contain three finite non-negative values")
    return result


@dataclass(frozen=True, slots=True)
class ComponentConsensusSettings:
    """Internal V2 component, frame-role, and anchored-aggregation policy."""

    roi_margin_xyz_m: tuple[float, float, float] = (1.0, 1.0, 0.55)
    seed_allowance_xyz_m: tuple[float, float, float] = (0.08, 0.08, 0.08)
    ground_candidate_below_bottom_m: float = 0.4
    ground_candidate_above_bottom_m: float = 0.24
    ground_distance_tolerance_m: float = 0.07
    ground_huber_delta_m: float = 0.05
    ground_max_rmse_m: float = 0.045
    ground_max_tilt_deg: float = 15.0
    ground_min_candidates: int = 8
    ground_min_inliers: int = 6
    ground_min_inlier_fraction: float = 0.5
    ground_irls_iterations: int = 6
    component_ground_clearance_m: float = 0.14
    component_voxel_size_m: float = 0.2
    stability_voxel_scale: float = 1.5
    minimum_component_points: int = 8
    minimum_seed_points: int = 3
    spread_quantile: float = 0.05
    maximum_selected_spread_allowance_xyz_m: tuple[float, float, float] = (
        0.6,
        0.4,
        0.4,
    )
    purity_envelope_allowance_xyz_m: tuple[float, float, float] = (0.35, 0.25, 0.25)
    maximum_outside_envelope_fraction: float = 0.02
    # The first supported MVP slice is deliberately dense. Sparse components
    # may still be useful for a later fixed-shape pose pass, but they must not
    # define canonical geometry.
    geometry_minimum_points: int = 1_000
    geometry_minimum_voxels: int = 100
    geometry_minimum_spread_xyz_m: tuple[float, float, float] = (0.9, 0.45, 0.45)
    geometry_minimum_dominance: float = 0.6
    geometry_minimum_stability_iou: float = 0.65
    geometry_reference_quantile: float = 0.8
    geometry_minimum_relative_points: float = 0.2
    geometry_minimum_relative_spread: float = 0.65
    track_minimum_geometry_frames: int = 5
    pose_minimum_points: int = 24
    pose_minimum_voxels: int = 8
    pose_minimum_horizontal_spread_m: float = 0.25
    pose_minimum_vertical_spread_m: float = 0.2
    pose_minimum_dominance: float = 0.45
    pose_minimum_stability_iou: float = 0.35
    # Stage 3 only consumes geometry frames. These values constrain a local
    # alignment primitive; the dense frame-role thresholds above are not part
    # of its objective.
    aggregation_voxel_size_m: float = 0.08
    aggregation_anchor_minimum_relative_quality: float = 0.8
    aggregation_maximum_iterations: int = 10
    aggregation_maximum_correspondence_distance_m: float = 0.28
    aggregation_correspondence_trim_fraction: float = 0.7
    aggregation_minimum_correspondences: int = 80
    aggregation_minimum_overlap_fraction: float = 0.12
    aggregation_huber_delta_m: float = 0.06
    aggregation_step_gain: float = 0.65
    aggregation_maximum_xy_step_m: float = 0.04
    aggregation_maximum_yaw_step_deg: float = 0.75
    aggregation_maximum_xy_correction_m: float = 0.25
    aggregation_maximum_yaw_correction_deg: float = 4.0
    aggregation_convergence_translation_m: float = 0.002
    aggregation_convergence_yaw_deg: float = 0.03
    aggregation_noop_translation_m: float = 0.006
    aggregation_noop_yaw_deg: float = 0.08
    aggregation_minimum_rmse_improvement_m: float = 0.001
    aggregation_minimum_relative_rmse_improvement: float = 0.01
    aggregation_sharpness_quantile: float = 0.01
    aggregation_sharpness_voxel_size_m: float = 0.1
    aggregation_maximum_axis_spread_regression_m: float = 0.02
    aggregation_maximum_concentration_regression: float = 0.005
    aggregation_maximum_correction_velocity_mps: float = 3.0
    aggregation_maximum_correction_acceleration_mps2: float = 50.0
    aggregation_maximum_correction_yaw_rate_degps: float = 30.0

    def __post_init__(self) -> None:
        roi = _triplet(self.roi_margin_xyz_m, "roi_margin_xyz_m")
        seed = _triplet(self.seed_allowance_xyz_m, "seed_allowance_xyz_m")
        if any(right > left for left, right in zip(roi, seed, strict=True)):
            raise ValueError("seed allowances must not exceed ROI margins")
        object.__setattr__(self, "roi_margin_xyz_m", roi)
        object.__setattr__(self, "seed_allowance_xyz_m", seed)
        object.__setattr__(
            self,
            "geometry_minimum_spread_xyz_m",
            _triplet(
                self.geometry_minimum_spread_xyz_m,
                "geometry_minimum_spread_xyz_m",
            ),
        )
        object.__setattr__(
            self,
            "maximum_selected_spread_allowance_xyz_m",
            _triplet(
                self.maximum_selected_spread_allowance_xyz_m,
                "maximum_selected_spread_allowance_xyz_m",
            ),
        )
        object.__setattr__(
            self,
            "purity_envelope_allowance_xyz_m",
            _triplet(
                self.purity_envelope_allowance_xyz_m,
                "purity_envelope_allowance_xyz_m",
            ),
        )
        for name in (
            "ground_candidate_below_bottom_m",
            "ground_candidate_above_bottom_m",
            "ground_distance_tolerance_m",
            "ground_huber_delta_m",
            "ground_max_rmse_m",
            "ground_max_tilt_deg",
            "component_ground_clearance_m",
            "component_voxel_size_m",
            "stability_voxel_scale",
            "pose_minimum_horizontal_spread_m",
            "pose_minimum_vertical_spread_m",
            "aggregation_voxel_size_m",
            "aggregation_maximum_correspondence_distance_m",
            "aggregation_huber_delta_m",
            "aggregation_maximum_xy_step_m",
            "aggregation_maximum_yaw_step_deg",
            "aggregation_maximum_xy_correction_m",
            "aggregation_maximum_yaw_correction_deg",
            "aggregation_convergence_translation_m",
            "aggregation_convergence_yaw_deg",
            "aggregation_noop_translation_m",
            "aggregation_noop_yaw_deg",
            "aggregation_minimum_rmse_improvement_m",
            "aggregation_sharpness_voxel_size_m",
            "aggregation_maximum_axis_spread_regression_m",
            "aggregation_maximum_correction_velocity_mps",
            "aggregation_maximum_correction_acceleration_mps2",
            "aggregation_maximum_correction_yaw_rate_degps",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if not 0 < self.ground_max_tilt_deg < 90:
            raise ValueError("ground_max_tilt_deg must be in (0, 90)")
        if self.stability_voxel_scale <= 1:
            raise ValueError("stability_voxel_scale must exceed one")
        for name in (
            "ground_min_inlier_fraction",
            "geometry_minimum_dominance",
            "geometry_minimum_stability_iou",
            "geometry_minimum_relative_points",
            "geometry_minimum_relative_spread",
            "maximum_outside_envelope_fraction",
            "pose_minimum_dominance",
            "pose_minimum_stability_iou",
            "aggregation_anchor_minimum_relative_quality",
            "aggregation_correspondence_trim_fraction",
            "aggregation_minimum_overlap_fraction",
            "aggregation_step_gain",
            "aggregation_minimum_relative_rmse_improvement",
            "aggregation_maximum_concentration_regression",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        reference_quantile = _fraction(
            self.geometry_reference_quantile, "geometry_reference_quantile"
        )
        if reference_quantile == 0:
            raise ValueError("geometry_reference_quantile must be greater than zero")
        object.__setattr__(self, "geometry_reference_quantile", reference_quantile)
        quantile = float(self.spread_quantile)
        if not isfinite(quantile) or not 0 < quantile < 0.5:
            raise ValueError("spread_quantile must be in (0, 0.5)")
        object.__setattr__(self, "spread_quantile", quantile)
        sharpness_quantile = float(self.aggregation_sharpness_quantile)
        if not isfinite(sharpness_quantile) or not 0 < sharpness_quantile < 0.5:
            raise ValueError("aggregation_sharpness_quantile must be in (0, 0.5)")
        object.__setattr__(self, "aggregation_sharpness_quantile", sharpness_quantile)
        for name, minimum in (
            ("ground_min_candidates", 3),
            ("ground_min_inliers", 3),
            ("ground_irls_iterations", 1),
            ("minimum_component_points", 1),
            ("minimum_seed_points", 1),
            ("geometry_minimum_points", 1),
            ("geometry_minimum_voxels", 1),
            ("track_minimum_geometry_frames", 1),
            ("pose_minimum_points", 1),
            ("pose_minimum_voxels", 1),
            ("aggregation_maximum_iterations", 1),
            ("aggregation_minimum_correspondences", 4),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if self.ground_min_inliers > self.ground_min_candidates:
            raise ValueError("ground_min_inliers cannot exceed candidates")
        if self.geometry_minimum_points < self.pose_minimum_points:
            raise ValueError("geometry point threshold must cover pose threshold")
        if self.geometry_minimum_voxels < self.pose_minimum_voxels:
            raise ValueError("geometry voxel threshold must cover pose threshold")
        if self.geometry_minimum_dominance < self.pose_minimum_dominance:
            raise ValueError("geometry dominance must cover pose dominance")
        if self.geometry_minimum_stability_iou < self.pose_minimum_stability_iou:
            raise ValueError("geometry stability must cover pose stability")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": COMPONENT_CONSENSUS_CONFIG_SCHEMA_VERSION,
            **asdict(self),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
