"""Versioned settings for the V2 component-consensus backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite

COMPONENT_CONSENSUS_ALGORITHM_VERSION = "component-consensus-refiner-v2"
COMPONENT_CONSENSUS_CONFIG_SCHEMA_VERSION = (
    "trackrefinery-component-consensus-settings-v1"
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
    """Internal V2 ROI, component, and provisional frame-role policy."""

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
    geometry_minimum_points: int = 80
    geometry_minimum_voxels: int = 20
    geometry_minimum_spread_xyz_m: tuple[float, float, float] = (0.9, 0.45, 0.45)
    geometry_minimum_dominance: float = 0.6
    geometry_minimum_stability_iou: float = 0.65
    pose_minimum_points: int = 24
    pose_minimum_voxels: int = 8
    pose_minimum_horizontal_spread_m: float = 0.25
    pose_minimum_vertical_spread_m: float = 0.2
    pose_minimum_dominance: float = 0.45
    pose_minimum_stability_iou: float = 0.35

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
            "pose_minimum_dominance",
            "pose_minimum_stability_iou",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        quantile = float(self.spread_quantile)
        if not isfinite(quantile) or not 0 < quantile < 0.5:
            raise ValueError("spread_quantile must be in (0, 0.5)")
        object.__setattr__(self, "spread_quantile", quantile)
        for name, minimum in (
            ("ground_min_candidates", 3),
            ("ground_min_inliers", 3),
            ("ground_irls_iterations", 1),
            ("minimum_component_points", 1),
            ("minimum_seed_points", 1),
            ("geometry_minimum_points", 1),
            ("geometry_minimum_voxels", 1),
            ("pose_minimum_points", 1),
            ("pose_minimum_voxels", 1),
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
