"""Versioned settings for deterministic geometric refinement."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import isfinite

GEOMETRIC_ALGORITHM_VERSION = "joint-cuboid-refiner-v1"
GEOMETRIC_CONFIG_SCHEMA_VERSION = "trackrefinery-geometric-settings-v2"


def _finite_triplet(
    value: tuple[float, float, float], name: str
) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    result = tuple(float(item) for item in value)
    if any(not isfinite(item) or item < 0 for item in result):
        raise ValueError(f"{name} must contain finite non-negative values")
    return result


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _fraction(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0 < result <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class EvidenceSelectionSettings:
    """Internal initial-ROI and ground-evidence policy.

    These values are backend configuration, not caller-supplied crop policy.
    Later optimization rounds may move or resize the region within these bounds.
    """

    roi_margin_xyz_m: tuple[float, float, float] = (1.0, 1.0, 0.55)
    target_allowance_xyz_m: tuple[float, float, float] = (0.08, 0.08, 0.08)
    ambiguity_margin_xyz_m: tuple[float, float, float] = (0.4, 0.4, 0.22)
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

    def __post_init__(self) -> None:
        roi = _finite_triplet(self.roi_margin_xyz_m, "roi_margin_xyz_m")
        target = _finite_triplet(self.target_allowance_xyz_m, "target_allowance_xyz_m")
        ambiguity = _finite_triplet(
            self.ambiguity_margin_xyz_m, "ambiguity_margin_xyz_m"
        )
        if any(right < left for left, right in zip(target, ambiguity, strict=True)):
            raise ValueError("ambiguity margins must cover target allowances")
        if any(right < left for left, right in zip(ambiguity, roi, strict=True)):
            raise ValueError("ROI margins must cover ambiguity margins")
        object.__setattr__(self, "roi_margin_xyz_m", roi)
        object.__setattr__(self, "target_allowance_xyz_m", target)
        object.__setattr__(self, "ambiguity_margin_xyz_m", ambiguity)

        positive_fields = (
            "ground_candidate_below_bottom_m",
            "ground_candidate_above_bottom_m",
            "ground_distance_tolerance_m",
            "ground_huber_delta_m",
            "ground_max_rmse_m",
            "ground_max_tilt_deg",
        )
        for name in positive_fields:
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if not 0 < self.ground_max_tilt_deg < 90:
            raise ValueError("ground_max_tilt_deg must be in (0, 90)")
        if self.ground_min_candidates < 3:
            raise ValueError("ground_min_candidates must be at least three")
        if not 3 <= self.ground_min_inliers <= self.ground_min_candidates:
            raise ValueError(
                "ground_min_inliers must be between three and ground_min_candidates"
            )
        if not 0 < self.ground_min_inlier_fraction <= 1:
            raise ValueError("ground_min_inlier_fraction must be in (0, 1]")
        if self.ground_irls_iterations < 1:
            raise ValueError("ground_irls_iterations must be positive")


@dataclass(frozen=True, slots=True)
class RegistrationSettings:
    """Deterministic multi-frame registration and shape aggregation policy."""

    voxel_size_m: float = 0.07
    normal_neighbor_count: int = 12
    cross_frame_neighbor_candidates: int = 24
    normal_min_planarity: float = 0.72
    minimum_target_points: int = 30
    minimum_correspondences: int = 24
    maximum_iterations: int = 12
    maximum_correspondence_distance_m: float = 0.4
    correspondence_trim_fraction: float = 0.8
    huber_delta_m: float = 0.08
    step_regularization: float = 20.0
    maximum_xy_step_m: float = 0.08
    maximum_z_step_m: float = 0.04
    maximum_yaw_step_deg: float = 1.5
    convergence_translation_m: float = 0.005
    convergence_yaw_deg: float = 0.1
    initialization_quantile: float = 0.02
    initialization_radial_trim_fraction: float = 0.95
    minimum_planar_anisotropy: float = 1.25
    maximum_initial_yaw_correction_deg: float = 12.0
    canonical_support_radius_m: float = 0.16
    canonical_minimum_frame_support: int = 2

    def __post_init__(self) -> None:
        positive_fields = (
            "voxel_size_m",
            "normal_min_planarity",
            "maximum_correspondence_distance_m",
            "huber_delta_m",
            "step_regularization",
            "maximum_xy_step_m",
            "maximum_z_step_m",
            "maximum_yaw_step_deg",
            "convergence_translation_m",
            "convergence_yaw_deg",
            "minimum_planar_anisotropy",
            "maximum_initial_yaw_correction_deg",
            "canonical_support_radius_m",
        )
        for name in positive_fields:
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        fraction_fields = (
            "normal_min_planarity",
            "correspondence_trim_fraction",
            "initialization_radial_trim_fraction",
        )
        for name in fraction_fields:
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        quantile = float(self.initialization_quantile)
        if not isfinite(quantile) or not 0 < quantile < 0.5:
            raise ValueError("initialization_quantile must be in (0, 0.5)")
        object.__setattr__(self, "initialization_quantile", quantile)
        if self.normal_neighbor_count < 4:
            raise ValueError("normal_neighbor_count must be at least four")
        if self.cross_frame_neighbor_candidates < 2:
            raise ValueError("cross_frame_neighbor_candidates must be at least two")
        if self.minimum_target_points < 4:
            raise ValueError("minimum_target_points must be at least four")
        if not 4 <= self.minimum_correspondences <= self.minimum_target_points:
            raise ValueError(
                "minimum_correspondences must be between four and minimum_target_points"
            )
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        if self.canonical_minimum_frame_support < 2:
            raise ValueError("canonical_minimum_frame_support must be at least two")


@dataclass(frozen=True, slots=True)
class GeometricRefinementSettings:
    """Complete versioned configuration for the first geometric backend."""

    evidence: EvidenceSelectionSettings = field(
        default_factory=EvidenceSelectionSettings
    )
    registration: RegistrationSettings = field(default_factory=RegistrationSettings)

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, EvidenceSelectionSettings):
            raise TypeError("evidence must be EvidenceSelectionSettings")
        if not isinstance(self.registration, RegistrationSettings):
            raise TypeError("registration must be RegistrationSettings")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": GEOMETRIC_CONFIG_SCHEMA_VERSION,
            "evidence": asdict(self.evidence),
            "registration": asdict(self.registration),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
