"""Experimental Stage 3 pairwise registration and anchored pose graph."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from math import atan2, degrees, radians
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trackrefinery.component_consensus.aggregation import (
    _aggregate_sharpness,
    _aggregation_order,
    _canonical_shape,
    _ckdtree_type,
    _correspondences,
    _force_retain_coarse,
    _FrameState,
    _geometry_states,
    _registration_trace,
    _rejected_registration,
    _trajectory_metrics,
    _upright_step,
    _yaw_rotation,
)
from trackrefinery.component_consensus.settings import ComponentConsensusSettings
from trackrefinery.contracts import Pose3D, RefinementCase
from trackrefinery.geometric.trace import (
    AnchoredAggregationTrace,
    GeometricRefinementTrace,
    validate_geometric_trace,
)
from trackrefinery.geometry import compose_pose, yaw_from_quaternion

POSE_GRAPH_EXPERIMENT_CONTRACT = "trackrefinery-stage3-pose-graph-experiment-v1"
POSE_GRAPH_VARIANTS = frozenset(
    {"point_to_point_pose_graph", "normal_aware_pose_graph"}
)


@dataclass(frozen=True, slots=True)
class PoseGraphExperimentSettings:
    temporal_neighbor_count: int = 3
    connect_anchor_to_all: bool = True
    pair_maximum_iterations: int = 12
    pair_maximum_translation_m: float = 0.5
    pair_maximum_yaw_deg: float = 8.0
    pair_minimum_overlap_fraction: float = 0.45
    bridge_minimum_overlap_fraction: float = 0.65
    normal_neighbor_count: int = 12
    normal_maximum_surface_variation: float = 0.18
    normal_minimum_absolute_dot: float = 0.7
    normal_information_minimum_relative_eigenvalue: float = 0.015
    normal_huber_delta_m: float = 0.05
    yaw_hypotheses_deg: tuple[float, ...] = (-4.0, -2.0, 0.0, 2.0, 4.0)
    graph_edge_sigma_m: float = 0.035
    graph_prior_weight: float = 0.03
    graph_trajectory_weight: float = 0.02
    graph_maximum_evaluations: int = 160

    def __post_init__(self) -> None:
        if self.temporal_neighbor_count < 1:
            raise ValueError("temporal_neighbor_count must be positive")
        if self.pair_maximum_iterations < 1:
            raise ValueError("pair_maximum_iterations must be positive")
        if self.normal_neighbor_count < 4:
            raise ValueError("normal_neighbor_count must be at least four")
        if self.graph_maximum_evaluations < 1:
            raise ValueError("graph_maximum_evaluations must be positive")
        positive = (
            self.pair_maximum_translation_m,
            self.pair_maximum_yaw_deg,
            self.pair_minimum_overlap_fraction,
            self.bridge_minimum_overlap_fraction,
            self.normal_maximum_surface_variation,
            self.normal_minimum_absolute_dot,
            self.normal_information_minimum_relative_eigenvalue,
            self.normal_huber_delta_m,
            self.graph_edge_sigma_m,
            self.graph_prior_weight,
            self.graph_trajectory_weight,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("pose-graph settings must be finite and positive")
        if self.normal_maximum_surface_variation >= 1:
            raise ValueError("normal_maximum_surface_variation must be below one")
        if self.pair_minimum_overlap_fraction > 1:
            raise ValueError("pair_minimum_overlap_fraction must not exceed one")
        if self.bridge_minimum_overlap_fraction > 1:
            raise ValueError("bridge_minimum_overlap_fraction must not exceed one")
        if self.normal_minimum_absolute_dot > 1:
            raise ValueError("normal_minimum_absolute_dot must not exceed one")
        hypotheses = tuple(float(value) for value in self.yaw_hypotheses_deg)
        if not hypotheses or any(not np.isfinite(value) for value in hypotheses):
            raise ValueError("yaw_hypotheses_deg must contain finite values")
        object.__setattr__(self, "yaw_hypotheses_deg", hypotheses)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PoseGraphEdgeTrace:
    source_frame_id: str
    target_frame_id: str
    temporal_separation: int
    status: str
    reason: str | None
    translation_xy_m: tuple[float, float] | None
    yaw_deg: float | None
    correspondence_count: int
    overlap_fraction: float
    initial_rmse_m: float | None
    final_rmse_m: float | None
    chosen_initial_yaw_deg: float | None
    observable_rank: int
    information_eigenvalues: tuple[float, float, float]
    information_condition: float | None
    information_matrix: tuple[tuple[float, float, float], ...]
    radius_m: float

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "translation_xy_m": (
                None if self.translation_xy_m is None else list(self.translation_xy_m)
            ),
            "information_eigenvalues": list(self.information_eigenvalues),
            "information_matrix": [list(row) for row in self.information_matrix],
        }


@dataclass(frozen=True, slots=True)
class PoseGraphExperimentTrace:
    case_id: str
    track_id: str
    variant: str
    anchor_frame_id: str
    settings: PoseGraphExperimentSettings
    attempted_edge_count: int
    accepted_edge_count: int
    connected_frame_ids: tuple[str, ...]
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_evaluations: int
    initial_cost: float
    final_cost: float
    guard_status: str
    guard_reason_codes: tuple[str, ...]
    edges: tuple[PoseGraphEdgeTrace, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": POSE_GRAPH_EXPERIMENT_CONTRACT,
            "case_id": self.case_id,
            "track_id": self.track_id,
            "variant": self.variant,
            "anchor_frame_id": self.anchor_frame_id,
            "settings": self.settings.to_dict(),
            "attempted_edge_count": self.attempted_edge_count,
            "accepted_edge_count": self.accepted_edge_count,
            "connected_frame_ids": list(self.connected_frame_ids),
            "optimizer_success": self.optimizer_success,
            "optimizer_status": self.optimizer_status,
            "optimizer_message": self.optimizer_message,
            "optimizer_evaluations": self.optimizer_evaluations,
            "initial_cost": self.initial_cost,
            "final_cost": self.final_cost,
            "guard_status": self.guard_status,
            "guard_reason_codes": list(self.guard_reason_codes),
            "edges": [edge.to_dict() for edge in self.edges],
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
class PoseGraphAggregationRun:
    trace: GeometricRefinementTrace
    pose_graph: PoseGraphExperimentTrace


@dataclass(frozen=True, slots=True)
class _PairMeasurement:
    source_index: int
    target_index: int
    temporal_separation: int
    transform: NDArray[np.float64] | None
    information: NDArray[np.float64]
    radius_m: float
    trace: PoseGraphEdgeTrace


def aggregate_geometry_components_pose_graph(
    case: RefinementCase,
    component_trace: GeometricRefinementTrace,
    settings: ComponentConsensusSettings | None = None,
    *,
    variant: str = "normal_aware_pose_graph",
    experiment_settings: PoseGraphExperimentSettings | None = None,
) -> PoseGraphAggregationRun:
    """Run one experimental pairwise pose-graph aggregation candidate."""

    if variant not in POSE_GRAPH_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(POSE_GRAPH_VARIANTS)}")
    validate_geometric_trace(case, component_trace)
    resolved = settings or ComponentConsensusSettings()
    experiment = experiment_settings or PoseGraphExperimentSettings()
    if component_trace.config_sha256 != resolved.sha256:
        raise ValueError("component trace and aggregation settings do not match")
    states = _geometry_states(case, component_trace, resolved)
    if len(states) < resolved.track_minimum_geometry_frames:
        raise ValueError("pose-graph experiment requires a supported geometry track")
    order = _aggregation_order(case, component_trace, states, resolved)
    anchor = order[0]
    pairs = _edge_pairs(case, states, anchor, experiment)
    normal_cache = (
        {
            state.index: _estimate_normals(state.reduced_points, experiment)
            for state in states
        }
        if variant == "normal_aware_pose_graph"
        else {}
    )
    measurements = tuple(
        _measure_pair(
            case,
            source,
            target,
            separation,
            resolved,
            experiment,
            variant,
            normal_cache,
        )
        for source, target, separation in pairs
    )
    accepted = tuple(item for item in measurements if item.transform is not None)
    connected = _connected_indices(anchor.index, accepted)
    if (
        len(connected) < resolved.track_minimum_geometry_frames
        or len(connected) != len(states)
    ):
        return _insufficient_pose_graph_run(
            case,
            component_trace,
            states,
            order,
            anchor,
            measurements,
            connected,
            variant,
            experiment,
            "disconnected_pose_graph",
        )
    weak_bridges = tuple(
        edge
        for edge in _bridge_edges(anchor.index, accepted)
        if edge.trace.overlap_fraction < experiment.bridge_minimum_overlap_fraction
    )
    if weak_bridges:
        return _insufficient_pose_graph_run(
            case,
            component_trace,
            states,
            order,
            anchor,
            measurements,
            connected,
            variant,
            experiment,
            "weak_partial_overlap_bridge",
        )
    active_states = [state for state in states if state.index in connected]
    solution, optimizer = _solve_pose_graph(
        case,
        active_states,
        accepted,
        anchor.index,
        resolved,
        experiment,
    )
    failure_reason = None
    if not optimizer.success:
        failure_reason = "pose_graph_optimizer_failed"
    else:
        translation_bound = resolved.aggregation_maximum_xy_correction_m
        yaw_bound = radians(resolved.aggregation_maximum_yaw_correction_deg)
        saturated = any(
            np.linalg.norm(_transform_vector(transform)[:2]) >= translation_bound - 1e-6
            or abs(_transform_vector(transform)[2]) >= yaw_bound - 1e-6
            for index, transform in solution.items()
            if index != anchor.index
        )
        if saturated:
            failure_reason = "correction_bound_saturated"
    if failure_reason is not None:
        return _insufficient_pose_graph_run(
            case,
            component_trace,
            states,
            order,
            anchor,
            measurements,
            connected,
            variant,
            experiment,
            failure_reason,
            optimizer,
        )
    for state in active_states:
        transform = solution[state.index]
        state.rotation = transform[:3, :3]
        state.translation = transform[:3, 3]
        state.accepted = True
        state.iterations = max(1, int(optimizer.nfev))
        incident = [
            edge
            for edge in accepted
            if state.index in {edge.source_index, edge.target_index}
        ]
        state.correspondence_count = sum(
            edge.trace.correspondence_count for edge in incident
        )
        initial = [
            edge.trace.initial_rmse_m
            for edge in incident
            if edge.trace.initial_rmse_m is not None
        ]
        final = [
            edge.trace.final_rmse_m
            for edge in incident
            if edge.trace.final_rmse_m is not None
        ]
        state.initial_rmse_m = float(np.mean(initial)) if initial else 0.0
        state.final_rmse_m = float(np.mean(final)) if final else 0.0
        if state.index == anchor.index:
            state.retained_reason = "anchor_frame"
    active_states, guard_status, guard_reasons = _protect_pose_graph_regressions(
        case, active_states, resolved
    )
    trace = _materialize_trace(case, component_trace, active_states, anchor, resolved)
    diagnostics = PoseGraphExperimentTrace(
        case_id=case.case_id,
        track_id=case.track.track_id,
        variant=variant,
        anchor_frame_id=case.frames[anchor.index].frame_id,
        settings=experiment,
        attempted_edge_count=len(measurements),
        accepted_edge_count=len(accepted),
        connected_frame_ids=tuple(
            case.frames[index].frame_id for index in sorted(connected)
        ),
        optimizer_success=bool(optimizer.success),
        optimizer_status=int(optimizer.status),
        optimizer_message=str(optimizer.message),
        optimizer_evaluations=int(optimizer.nfev),
        initial_cost=float(optimizer.initial_cost),
        final_cost=float(optimizer.cost),
        guard_status=guard_status,
        guard_reason_codes=guard_reasons,
        edges=tuple(item.trace for item in measurements),
    )
    return PoseGraphAggregationRun(trace, diagnostics)


def _edge_pairs(
    case: RefinementCase,
    states: list[_FrameState],
    anchor: _FrameState,
    settings: PoseGraphExperimentSettings,
) -> list[tuple[_FrameState, _FrameState, int]]:
    ordered = sorted(states, key=lambda item: case.frames[item.index].timestamp_ns)
    pairs: dict[tuple[int, int], tuple[_FrameState, _FrameState, int]] = {}
    for position, source in enumerate(ordered):
        for separation in range(1, settings.temporal_neighbor_count + 1):
            target_position = position + separation
            if target_position >= len(ordered):
                break
            target = ordered[target_position]
            pairs[(source.index, target.index)] = (source, target, separation)
    if settings.connect_anchor_to_all:
        anchor_position = ordered.index(anchor)
        for position, state in enumerate(ordered):
            if state is anchor:
                continue
            source, target = (
                (state, anchor) if position < anchor_position else (anchor, state)
            )
            key = (source.index, target.index)
            pairs.setdefault(key, (source, target, abs(position - anchor_position)))
    return [pairs[key] for key in sorted(pairs)]


def _measure_pair(
    case: RefinementCase,
    source: _FrameState,
    target: _FrameState,
    separation: int,
    component_settings: ComponentConsensusSettings,
    settings: PoseGraphExperimentSettings,
    variant: str,
    normal_cache: dict[int, tuple[NDArray[np.float64], NDArray[np.bool_]]],
) -> _PairMeasurement:
    if variant == "point_to_point_pose_graph":
        result = _point_to_point_measurement(
            source.reduced_points,
            target.reduced_points,
            component_settings,
            settings,
        )
    else:
        result = _normal_aware_measurement(
            source.reduced_points,
            target.reduced_points,
            normal_cache[source.index],
            normal_cache[target.index],
            component_settings,
            settings,
        )
    source_id = case.frames[source.index].frame_id
    target_id = case.frames[target.index].frame_id
    if result is None:
        trace = PoseGraphEdgeTrace(
            source_frame_id=source_id,
            target_frame_id=target_id,
            temporal_separation=separation,
            status="rejected",
            reason="pairwise_alignment_failed",
            translation_xy_m=None,
            yaw_deg=None,
            correspondence_count=0,
            overlap_fraction=0.0,
            initial_rmse_m=None,
            final_rmse_m=None,
            chosen_initial_yaw_deg=None,
            observable_rank=0,
            information_eigenvalues=(0.0, 0.0, 0.0),
            information_condition=None,
            information_matrix=((0.0, 0.0, 0.0),) * 3,
            radius_m=1.0,
        )
        return _PairMeasurement(
            source.index,
            target.index,
            separation,
            None,
            np.zeros((3, 3)),
            1.0,
            trace,
        )
    transform, information, radius, metrics = result
    yaw = atan2(transform[1, 0], transform[0, 0])
    eigenvalues = tuple(float(value) for value in np.linalg.eigvalsh(information))
    positive_eigenvalues = [value for value in eigenvalues if value > 1e-12]
    trace = PoseGraphEdgeTrace(
        source_frame_id=source_id,
        target_frame_id=target_id,
        temporal_separation=separation,
        status="accepted",
        reason=None,
        translation_xy_m=(float(transform[0, 3]), float(transform[1, 3])),
        yaw_deg=degrees(yaw),
        correspondence_count=int(metrics["correspondence_count"]),
        overlap_fraction=float(metrics["overlap_fraction"]),
        initial_rmse_m=float(metrics["initial_rmse_m"]),
        final_rmse_m=float(metrics["final_rmse_m"]),
        chosen_initial_yaw_deg=float(metrics["chosen_initial_yaw_deg"]),
        observable_rank=int(np.count_nonzero(np.asarray(eigenvalues) > 1e-12)),
        information_eigenvalues=eigenvalues,
        information_condition=(
            max(positive_eigenvalues) / min(positive_eigenvalues)
            if positive_eigenvalues
            else None
        ),
        information_matrix=tuple(
            tuple(float(value) for value in row) for row in information
        ),
        radius_m=radius,
    )
    return _PairMeasurement(
        source.index,
        target.index,
        separation,
        transform,
        information,
        radius,
        trace,
    )


def _point_to_point_measurement(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    component_settings: ComponentConsensusSettings,
    settings: PoseGraphExperimentSettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, dict[str, float]] | None:
    rotation = np.eye(3)
    translation = np.zeros(3)
    tree = _ckdtree_type()(target)
    initial = _correspondences(source, target, tree, component_settings)
    if initial is None:
        return None
    for _ in range(settings.pair_maximum_iterations):
        aligned = source @ rotation.T + translation
        correspondences = _correspondences(aligned, target, tree, component_settings)
        if correspondences is None:
            return None
        step = _upright_step(aligned, target, correspondences, component_settings)
        if step is None:
            return None
        step_rotation, step_translation, step_yaw = step
        rotation = step_rotation @ rotation
        translation = step_rotation @ translation + step_translation
        total_yaw = abs(atan2(rotation[1, 0], rotation[0, 0]))
        if np.linalg.norm(
            translation[:2]
        ) > settings.pair_maximum_translation_m or total_yaw > radians(
            settings.pair_maximum_yaw_deg
        ):
            return None
        if np.linalg.norm(
            step_translation[:2]
        ) <= component_settings.aggregation_convergence_translation_m and abs(
            step_yaw
        ) <= radians(component_settings.aggregation_convergence_yaw_deg):
            break
    aligned = source @ rotation.T + translation
    final = _correspondences(aligned, target, tree, component_settings)
    if (
        final is None
        or final.overlap_fraction < settings.pair_minimum_overlap_fraction
        or final.rmse_m > initial.rmse_m + 1e-9
    ):
        return None
    radius = _horizontal_radius(np.concatenate((source, target)))
    information = _point_information(aligned, final.source_positions, radius, settings)
    transform = _homogeneous(rotation, translation)
    return (
        transform,
        information,
        radius,
        {
            "correspondence_count": float(len(final.distances)),
            "overlap_fraction": final.overlap_fraction,
            "initial_rmse_m": initial.rmse_m,
            "final_rmse_m": final.rmse_m,
            "chosen_initial_yaw_deg": 0.0,
        },
    )


def _normal_aware_measurement(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    source_normal_data: tuple[NDArray[np.float64], NDArray[np.bool_]],
    target_normal_data: tuple[NDArray[np.float64], NDArray[np.bool_]],
    component_settings: ComponentConsensusSettings,
    settings: PoseGraphExperimentSettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, dict[str, float]] | None:
    source_normals, source_valid = source_normal_data
    target_normals, target_valid = target_normal_data
    candidates = []
    for hypothesis in settings.yaw_hypotheses_deg:
        result = _run_normal_alignment(
            source,
            target,
            source_normals,
            source_valid,
            target_normals,
            target_valid,
            radians(hypothesis),
            component_settings,
            settings,
        )
        if (
            result is not None
            and result[3]["overlap_fraction"] >= settings.pair_minimum_overlap_fraction
        ):
            candidates.append(result)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item[3]["final_rmse_m"],
            -item[3]["overlap_fraction"],
            abs(atan2(item[0][1, 0], item[0][0, 0])),
        )
    )
    return candidates[0]


def _run_normal_alignment(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    source_normals: NDArray[np.float64],
    source_valid: NDArray[np.bool_],
    target_normals: NDArray[np.float64],
    target_valid: NDArray[np.bool_],
    initial_yaw: float,
    component_settings: ComponentConsensusSettings,
    settings: PoseGraphExperimentSettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, dict[str, float]] | None:
    rotation = _yaw_rotation(initial_yaw)
    translation = np.zeros(3)
    target_tree = _ckdtree_type()(target)
    initial = _normal_correspondences(
        source,
        target,
        source_normals,
        source_valid,
        target_normals,
        target_valid,
        rotation,
        translation,
        target_tree,
        component_settings,
        settings,
    )
    if initial is None:
        return None
    for _ in range(settings.pair_maximum_iterations):
        current = _normal_correspondences(
            source,
            target,
            source_normals,
            source_valid,
            target_normals,
            target_valid,
            rotation,
            translation,
            target_tree,
            component_settings,
            settings,
        )
        if current is None:
            return None
        source_positions, target_positions, distances, overlap = current
        aligned = source @ rotation.T + translation
        aligned_normals = source_normals @ rotation.T
        residuals, jacobian, radius = _normal_system(
            aligned,
            target,
            aligned_normals,
            target_normals,
            source_positions,
            target_positions,
        )
        if len(residuals) < component_settings.aggregation_minimum_correspondences:
            return None
        weights = np.minimum(
            1.0,
            settings.normal_huber_delta_m / np.maximum(np.abs(residuals), 1e-12),
        )
        weighted = jacobian * np.sqrt(weights)[:, None]
        right = residuals * np.sqrt(weights)
        step_scaled = -np.linalg.pinv(weighted, rcond=1e-5) @ right
        step = np.asarray(
            (step_scaled[0], step_scaled[1], step_scaled[2] / radius),
            dtype=np.float64,
        )
        xy_norm = float(np.linalg.norm(step[:2]))
        if xy_norm > component_settings.aggregation_maximum_xy_step_m:
            step[:2] *= component_settings.aggregation_maximum_xy_step_m / xy_norm
        yaw_limit = radians(component_settings.aggregation_maximum_yaw_step_deg)
        step[2] = np.clip(step[2], -yaw_limit, yaw_limit)
        step_rotation = _yaw_rotation(float(step[2]))
        step_translation = np.asarray((step[0], step[1], 0.0))
        rotation = step_rotation @ rotation
        translation = step_rotation @ translation + step_translation
        total_yaw = abs(atan2(rotation[1, 0], rotation[0, 0]))
        if np.linalg.norm(
            translation[:2]
        ) > settings.pair_maximum_translation_m or total_yaw > radians(
            settings.pair_maximum_yaw_deg
        ):
            return None
        if np.linalg.norm(
            step[:2]
        ) <= component_settings.aggregation_convergence_translation_m and abs(
            step[2]
        ) <= radians(component_settings.aggregation_convergence_yaw_deg):
            break
    final = _normal_correspondences(
        source,
        target,
        source_normals,
        source_valid,
        target_normals,
        target_valid,
        rotation,
        translation,
        target_tree,
        component_settings,
        settings,
    )
    if final is None:
        return None
    source_positions, target_positions, distances, overlap = final
    aligned = source @ rotation.T + translation
    aligned_normals = source_normals @ rotation.T
    residuals, jacobian, radius = _normal_system(
        aligned,
        target,
        aligned_normals,
        target_normals,
        source_positions,
        target_positions,
    )
    final_rmse = float(np.sqrt(np.mean(np.square(residuals))))
    initial_positions, initial_targets, _, _ = initial
    initial_aligned = source @ _yaw_rotation(initial_yaw).T
    initial_residuals, _, _ = _normal_system(
        initial_aligned,
        target,
        source_normals @ _yaw_rotation(initial_yaw).T,
        target_normals,
        initial_positions,
        initial_targets,
    )
    initial_rmse = float(np.sqrt(np.mean(np.square(initial_residuals))))
    if final_rmse > initial_rmse + 1e-9:
        return None
    weights = np.minimum(
        1.0,
        settings.normal_huber_delta_m / np.maximum(np.abs(residuals), 1e-12),
    )
    information = _normalized_information(
        jacobian.T @ (weights[:, None] * jacobian), settings
    )
    if not np.any(np.linalg.eigvalsh(information) > 1e-12):
        return None
    return (
        _homogeneous(rotation, translation),
        information,
        radius,
        {
            "correspondence_count": float(len(distances)),
            "overlap_fraction": overlap,
            "initial_rmse_m": initial_rmse,
            "final_rmse_m": final_rmse,
            "chosen_initial_yaw_deg": degrees(initial_yaw),
        },
    )


def _normal_correspondences(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    source_normals: NDArray[np.float64],
    source_valid: NDArray[np.bool_],
    target_normals: NDArray[np.float64],
    target_valid: NDArray[np.bool_],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
    target_tree: object,
    component_settings: ComponentConsensusSettings,
    settings: PoseGraphExperimentSettings,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], float] | None:
    aligned = source @ rotation.T + translation
    distances, nearest = target_tree.query(aligned, k=1, workers=1)
    reverse_tree = _ckdtree_type()(aligned)
    _, reverse = reverse_tree.query(target, k=1, workers=1)
    positions = np.arange(len(source), dtype=np.int64)
    nearest = np.asarray(nearest, dtype=np.int64)
    aligned_normals = source_normals @ rotation.T
    compatibility = np.abs(np.sum(aligned_normals * target_normals[nearest], axis=1))
    mask = (
        (distances <= component_settings.aggregation_maximum_correspondence_distance_m)
        & source_valid
        & target_valid[nearest]
        & (compatibility >= settings.normal_minimum_absolute_dot)
    )
    mutual = np.asarray(reverse, dtype=np.int64)[nearest] == positions
    positions = positions[mask]
    overlap = min(1.0, len(positions) / max(1, min(len(source), len(target))))
    if (
        len(positions) < component_settings.aggregation_minimum_correspondences
        or overlap < component_settings.aggregation_minimum_overlap_fraction
    ):
        return None
    keep_count = max(
        component_settings.aggregation_minimum_correspondences,
        int(
            len(positions) * component_settings.aggregation_correspondence_trim_fraction
        ),
    )
    ordering = np.lexsort(
        (
            np.asarray(distances[positions], dtype=np.float64),
            (~mutual[positions]).astype(np.int8),
        )
    )[:keep_count]
    positions = positions[ordering]
    return (
        positions,
        nearest[positions],
        np.asarray(distances[positions], dtype=np.float64),
        float(overlap),
    )


def _normal_system(
    aligned: NDArray[np.float64],
    target: NDArray[np.float64],
    aligned_normals: NDArray[np.float64],
    target_normals: NDArray[np.float64],
    source_positions: NDArray[np.int64],
    target_positions: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    source_points = aligned[source_positions]
    target_points = target[target_positions]
    difference = source_points - target_points
    normals = np.concatenate(
        (target_normals[target_positions], aligned_normals[source_positions])
    )
    repeated_points = np.concatenate((source_points, source_points))
    repeated_difference = np.concatenate((difference, difference))
    residuals = np.sum(normals * repeated_difference, axis=1)
    radius = _horizontal_radius(np.concatenate((source_points, target_points)))
    yaw_derivative = np.column_stack(
        (-repeated_points[:, 1], repeated_points[:, 0], np.zeros(len(normals)))
    )
    jacobian = np.column_stack(
        (
            normals[:, 0],
            normals[:, 1],
            np.sum(normals * yaw_derivative, axis=1) / radius,
        )
    )
    informative = np.linalg.norm(jacobian, axis=1) > 1e-8
    return residuals[informative], jacobian[informative], radius


def _estimate_normals(
    points: NDArray[np.float64], settings: PoseGraphExperimentSettings
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    count = min(settings.normal_neighbor_count, len(points))
    _, neighbors = _ckdtree_type()(points).query(points, k=count, workers=1)
    if count == 1:
        neighbors = np.asarray(neighbors)[:, None]
    normals = np.zeros_like(points)
    valid = np.zeros(len(points), dtype=bool)
    for index, positions in enumerate(np.asarray(neighbors, dtype=np.int64)):
        local = points[positions]
        centered = local - np.mean(local, axis=0)
        covariance = centered.T @ centered / max(1, len(local))
        values, vectors = np.linalg.eigh(covariance)
        total = float(np.sum(values))
        variation = float(values[0] / total) if total > 1e-12 else 1.0
        if variation <= settings.normal_maximum_surface_variation:
            normals[index] = vectors[:, 0]
            valid[index] = True
    return normals, valid


def _point_information(
    aligned: NDArray[np.float64],
    positions: NDArray[np.int64],
    radius: float,
    settings: PoseGraphExperimentSettings,
) -> NDArray[np.float64]:
    points = aligned[positions]
    first = np.column_stack(
        (np.ones(len(points)), np.zeros(len(points)), -points[:, 1] / radius)
    )
    second = np.column_stack(
        (np.zeros(len(points)), np.ones(len(points)), points[:, 0] / radius)
    )
    jacobian = np.concatenate((first, second))
    return _normalized_information(jacobian.T @ jacobian, settings)


def _normalized_information(
    matrix: NDArray[np.float64], settings: PoseGraphExperimentSettings
) -> NDArray[np.float64]:
    values, vectors = np.linalg.eigh(matrix)
    maximum = float(np.max(values))
    if maximum <= 1e-12:
        return np.zeros((3, 3))
    normalized = values / maximum
    normalized[normalized < settings.normal_information_minimum_relative_eigenvalue] = (
        0.0
    )
    return vectors @ np.diag(normalized) @ vectors.T


def _connected_indices(
    anchor_index: int, edges: tuple[_PairMeasurement, ...]
) -> set[int]:
    adjacency: dict[int, set[int]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source_index, set()).add(edge.target_index)
        adjacency.setdefault(edge.target_index, set()).add(edge.source_index)
    connected = {anchor_index}
    frontier = [anchor_index]
    while frontier:
        current = frontier.pop()
        for neighbor in sorted(adjacency.get(current, ())):
            if neighbor not in connected:
                connected.add(neighbor)
                frontier.append(neighbor)
    return connected


def _bridge_edges(
    anchor_index: int, edges: tuple[_PairMeasurement, ...]
) -> tuple[_PairMeasurement, ...]:
    baseline = _connected_indices(anchor_index, edges)
    bridges = []
    for position, edge in enumerate(edges):
        remaining = edges[:position] + edges[position + 1 :]
        if _connected_indices(anchor_index, remaining) != baseline:
            bridges.append(edge)
    return tuple(bridges)


def _insufficient_pose_graph_run(
    case: RefinementCase,
    component_trace: GeometricRefinementTrace,
    states: list[_FrameState],
    order: list[_FrameState],
    anchor: _FrameState,
    measurements: tuple[_PairMeasurement, ...],
    connected: set[int],
    variant: str,
    settings: PoseGraphExperimentSettings,
    reason: str,
    optimizer: _OptimizerSummary | None = None,
) -> PoseGraphAggregationRun:
    by_index = {state.index: state for state in states}
    for state in states:
        state.failure_reason = reason
    attempted = tuple(case.frames[state.index].frame_id for state in order)
    frames = tuple(
        replace(
            frame,
            registration=(
                _rejected_registration(by_index[index]) if index in by_index else None
            ),
        )
        for index, frame in enumerate(component_trace.frames)
    )
    trace = replace(
        component_trace,
        stage="pose_graph_aggregation_v3_experiment",
        frames=frames,
        anchored_aggregation=AnchoredAggregationTrace(
            status="insufficient_evidence",
            reason_codes=(reason,),
            anchor_frame_id=case.frames[anchor.index].frame_id,
            attempted_frame_ids=attempted,
            accepted_frame_ids=(),
            rejected_frame_ids=attempted,
            baseline_sharpness=None,
            candidate_sharpness=None,
            maximum_correction_velocity_mps=None,
            maximum_correction_acceleration_mps2=None,
            maximum_correction_yaw_rate_degps=None,
        ),
        canonical_shape=None,
    )
    accepted_count = sum(item.transform is not None for item in measurements)
    diagnostics = PoseGraphExperimentTrace(
        case_id=case.case_id,
        track_id=case.track.track_id,
        variant=variant,
        anchor_frame_id=case.frames[anchor.index].frame_id,
        settings=settings,
        attempted_edge_count=len(measurements),
        accepted_edge_count=accepted_count,
        connected_frame_ids=tuple(
            case.frames[index].frame_id for index in sorted(connected)
        ),
        optimizer_success=False if optimizer is None else optimizer.success,
        optimizer_status=0 if optimizer is None else optimizer.status,
        optimizer_message=(
            f"not_run:{reason}" if optimizer is None else optimizer.message
        ),
        optimizer_evaluations=0 if optimizer is None else optimizer.nfev,
        initial_cost=0.0 if optimizer is None else optimizer.initial_cost,
        final_cost=0.0 if optimizer is None else optimizer.cost,
        guard_status="insufficient_evidence",
        guard_reason_codes=(reason,),
        edges=tuple(item.trace for item in measurements),
    )
    return PoseGraphAggregationRun(trace, diagnostics)


@dataclass(frozen=True, slots=True)
class _OptimizerSummary:
    success: bool
    status: int
    message: str
    nfev: int
    cost: float
    initial_cost: float


def _solve_pose_graph(
    case: RefinementCase,
    states: list[_FrameState],
    edges: tuple[_PairMeasurement, ...],
    anchor_index: int,
    component_settings: ComponentConsensusSettings,
    settings: PoseGraphExperimentSettings,
) -> tuple[dict[int, NDArray[np.float64]], _OptimizerSummary]:
    try:
        from scipy.optimize import least_squares
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pose-graph aggregation requires trackrefinery[geometric]"
        ) from error
    active = {state.index for state in states}
    active_edges = tuple(
        edge
        for edge in edges
        if edge.source_index in active and edge.target_index in active
    )
    variable_indices = sorted(active - {anchor_index})
    positions = {index: position for position, index in enumerate(variable_indices)}

    def unpack(values: NDArray[np.float64]) -> dict[int, NDArray[np.float64]]:
        transforms = {anchor_index: np.eye(4)}
        for index in variable_indices:
            offset = positions[index] * 3
            transforms[index] = _vector_transform(values[offset : offset + 3])
        return transforms

    def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
        transforms = unpack(values)
        rows: list[NDArray[np.float64]] = []
        for edge in active_edges:
            assert edge.transform is not None
            source = transforms[edge.source_index]
            target = transforms[edge.target_index]
            error = np.linalg.inv(edge.transform) @ np.linalg.inv(target) @ source
            vector = _transform_vector(error)
            scaled = np.asarray((vector[0], vector[1], edge.radius_m * vector[2]))
            values_eig, vectors_eig = np.linalg.eigh(edge.information)
            square_root = (
                vectors_eig
                @ np.diag(np.sqrt(np.maximum(values_eig, 0)))
                @ vectors_eig.T
            )
            rows.append(square_root @ scaled / settings.graph_edge_sigma_m)
        prior_scale = np.sqrt(settings.graph_prior_weight)
        for index in variable_indices:
            vector = _transform_vector(transforms[index])
            rows.append(
                prior_scale
                * np.asarray(
                    (
                        vector[0]
                        / component_settings.aggregation_maximum_xy_correction_m,
                        vector[1]
                        / component_settings.aggregation_maximum_xy_correction_m,
                        vector[2]
                        / radians(
                            component_settings.aggregation_maximum_yaw_correction_deg
                        ),
                    )
                )
            )
        rows.extend(
            _trajectory_residuals(
                case,
                states,
                transforms,
                settings.graph_trajectory_weight,
            )
        )
        return np.concatenate(rows) if rows else np.zeros(0)

    initial = _initial_graph_values(
        variable_indices,
        positions,
        active_edges,
        anchor_index,
        component_settings,
    )
    translation_bound = component_settings.aggregation_maximum_xy_correction_m
    yaw_bound = radians(component_settings.aggregation_maximum_yaw_correction_deg)
    lower = np.tile(
        (-translation_bound, -translation_bound, -yaw_bound), len(variable_indices)
    )
    upper = -lower
    initial = np.clip(initial, lower + 1e-9, upper - 1e-9)
    initial_residual = residual(initial)
    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="huber",
        f_scale=1.0,
        max_nfev=settings.graph_maximum_evaluations,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    summary = _OptimizerSummary(
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        nfev=int(result.nfev),
        cost=float(result.cost),
        initial_cost=float(0.5 * np.sum(np.square(initial_residual))),
    )
    return unpack(np.asarray(result.x, dtype=np.float64)), summary


def _initial_graph_values(
    variable_indices: list[int],
    positions: dict[int, int],
    edges: tuple[_PairMeasurement, ...],
    anchor_index: int,
    component_settings: ComponentConsensusSettings,
) -> NDArray[np.float64]:
    transforms = {anchor_index: np.eye(4)}
    while True:
        changed = False
        for edge in edges:
            assert edge.transform is not None
            if edge.target_index in transforms and edge.source_index not in transforms:
                transforms[edge.source_index] = (
                    transforms[edge.target_index] @ edge.transform
                )
                changed = True
            elif (
                edge.source_index in transforms and edge.target_index not in transforms
            ):
                transforms[edge.target_index] = transforms[
                    edge.source_index
                ] @ np.linalg.inv(edge.transform)
                changed = True
        if not changed:
            break
    values = np.zeros(len(variable_indices) * 3)
    for index in variable_indices:
        vector = _transform_vector(transforms.get(index, np.eye(4)))
        vector[:2] = np.clip(
            vector[:2],
            -component_settings.aggregation_maximum_xy_correction_m,
            component_settings.aggregation_maximum_xy_correction_m,
        )
        vector[2] = np.clip(
            vector[2],
            -radians(component_settings.aggregation_maximum_yaw_correction_deg),
            radians(component_settings.aggregation_maximum_yaw_correction_deg),
        )
        offset = positions[index] * 3
        values[offset : offset + 3] = vector
    return values


def _trajectory_residuals(
    case: RefinementCase,
    states: list[_FrameState],
    transforms: dict[int, NDArray[np.float64]],
    weight: float,
) -> list[NDArray[np.float64]]:
    ordered = sorted(states, key=lambda state: case.frames[state.index].timestamp_ns)
    positions: list[NDArray[np.float64]] = []
    yaws: list[float] = []
    times: list[float] = []
    origin = case.frames[ordered[0].index].timestamp_ns
    for state in ordered:
        index = state.index
        frame = case.frames[index]
        coarse = case.track.observations[index].coarse_box.pose
        coarse_world = compose_pose(frame.world_from_annotation, coarse)
        coarse_matrix = _pose_planar_matrix(coarse_world)
        candidate = coarse_matrix @ np.linalg.inv(transforms[index])
        positions.append(np.asarray(candidate[:2, 3], dtype=np.float64))
        yaws.append(atan2(candidate[1, 0], candidate[0, 0]))
        times.append((frame.timestamp_ns - origin) * 1e-9)
    velocities: list[NDArray[np.float64]] = []
    yaw_rates: list[float] = []
    intervals: list[float] = []
    for index in range(1, len(ordered)):
        delta_t = times[index] - times[index - 1]
        if delta_t <= 0:
            raise ValueError("frame timestamps must be strictly increasing")
        intervals.append(delta_t)
        velocities.append((positions[index] - positions[index - 1]) / delta_t)
        yaw_delta = atan2(
            np.sin(yaws[index] - yaws[index - 1]),
            np.cos(yaws[index] - yaws[index - 1]),
        )
        yaw_rates.append(yaw_delta / delta_t)
    scale = np.sqrt(weight)
    rows: list[NDArray[np.float64]] = []
    for index in range(1, len(velocities)):
        delta_t = (intervals[index] + intervals[index - 1]) / 2
        acceleration = (velocities[index] - velocities[index - 1]) / delta_t
        yaw_acceleration = (yaw_rates[index] - yaw_rates[index - 1]) / delta_t
        rows.append(
            scale
            * np.asarray(
                (
                    acceleration[0] / 5.0,
                    acceleration[1] / 5.0,
                    yaw_acceleration / radians(50.0),
                )
            )
        )
    return rows


def _materialize_trace(
    case: RefinementCase,
    component_trace: GeometricRefinementTrace,
    accepted: list[_FrameState],
    anchor: _FrameState,
    settings: ComponentConsensusSettings,
) -> GeometricRefinementTrace:
    by_index = {state.index: state for state in accepted}
    baseline = _aggregate_sharpness(
        [state.reduced_points for state in accepted], settings
    )
    candidate = _aggregate_sharpness(
        [state.aligned(reduced=True) for state in accepted], settings
    )
    velocity, acceleration, yaw_rate = _trajectory_metrics(case, accepted)
    ordered = [anchor, *(state for state in accepted if state.index != anchor.index)]
    frame_ids = tuple(case.frames[state.index].frame_id for state in ordered)
    aggregation = AnchoredAggregationTrace(
        status="candidate",
        reason_codes=(),
        anchor_frame_id=case.frames[anchor.index].frame_id,
        attempted_frame_ids=frame_ids,
        accepted_frame_ids=frame_ids,
        rejected_frame_ids=(),
        baseline_sharpness=baseline,
        candidate_sharpness=candidate,
        maximum_correction_velocity_mps=velocity,
        maximum_correction_acceleration_mps2=acceleration,
        maximum_correction_yaw_rate_degps=yaw_rate,
    )
    canonical = _canonical_shape(case, accepted, settings)
    frames = tuple(
        replace(
            frame,
            registration=(
                _registration_trace(case, by_index[index])
                if index in by_index
                else None
            ),
        )
        for index, frame in enumerate(component_trace.frames)
    )
    return replace(
        component_trace,
        stage="pose_graph_aggregation_v3_experiment",
        frames=frames,
        anchored_aggregation=aggregation,
        canonical_shape=canonical,
    )


def _protect_pose_graph_regressions(
    case: RefinementCase,
    states: list[_FrameState],
    settings: ComponentConsensusSettings,
) -> tuple[list[_FrameState], str, tuple[str, ...]]:
    """Apply a resolution-aware aggregate and trajectory safety gate."""

    if len(states) < 2:
        return states, "accepted", ()
    baseline = _aggregate_sharpness(
        [state.reduced_points for state in states], settings
    )
    candidate = _aggregate_sharpness(
        [state.aligned(reduced=True) for state in states], settings
    )
    velocity, acceleration, yaw_rate = _trajectory_metrics(case, states)
    trajectory_safe = (
        velocity <= settings.aggregation_maximum_correction_velocity_mps
        and acceleration <= settings.aggregation_maximum_correction_acceleration_mps2
        and yaw_rate <= settings.aggregation_maximum_correction_yaw_rate_degps
    )
    resolution_allowance = max(
        settings.aggregation_maximum_axis_spread_regression_m,
        settings.aggregation_sharpness_voxel_size_m / 2,
    )
    axis_safe = all(
        after <= before + resolution_allowance
        for before, after in zip(
            baseline.robust_spread_xyz_m,
            candidate.robust_spread_xyz_m,
            strict=True,
        )
    )
    length, width, _ = baseline.robust_spread_xyz_m
    area_allowance = resolution_allowance * (length + width) + resolution_allowance**2
    area_safe = candidate.xy_area_m2 <= baseline.xy_area_m2 + area_allowance
    concentration_safe = (
        candidate.voxel_concentration
        >= baseline.voxel_concentration
        - settings.aggregation_maximum_concentration_regression
    )
    residual_safe = candidate.cross_frame_rmse_m <= baseline.cross_frame_rmse_m + 1e-12
    reasons = []
    if not trajectory_safe:
        reasons.append("trajectory_regression")
    if not axis_safe:
        reasons.append("axis_spread_regression")
    if not area_safe:
        reasons.append("footprint_area_regression")
    if not concentration_safe:
        reasons.append("voxel_concentration_regression")
    if not residual_safe:
        reasons.append("cross_frame_residual_regression")
    if not reasons:
        return states, "accepted", ()
    reason = reasons[0]
    for state in states:
        if state.retained_reason == "anchor_frame":
            continue
        _force_retain_coarse(state, reason, settings)
    return states, "retained_coarse", tuple(reasons)


def _horizontal_radius(points: NDArray[np.float64]) -> float:
    center = np.median(points[:, :2], axis=0)
    distances = np.linalg.norm(points[:, :2] - center, axis=1)
    return max(0.5, float(np.quantile(distances, 0.9)))


def _homogeneous(
    rotation: NDArray[np.float64], translation: NDArray[np.float64]
) -> NDArray[np.float64]:
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def _vector_transform(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    value = np.eye(4)
    value[:3, :3] = _yaw_rotation(float(vector[2]))
    value[0, 3] = vector[0]
    value[1, 3] = vector[1]
    return value


def _transform_vector(transform: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(
        (
            transform[0, 3],
            transform[1, 3],
            atan2(transform[1, 0], transform[0, 0]),
        ),
        dtype=np.float64,
    )


def _pose_planar_matrix(pose: Pose3D) -> NDArray[np.float64]:
    value = np.eye(4)
    value[:3, :3] = _yaw_rotation(yaw_from_quaternion(pose.orientation_xyzw))
    value[0, 3] = pose.translation_xyz[0]
    value[1, 3] = pose.translation_xyz[1]
    value[2, 3] = pose.translation_xyz[2]
    return value
