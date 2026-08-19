"""Deterministic V2 object-component extraction and provisional frame roles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from trackrefinery.component_consensus.settings import (
    COMPONENT_CONSENSUS_ALGORITHM_VERSION,
    COMPONENT_CONSENSUS_CONFIG_SCHEMA_VERSION,
    ComponentConsensusSettings,
)
from trackrefinery.contracts import Box3D, FrameCloud, RefinementCase
from trackrefinery.geometric.ground import classify_ground, estimate_ground_plane
from trackrefinery.geometric.trace import (
    EvidenceState,
    FrameComponentTrace,
    FrameEvidenceTrace,
    FrameRole,
    GeometricRefinementTrace,
)
from trackrefinery.geometry import inverse_transform_points


@dataclass(frozen=True, slots=True)
class _Partition:
    labels: NDArray[np.int32]
    component_positions: tuple[NDArray[np.int64], ...]
    candidate_ids: tuple[int, ...]
    selected_id: int | None
    seed_counts: tuple[int, ...]
    voxel_counts: tuple[int, ...]


def select_object_components(
    case: RefinementCase,
    settings: ComponentConsensusSettings | None = None,
) -> GeometricRefinementTrace:
    """Select one non-ground spatial component in every track observation."""

    resolved = settings or ComponentConsensusSettings()
    frames = tuple(
        _select_frame_component(frame, observation.coarse_box, resolved)
        for frame, observation in zip(case.frames, case.track.observations, strict=True)
    )
    return GeometricRefinementTrace(
        case_id=case.case_id,
        track_id=case.track.track_id,
        algorithm_version=COMPONENT_CONSENSUS_ALGORITHM_VERSION,
        config_schema_version=COMPONENT_CONSENSUS_CONFIG_SCHEMA_VERSION,
        config_sha256=resolved.sha256,
        settings_json=resolved.canonical_json(),
        stage="component_selection_v2",
        frames=frames,
    )


def _select_frame_component(
    frame: FrameCloud,
    coarse_box: Box3D,
    settings: ComponentConsensusSettings,
) -> FrameEvidenceTrace:
    local = inverse_transform_points(frame.points_xyz, coarse_box.pose)
    half_size = np.asarray(coarse_box.size_lwh, dtype=np.float64) / 2.0
    roi_half_size = half_size + np.asarray(settings.roi_margin_xyz_m)
    roi_mask = np.all(np.abs(local) <= roi_half_size + 1e-9, axis=1)
    roi_indices = np.flatnonzero(roi_mask).astype(np.int64, copy=False)
    roi_points = np.asarray(frame.points_xyz[roi_indices], dtype=np.float64)
    roi_local = np.asarray(local[roi_indices], dtype=np.float64)
    bottom = -float(half_size[2])
    ground = estimate_ground_plane(
        roi_points,
        roi_local,
        bottom_local_z=bottom,
        settings=settings,
    )
    ground_mask = classify_ground(
        roi_points,
        roi_local,
        ground,
        bottom_local_z=bottom,
        settings=settings,
    )
    if ground is not None:
        a, b, c = ground.z_from_xyc
        height_above_ground = roi_points[:, 2] - (
            a * roi_points[:, 0] + b * roi_points[:, 1] + c
        )
        component_ground_mask = ground_mask | (
            height_above_ground <= settings.component_ground_clearance_m
        )
    else:
        component_ground_mask = ground_mask | (
            roi_local[:, 2] <= bottom + min(settings.component_ground_clearance_m, 0.08)
        )
    seed_half_size = half_size + np.asarray(settings.seed_allowance_xyz_m)
    seed_mask = np.all(np.abs(roi_local) <= seed_half_size + 1e-9, axis=1)
    eligible_mask = ~component_ground_mask
    primary = _partition_components(
        roi_local,
        eligible_mask,
        seed_mask,
        settings,
        settings.component_voxel_size_m,
    )
    states = np.full(len(roi_indices), EvidenceState.BACKGROUND.value, dtype=np.uint8)
    states[component_ground_mask] = EvidenceState.GROUND.value
    for component_id in primary.candidate_ids:
        states[primary.component_positions[component_id]] = (
            EvidenceState.AMBIGUOUS.value
        )

    component_trace: FrameComponentTrace
    if primary.selected_id is None:
        component_trace = FrameComponentTrace(
            status="insufficient_evidence",
            frame_role=FrameRole.TRAJECTORY_ONLY,
            reason_codes=("component_not_separable",),
            component_count=len(primary.component_positions),
            candidate_component_count=len(primary.candidate_ids),
            selected_component_id=None,
            selected_point_count=0,
            selected_voxel_count=0,
            seed_point_count=0,
            component_dominance=None,
            nearest_competing_distance_m=None,
            robust_spread_xyz_m=None,
            resolution_stability_iou=None,
        )
    else:
        selected_positions = primary.component_positions[primary.selected_id]
        secondary = _partition_components(
            roi_local,
            eligible_mask,
            seed_mask,
            settings,
            settings.component_voxel_size_m * settings.stability_voxel_scale,
        )
        secondary_positions = (
            np.empty(0, dtype=np.int64)
            if secondary.selected_id is None
            else secondary.component_positions[secondary.selected_id]
        )
        stability = _jaccard(selected_positions, secondary_positions)
        selected_seed_count = primary.seed_counts[primary.selected_id]
        total_candidate_seed_count = sum(
            primary.seed_counts[component_id] for component_id in primary.candidate_ids
        )
        dominance = selected_seed_count / total_candidate_seed_count
        spread = _robust_spread(roi_local[selected_positions], settings.spread_quantile)
        competitor_distance = _nearest_competing_distance(
            roi_local,
            primary,
            primary.selected_id,
        )
        component_too_broad = any(
            actual > coarse + allowance
            for actual, coarse, allowance in zip(
                spread,
                coarse_box.size_lwh,
                settings.maximum_selected_spread_allowance_xyz_m,
                strict=True,
            )
        )
        if component_too_broad:
            status = "ambiguous"
            role = FrameRole.TRAJECTORY_ONLY
            reasons = ("component_not_separable",)
        else:
            status = "selected"
            states[selected_positions] = EvidenceState.TARGET.value
            role, reasons = _classify_frame_role(
                point_count=len(selected_positions),
                voxel_count=primary.voxel_counts[primary.selected_id],
                spread=spread,
                dominance=dominance,
                stability=stability,
                has_ground=ground is not None,
                settings=settings,
            )
        component_trace = FrameComponentTrace(
            status=status,
            frame_role=role,
            reason_codes=reasons,
            component_count=len(primary.component_positions),
            candidate_component_count=len(primary.candidate_ids),
            selected_component_id=primary.selected_id,
            selected_point_count=len(selected_positions),
            selected_voxel_count=primary.voxel_counts[primary.selected_id],
            seed_point_count=selected_seed_count,
            component_dominance=dominance,
            nearest_competing_distance_m=competitor_distance,
            robust_spread_xyz_m=spread,
            resolution_stability_iou=stability,
        )
    return FrameEvidenceTrace(
        frame_id=frame.frame_id,
        roi_point_indices=roi_indices,
        point_states=states,
        ground_plane=ground,
        represented_sensor_ids=_represented_sensors(frame, roi_indices),
        component=component_trace,
    )


def _partition_components(
    local: NDArray[np.float64],
    eligible_mask: NDArray[np.bool_],
    seed_mask: NDArray[np.bool_],
    settings: ComponentConsensusSettings,
    voxel_size_m: float,
) -> _Partition:
    eligible_positions = np.flatnonzero(eligible_mask).astype(np.int64, copy=False)
    labels = np.full(len(local), -1, dtype=np.int32)
    if not len(eligible_positions):
        return _Partition(labels, (), (), None, (), ())
    cells = np.floor(local[eligible_positions, :3] / voxel_size_m).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    cell_lookup = {
        (int(cell[0]), int(cell[1]), int(cell[2])): index
        for index, cell in enumerate(unique_cells)
    }
    cell_component = np.full(len(unique_cells), -1, dtype=np.int32)
    component_id = 0
    for start in range(len(unique_cells)):
        if cell_component[start] >= 0:
            continue
        cell_component[start] = component_id
        queue = [start]
        while queue:
            current = queue.pop()
            x, y, z = unique_cells[current]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor = cell_lookup.get(
                            (int(x + dx), int(y + dy), int(z + dz))
                        )
                        if neighbor is not None and cell_component[neighbor] < 0:
                            cell_component[neighbor] = component_id
                            queue.append(neighbor)
        component_id += 1
    raw_point_components = cell_component[inverse]
    raw_positions = tuple(
        eligible_positions[raw_point_components == value]
        for value in range(component_id)
    )
    retained = tuple(
        positions
        for positions in raw_positions
        if len(positions) >= settings.minimum_component_points
    )
    seed_counts = tuple(
        int(np.count_nonzero(seed_mask[positions])) for positions in retained
    )
    voxel_counts = tuple(
        len(
            np.unique(
                np.floor(local[positions, :3] / voxel_size_m).astype(np.int64),
                axis=0,
            )
        )
        for positions in retained
    )
    for retained_id, positions in enumerate(retained):
        labels[positions] = retained_id
    candidate_ids = tuple(
        index
        for index, seed_count in enumerate(seed_counts)
        if seed_count >= settings.minimum_seed_points
    )
    selected_id = None
    if candidate_ids:
        selected_id = min(
            candidate_ids,
            key=lambda index: (
                -seed_counts[index],
                -len(retained[index]),
                float(np.linalg.norm(np.median(local[retained[index], :2], axis=0))),
                index,
            ),
        )
    return _Partition(
        labels,
        retained,
        candidate_ids,
        selected_id,
        seed_counts,
        voxel_counts,
    )


def _classify_frame_role(
    *,
    point_count: int,
    voxel_count: int,
    spread: tuple[float, float, float],
    dominance: float,
    stability: float,
    has_ground: bool,
    settings: ComponentConsensusSettings,
) -> tuple[FrameRole, tuple[str, ...]]:
    geometry_checks = (
        point_count >= settings.geometry_minimum_points,
        voxel_count >= settings.geometry_minimum_voxels,
        all(
            actual >= required
            for actual, required in zip(
                spread, settings.geometry_minimum_spread_xyz_m, strict=True
            )
        ),
        dominance >= settings.geometry_minimum_dominance,
        stability >= settings.geometry_minimum_stability_iou,
        has_ground,
    )
    if all(geometry_checks):
        return FrameRole.GEOMETRY, ()
    pose_checks = (
        point_count >= settings.pose_minimum_points,
        voxel_count >= settings.pose_minimum_voxels,
        max(spread[:2]) >= settings.pose_minimum_horizontal_spread_m,
        spread[2] >= settings.pose_minimum_vertical_spread_m,
        dominance >= settings.pose_minimum_dominance,
        stability >= settings.pose_minimum_stability_iou,
    )
    reasons: list[str] = []
    if point_count < settings.geometry_minimum_points:
        reasons.append("insufficient_component_points")
    if dominance < settings.geometry_minimum_dominance:
        reasons.append("weak_component_dominance")
    if stability < settings.geometry_minimum_stability_iou:
        reasons.append("unstable_component_resolution")
    if not all(
        actual >= required
        for actual, required in zip(
            spread, settings.geometry_minimum_spread_xyz_m, strict=True
        )
    ):
        reasons.append("insufficient_geometry_spread")
    if not has_ground:
        reasons.append("ground_support_unavailable")
    return (
        (FrameRole.POSE_ONLY if all(pose_checks) else FrameRole.TRAJECTORY_ONLY),
        tuple(reasons),
    )


def _robust_spread(
    points: NDArray[np.float64], quantile: float
) -> tuple[float, float, float]:
    lower = np.quantile(points, quantile, axis=0)
    upper = np.quantile(points, 1.0 - quantile, axis=0)
    return tuple(float(value) for value in upper - lower)


def _jaccard(left: NDArray[np.int64], right: NDArray[np.int64]) -> float:
    intersection = len(np.intersect1d(left, right, assume_unique=True))
    union = len(left) + len(right) - intersection
    return 0.0 if union == 0 else intersection / union


def _nearest_competing_distance(
    local: NDArray[np.float64], partition: _Partition, selected_id: int
) -> float | None:
    competitors = [value for value in partition.candidate_ids if value != selected_id]
    if not competitors:
        return None
    selected = local[partition.component_positions[selected_id], :2]
    selected_min = selected.min(axis=0)
    selected_max = selected.max(axis=0)
    distances = []
    for component_id in competitors:
        other = local[partition.component_positions[component_id], :2]
        other_min = other.min(axis=0)
        other_max = other.max(axis=0)
        gap = np.maximum(
            np.maximum(other_min - selected_max, selected_min - other_max), 0
        )
        distances.append(float(np.linalg.norm(gap)))
    return min(distances)


def _represented_sensors(
    frame: FrameCloud, roi_indices: NDArray[np.int64]
) -> tuple[str, ...]:
    if frame.point_sensor_index is None or not len(roi_indices):
        return ()
    indices = sorted(set(int(value) for value in frame.point_sensor_index[roi_indices]))
    return tuple(frame.sensor_ids[index] for index in indices)
