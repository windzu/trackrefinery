"""Deterministic initial ROI, ground, and point-state selection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from trackrefinery.contracts import Box3D, FrameCloud, RefinementCase
from trackrefinery.geometric.ground import classify_ground, estimate_ground_plane
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

    ground = estimate_ground_plane(
        roi_points,
        roi_local,
        bottom_local_z=-half_size[2],
        settings=settings,
    )
    ground_mask = classify_ground(
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


def _represented_sensors(
    frame: FrameCloud, roi_indices: NDArray[np.int64]
) -> tuple[str, ...]:
    if frame.point_sensor_index is None or not len(roi_indices):
        return ()
    indices = sorted(set(int(value) for value in frame.point_sensor_index[roi_indices]))
    return tuple(frame.sensor_ids[index] for index in indices)
