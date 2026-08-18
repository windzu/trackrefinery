from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackrefinery import (
    EvidenceSelectionSettings,
    EvidenceState,
    FrameCloud,
    GeometricRefinementSettings,
    InferenceDataset,
    JointCuboidRefiner,
    RefinementCase,
    read_geometric_trace,
    validate_geometric_trace,
    write_geometric_trace,
)


def _case(case_id: str = "static_complete") -> RefinementCase:
    root = Path(__file__).parent / "fixtures" / "synthetic_v1" / "inference"
    return InferenceDataset.open(root).load_case(case_id)


def test_joint_refiner_exposes_deterministic_initial_evidence_without_success() -> None:
    case = _case()
    backend = JointCuboidRefiner()

    first = backend.refine_with_trace(case)
    second = backend.refine_with_trace(case)

    assert first.outcome.status == "insufficient_evidence"
    assert first.outcome.reasons == ("algorithm_stage_incomplete",)
    assert first.trace.config_sha256 == second.trace.config_sha256
    assert first.trace.stage == "initial_evidence_v1"
    assert [frame.frame_id for frame in first.trace.frames] == [
        frame.frame_id for frame in case.frames
    ]
    assert first.trace.total_counts["target"] > 0
    assert first.trace.total_counts["ground"] > 0
    for left, right in zip(first.trace.frames, second.trace.frames, strict=True):
        assert left.ground_plane is not None
        assert sum(left.counts.values()) == len(left.roi_point_indices)
        assert not left.roi_point_indices.flags.writeable
        assert not left.point_states.flags.writeable
        assert left.represented_sensor_ids == ("lidar_front", "lidar_left")
        np.testing.assert_array_equal(left.roi_point_indices, right.roi_point_indices)
        np.testing.assert_array_equal(left.point_states, right.point_states)
        assert set(np.unique(left.point_states)).issubset(
            {state.value for state in EvidenceState}
        )


def test_initial_evidence_is_invariant_to_input_point_order() -> None:
    case = _case("robust_outliers")
    permutations: list[np.ndarray] = []
    permuted_frames = []
    for frame in case.frames:
        permutation = np.arange(len(frame.points_xyz) - 1, -1, -1, dtype=np.int64)
        permutations.append(permutation)
        permuted_frames.append(
            FrameCloud(
                frame_id=frame.frame_id,
                timestamp_ns=frame.timestamp_ns,
                annotation_frame_id=frame.annotation_frame_id,
                world_from_annotation=frame.world_from_annotation,
                points_xyz=frame.points_xyz[permutation].copy(),
                point_features=(
                    None
                    if frame.point_features is None
                    else frame.point_features[permutation].copy()
                ),
                feature_names=frame.feature_names,
                point_timestamps_ns=(
                    None
                    if frame.point_timestamps_ns is None
                    else frame.point_timestamps_ns[permutation].copy()
                ),
                point_sensor_index=(
                    None
                    if frame.point_sensor_index is None
                    else frame.point_sensor_index[permutation].copy()
                ),
                sensor_ids=frame.sensor_ids,
                sensor_origins=frame.sensor_origins,
            )
        )
    permuted = RefinementCase(case.case_id, tuple(permuted_frames), case.track)

    original_trace = JointCuboidRefiner().refine_with_trace(case).trace
    permuted_trace = JointCuboidRefiner().refine_with_trace(permuted).trace

    for frame, permutation, original, reordered in zip(
        case.frames,
        permutations,
        original_trace.frames,
        permuted_trace.frames,
        strict=True,
    ):
        original_states = np.zeros(len(frame.points_xyz), dtype=np.uint8)
        original_states[original.roi_point_indices] = original.point_states
        reordered_states = np.zeros(len(frame.points_xyz), dtype=np.uint8)
        reordered_states[permutation[reordered.roi_point_indices]] = (
            reordered.point_states
        )
        np.testing.assert_array_equal(original_states, reordered_states)


def test_evidence_trace_sidecar_round_trip(tmp_path: Path) -> None:
    case = _case()
    trace = JointCuboidRefiner().refine_with_trace(case).trace

    manifest_path, arrays_path = write_geometric_trace(tmp_path, trace)
    restored = read_geometric_trace(manifest_path)

    assert manifest_path.is_file()
    assert arrays_path.is_file()
    assert restored.to_summary_dict() == trace.to_summary_dict()
    validate_geometric_trace(case, restored)
    for expected, actual in zip(trace.frames, restored.frames, strict=True):
        np.testing.assert_array_equal(
            expected.roi_point_indices, actual.roi_point_indices
        )
        np.testing.assert_array_equal(expected.point_states, actual.point_states)


def test_geometric_settings_are_validated_and_content_addressed() -> None:
    default = GeometricRefinementSettings()
    changed = GeometricRefinementSettings(
        evidence=EvidenceSelectionSettings(roi_margin_xyz_m=(1.1, 1.0, 0.55))
    )

    assert len(default.sha256) == 64
    assert default.sha256 != changed.sha256
    assert default.sha256 == GeometricRefinementSettings().sha256
    with pytest.raises(ValueError, match="ROI margins"):
        EvidenceSelectionSettings(
            roi_margin_xyz_m=(0.1, 0.1, 0.1),
            ambiguity_margin_xyz_m=(0.4, 0.4, 0.22),
        )
    with pytest.raises(TypeError, match="EvidenceSelectionSettings"):
        GeometricRefinementSettings(evidence=None)  # type: ignore[arg-type]
