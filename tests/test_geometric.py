from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackrefinery import (
    EnvelopeFittingSettings,
    EvidenceSelectionSettings,
    EvidenceState,
    FrameCloud,
    GeometricRefinementSettings,
    InferenceDataset,
    JointCuboidRefiner,
    RefinementCase,
    RegistrationSettings,
    TargetDataset,
    read_geometric_trace,
    select_initial_evidence,
    validate_geometric_trace,
    write_geometric_trace,
)
from trackrefinery.geometry import angle_difference, compose_pose, yaw_from_quaternion


def _case(case_id: str = "static_complete") -> RefinementCase:
    root = Path(__file__).parent / "fixtures" / "synthetic_v1" / "inference"
    return InferenceDataset.open(root).load_case(case_id)


def test_joint_refiner_exposes_deterministic_envelope_without_success() -> None:
    case = _case()
    backend = JointCuboidRefiner()

    first = backend.refine_with_trace(case)
    second = backend.refine_with_trace(case)

    assert first.outcome.status == "insufficient_evidence"
    assert first.outcome.reasons == ("algorithm_stage_incomplete",)
    assert first.trace.config_sha256 == second.trace.config_sha256
    assert first.trace.stage == "alternating_envelope_v1"
    assert first.trace.cuboid_fit is not None
    assert first.trace.cuboid_fit.status == "candidate"
    assert first.trace.cuboid_fit.converged is True
    assert first.trace.cuboid_fit.to_dict() == second.trace.cuboid_fit.to_dict()
    assert first.trace.canonical_shape is not None
    assert first.trace.canonical_shape.registered_frame_ids == tuple(
        frame.frame_id for frame in case.frames
    )
    np.testing.assert_array_equal(
        first.trace.canonical_shape.points_xyz,
        second.trace.canonical_shape.points_xyz,
    )
    np.testing.assert_array_equal(
        first.trace.canonical_shape.frame_support_count,
        second.trace.canonical_shape.frame_support_count,
    )
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
        assert left.registration is not None
        assert left.registration.status == "registered"
        assert left.registration.candidate_pose_annotation is not None
        assert left.registration.final_rmse_m is not None
        assert left.registration.initial_rmse_m is not None
        assert left.registration.final_rmse_m < left.registration.initial_rmse_m
        assert left.registration.to_dict() == right.registration.to_dict()
        np.testing.assert_array_equal(left.roi_point_indices, right.roi_point_indices)
        np.testing.assert_array_equal(left.point_states, right.point_states)
        assert set(np.unique(left.point_states)).issubset(
            {state.value for state in EvidenceState}
        )


def test_initial_evidence_api_remains_a_separate_trace_stage() -> None:
    trace = select_initial_evidence(_case())

    assert trace.stage == "initial_evidence_v1"
    assert trace.canonical_shape is None
    assert all(frame.registration is None for frame in trace.frames)


def test_sparse_frames_remain_explicitly_unobservable() -> None:
    run = JointCuboidRefiner().refine_with_trace(_case("partial_visibility"))

    assert run.outcome.status == "insufficient_evidence"
    assert "optimization_not_converged" in run.outcome.reasons
    assert any(
        reason.startswith("pose_unobservable:") for reason in run.outcome.reasons
    )
    assert any(
        frame.registration is not None
        and frame.registration.status == "insufficient_evidence"
        for frame in run.trace.frames
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
        assert original.registration is not None
        assert reordered.registration is not None
        assert original.registration.to_dict() == reordered.registration.to_dict()
    assert original_trace.canonical_shape is not None
    assert permuted_trace.canonical_shape is not None
    np.testing.assert_array_equal(
        original_trace.canonical_shape.points_xyz,
        permuted_trace.canonical_shape.points_xyz,
    )
    np.testing.assert_array_equal(
        original_trace.canonical_shape.frame_support_count,
        permuted_trace.canonical_shape.frame_support_count,
    )


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
        assert expected.registration == actual.registration
    assert trace.canonical_shape is not None
    assert restored.canonical_shape is not None
    np.testing.assert_array_equal(
        trace.canonical_shape.points_xyz, restored.canonical_shape.points_xyz
    )
    np.testing.assert_array_equal(
        trace.canonical_shape.frame_support_count,
        restored.canonical_shape.frame_support_count,
    )


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
    with pytest.raises(ValueError, match="minimum_correspondences"):
        RegistrationSettings(
            minimum_target_points=20,
            minimum_correspondences=21,
        )
    with pytest.raises(ValueError, match="ambiguity allowance"):
        EnvelopeFittingSettings(
            target_envelope_allowance_m=0.3,
            ambiguity_envelope_allowance_m=0.2,
        )


def test_registration_improves_clean_synthetic_pose_without_reading_targets() -> None:
    case = _case("moving_complete")
    target_root = Path(__file__).parent / "fixtures" / "synthetic_v1" / "targets"
    target = TargetDataset.open(target_root).load_target(case.case_id)

    trace = JointCuboidRefiner().refine_with_trace(case).trace

    coarse_xy: list[float] = []
    candidate_xy: list[float] = []
    coarse_yaw: list[float] = []
    candidate_yaw: list[float] = []
    for observation, frame_trace, gold in zip(
        case.track.observations, trace.frames, target.frame_poses, strict=True
    ):
        registration = frame_trace.registration
        assert registration is not None
        assert registration.candidate_pose_annotation is not None
        coarse_xy.append(
            float(
                np.linalg.norm(
                    np.asarray(observation.coarse_box.center[:2])
                    - np.asarray(gold.pose.translation_xyz[:2])
                )
            )
        )
        candidate_xy.append(
            float(
                np.linalg.norm(
                    np.asarray(
                        registration.candidate_pose_annotation.translation_xyz[:2]
                    )
                    - np.asarray(gold.pose.translation_xyz[:2])
                )
            )
        )
        coarse_yaw.append(
            abs(
                angle_difference(
                    yaw_from_quaternion(observation.coarse_box.orientation_xyzw),
                    yaw_from_quaternion(gold.pose.orientation_xyzw),
                )
            )
        )
        candidate_yaw.append(
            abs(
                angle_difference(
                    yaw_from_quaternion(
                        registration.candidate_pose_annotation.orientation_xyzw
                    ),
                    yaw_from_quaternion(gold.pose.orientation_xyzw),
                )
            )
        )
    assert np.median(candidate_xy) < np.median(coarse_xy) * 0.7
    assert np.median(candidate_yaw) < np.median(coarse_yaw) * 0.25


def test_visible_envelope_recovers_clean_size_without_publishing_success() -> None:
    case = _case("moving_complete")
    target_root = Path(__file__).parent / "fixtures" / "synthetic_v1" / "targets"
    target = TargetDataset.open(target_root).load_target(case.case_id)

    run = JointCuboidRefiner().refine_with_trace(case)

    assert run.outcome.status == "insufficient_evidence"
    assert run.outcome.reasons == ("algorithm_stage_incomplete",)
    fit = run.trace.cuboid_fit
    assert fit is not None
    assert fit.status == "candidate"
    assert fit.converged is True
    assert fit.canonical_size_lwh is not None
    np.testing.assert_allclose(
        fit.canonical_size_lwh,
        target.canonical_size_lwh,
        atol=0.01,
        rtol=0.0,
    )


def test_registration_is_invariant_to_world_coordinate_gauge() -> None:
    case = _case("static_complete")
    global_from_world = compose_pose(
        case.frames[0].world_from_annotation,
        case.frames[1].world_from_annotation,
    )
    transformed_frames = tuple(
        FrameCloud(
            frame_id=frame.frame_id,
            timestamp_ns=frame.timestamp_ns,
            annotation_frame_id=frame.annotation_frame_id,
            world_from_annotation=compose_pose(
                global_from_world, frame.world_from_annotation
            ),
            points_xyz=frame.points_xyz,
            point_features=frame.point_features,
            feature_names=frame.feature_names,
            point_timestamps_ns=frame.point_timestamps_ns,
            point_sensor_index=frame.point_sensor_index,
            sensor_ids=frame.sensor_ids,
            sensor_origins=frame.sensor_origins,
        )
        for frame in case.frames
    )
    transformed = RefinementCase(case.case_id, transformed_frames, case.track)

    original = JointCuboidRefiner().refine_with_trace(case).trace
    changed = JointCuboidRefiner().refine_with_trace(transformed).trace

    for left, right in zip(original.frames, changed.frames, strict=True):
        assert left.registration is not None
        assert right.registration is not None
        assert left.registration.candidate_pose_annotation is not None
        assert right.registration.candidate_pose_annotation is not None
        np.testing.assert_allclose(
            left.registration.candidate_pose_annotation.translation_xyz,
            right.registration.candidate_pose_annotation.translation_xyz,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            left.registration.candidate_pose_annotation.orientation_xyzw,
            right.registration.candidate_pose_annotation.orientation_xyzw,
            atol=1e-12,
        )
