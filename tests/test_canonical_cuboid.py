from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trackrefinery import (
    CANONICAL_CUBOID_EXPERIMENT_CONTRACT,
    CANONICAL_CUBOID_STAGE,
    CanonicalCuboidExperimentSettings,
    ComponentConsensusSettings,
    EvidenceState,
    InferenceDataset,
    InsufficientEvidence,
    Pose3D,
    TargetDataset,
    build_clip_review_suite,
    build_review_bundle,
    fit_observable_canonical_cuboid,
    select_object_components,
)
from trackrefinery.component_consensus.aggregation import (
    _canonical_shape,
    _geometry_states,
    _registration_trace,
    aggregate_geometry_components,
)
from trackrefinery.geometric.trace import FrameRole
from trackrefinery.geometry import (
    angle_difference,
    compose_pose,
    inverse_pose,
    inverse_transform_points,
    quaternion_matrix,
    transform_points,
    yaw_from_quaternion,
)


def _case():
    root = Path(__file__).parent / "fixtures" / "synthetic_v1" / "inference"
    return InferenceDataset.open(root).load_case("static_complete")


def _target():
    root = Path(__file__).parent / "fixtures" / "synthetic_v1" / "targets"
    return TargetDataset.open(root).load_target("static_complete")


def _component_settings() -> ComponentConsensusSettings:
    return ComponentConsensusSettings(
        geometry_minimum_points=80,
        geometry_minimum_voxels=20,
        track_minimum_geometry_frames=2,
        maximum_outside_envelope_fraction=0.1,
    )


def _exact_stage3_trace(*, remove_length_upper: bool = False):
    """Evaluator-only exact relative alignment; targets never reach Stage 4."""

    case = _case()
    target = _target()
    settings = _component_settings()
    target_poses = {item.frame_id: item.pose for item in target.frame_poses}
    components = select_object_components(case, settings)
    if remove_length_upper:
        updated = []
        for frame, frame_trace in zip(case.frames, components.frames, strict=True):
            pose = target_poses[frame.frame_id]
            positions = np.flatnonzero(
                frame_trace.point_states == EvidenceState.TARGET.value
            )
            indices = frame_trace.roi_point_indices[positions]
            local = inverse_transform_points(frame.points_xyz[indices], pose)
            states = frame_trace.point_states.copy()
            states[positions[local[:, 0] > 1.3]] = EvidenceState.BACKGROUND.value
            updated.append(replace(frame_trace, point_states=states))
        components = replace(components, frames=tuple(updated))

    states = _geometry_states(case, components, settings)
    for state in states:
        frame = case.frames[state.index]
        coarse = case.track.observations[state.index].coarse_box.pose
        target_pose = target_poses[frame.frame_id]
        world_from_coarse = compose_pose(frame.world_from_annotation, coarse)
        world_from_target = compose_pose(frame.world_from_annotation, target_pose)
        correction = compose_pose(inverse_pose(world_from_target), world_from_coarse)
        state.rotation = quaternion_matrix(correction.orientation_xyzw)
        state.translation = np.asarray(correction.translation_xyz)
        state.accepted = True
        state.iterations = 1
        state.correspondence_count = len(state.reduced_points)
        state.initial_rmse_m = 0.0
        state.final_rmse_m = 0.0
    canonical = _canonical_shape(case, states, settings)
    assert canonical is not None
    sequential = aggregate_geometry_components(case, components, settings)
    frames = tuple(
        replace(frame, registration=_registration_trace(case, states[index]))
        for index, frame in enumerate(components.frames)
    )
    return (
        case,
        target,
        replace(
            sequential,
            stage="pose_graph_aggregation_v3_experiment",
            frames=frames,
            canonical_shape=canonical,
        ),
    )


def _with_common_registration_gauge(case, target, stage3, correction: Pose3D):
    target_poses = {item.frame_id: item.pose for item in target.frame_poses}
    inverse_correction = inverse_pose(correction)
    frames = []
    for index, frame in enumerate(stage3.frames):
        registration = frame.registration
        assert registration is not None
        stage3_pose = compose_pose(target_poses[frame.frame_id], inverse_correction)
        world_from_coarse = compose_pose(
            case.frames[index].world_from_annotation,
            case.track.observations[index].coarse_box.pose,
        )
        world_from_stage3 = compose_pose(
            case.frames[index].world_from_annotation, stage3_pose
        )
        canonical_from_coarse = compose_pose(
            inverse_pose(world_from_stage3), world_from_coarse
        )
        frames.append(
            replace(
                frame,
                registration=replace(
                    registration,
                    canonical_from_coarse=canonical_from_coarse,
                    candidate_pose_annotation=stage3_pose,
                    translation_correction_m=float(
                        np.linalg.norm(canonical_from_coarse.translation_xyz[:2])
                    ),
                    yaw_correction_deg=abs(
                        np.degrees(
                            yaw_from_quaternion(canonical_from_coarse.orientation_xyzw)
                        )
                    ),
                ),
            )
        )
    shape = stage3.canonical_shape
    assert shape is not None
    return replace(
        stage3,
        frames=tuple(frames),
        canonical_shape=replace(
            shape,
            points_xyz=transform_points(shape.points_xyz, correction).astype(
                np.float32
            ),
        ),
    )


def test_observable_canonical_cuboid_recovers_exact_aligned_synthetic_size(
    tmp_path: Path,
) -> None:
    case, target, stage3 = _exact_stage3_trace()

    first = fit_observable_canonical_cuboid(case, stage3)
    second = fit_observable_canonical_cuboid(case, stage3)

    assert first.canonical_cuboid.to_dict() == second.canonical_cuboid.to_dict()
    assert first.trace.to_summary_dict() == second.trace.to_summary_dict()
    assert first.canonical_cuboid.status == "candidate"
    assert first.trace.stage == CANONICAL_CUBOID_STAGE
    fit = first.trace.cuboid_fit
    assert fit is not None
    assert fit.status == "candidate"
    assert fit.canonical_size_lwh is not None
    np.testing.assert_allclose(
        fit.canonical_size_lwh,
        target.canonical_size_lwh,
        atol=0.06,
        rtol=0.0,
    )
    assert all(face.accepted for face in first.canonical_cuboid.face_support)
    assert first.canonical_cuboid.provisional_center_in_registration_xyz is not None
    assert (
        np.linalg.norm(first.canonical_cuboid.provisional_center_in_registration_xyz)
        < 0.06
    )
    assert first.canonical_cuboid.provisional_yaw_in_registration_deg is not None
    assert abs(first.canonical_cuboid.provisional_yaw_in_registration_deg) < 0.5

    output = first.canonical_cuboid.write_json(tmp_path / "canonical-cuboid.json")
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored["contract_version"] == CANONICAL_CUBOID_EXPERIMENT_CONTRACT
    assert restored["status"] == "candidate"


def test_canonical_cuboid_rejects_missing_supported_length_face() -> None:
    case, _, stage3 = _exact_stage3_trace(remove_length_upper=True)

    run = fit_observable_canonical_cuboid(case, stage3)

    assert run.canonical_cuboid.status == "insufficient_evidence"
    assert "insufficient_face_support:length_upper" in run.canonical_cuboid.reason_codes
    assert run.canonical_cuboid.provisional_size_lwh is not None
    assert run.trace.cuboid_fit is not None
    assert run.trace.cuboid_fit.status == "insufficient_evidence"
    assert run.trace.cuboid_fit.canonical_size_lwh is None


def test_canonical_cuboid_recovers_common_center_and_yaw_gauge() -> None:
    case, target, stage3 = _exact_stage3_trace()
    yaw = np.deg2rad(2.0)
    correction = Pose3D(
        (0.12, -0.08, 0.05),
        (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))),
    )
    stage3 = _with_common_registration_gauge(case, target, stage3, correction)

    run = fit_observable_canonical_cuboid(
        case,
        stage3,
        CanonicalCuboidExperimentSettings(leave_one_out_maximum_yaw_change_deg=1.0),
    )

    assert run.canonical_cuboid.status == "candidate"
    fitted_center = run.canonical_cuboid.provisional_center_in_registration_xyz
    fitted_yaw = run.canonical_cuboid.provisional_yaw_in_registration_deg
    assert fitted_center is not None and fitted_yaw is not None
    np.testing.assert_allclose(fitted_center, correction.translation_xyz, atol=0.06)
    assert abs(fitted_yaw - 2.0) < 0.5
    target_poses = {item.frame_id: item.pose for item in target.frame_poses}
    for frame in run.trace.frames:
        registration = frame.registration
        assert registration is not None
        candidate = registration.candidate_pose_annotation
        assert candidate is not None
        expected = target_poses[frame.frame_id]
        np.testing.assert_allclose(
            candidate.translation_xyz, expected.translation_xyz, atol=0.06
        )
        assert (
            abs(
                np.degrees(
                    angle_difference(
                        yaw_from_quaternion(candidate.orientation_xyzw),
                        yaw_from_quaternion(expected.orientation_xyzw),
                    )
                )
            )
            < 0.5
        )


def test_canonical_cuboid_rejects_non_stage3_trace() -> None:
    case = _case()
    components = select_object_components(case, _component_settings())

    run = fit_observable_canonical_cuboid(case, components)

    assert run.canonical_cuboid.status == "insufficient_evidence"
    assert "unsupported_stage3_trace" in run.canonical_cuboid.reason_codes
    assert "stage3_insufficient_evidence" in run.canonical_cuboid.reason_codes


def test_canonical_cuboid_settings_validate_observability_policy() -> None:
    with pytest.raises(ValueError, match="minimum_geometry_frames"):
        CanonicalCuboidExperimentSettings(minimum_geometry_frames=2)
    with pytest.raises(ValueError, match="resolution_scales"):
        CanonicalCuboidExperimentSettings(resolution_scales=(0.75, 1.25))
    with pytest.raises(ValueError, match="boundary_tail_fraction"):
        CanonicalCuboidExperimentSettings(
            boundary_tail_fraction=0.03,
            boundary_maximum_tail_fraction=0.02,
        )


def test_exact_stage3_fixture_keeps_every_frame_as_geometry() -> None:
    _, _, stage3 = _exact_stage3_trace()

    assert all(
        frame.component is not None and frame.component.frame_role is FrameRole.GEOMETRY
        for frame in stage3.frames
    )


def test_stage4_review_exposes_observability_decision(tmp_path: Path) -> None:
    case, _, stage3 = _exact_stage3_trace()
    run = fit_observable_canonical_cuboid(case, stage3)
    outcome = InsufficientEvidence(
        track_id=case.track.track_id,
        reasons=("stage4_experiment_not_released",),
        diagnostics={"canonical_cuboid": run.canonical_cuboid.to_dict()},
    )
    root = tmp_path / "review"
    bundle = build_review_bundle(
        case,
        outcome,
        root / "cases" / case.case_id,
        trace=run.trace,
        data_source="synthetic evaluator-only exact registration",
        max_points_per_frame=300,
    )
    build_clip_review_suite(root, {"synthetic-clip": [bundle]})

    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    diagnostic = manifest["canonical_cuboid_experiment"]
    assert diagnostic["status"] == "candidate"
    assert len(diagnostic["face_support"]) == 6
    assert manifest["algorithm_stage"] == CANONICAL_CUBOID_STAGE
    preview = (bundle / "preview.html").read_text(encoding="utf-8")
    assert "V4 observable canonical-cuboid alignment" in preview
    assert "Stage 4 cuboid decision" in preview
    assert "all observability gates passed" in preview
    catalog = (root / "index.html").read_text(encoding="utf-8")
    assert "SIZE CANDIDATE · NOT RELEASED" in catalog
    assert "Boundary support:" in catalog
    assert "LOO Δsize" in catalog
    assert "input " in catalog and "→ fitted" in catalog
