from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trackrefinery import (
    ComponentConsensusRefiner,
    ComponentConsensusSettings,
    EvidenceState,
    FrameRole,
    InferenceDataset,
    build_review_bundle,
    read_geometric_trace,
    write_geometric_trace,
)


def _case(case_id: str):
    root = Path(__file__).parent / "fixtures" / "synthetic_v1" / "inference"
    return InferenceDataset.open(root).load_case(case_id)


def test_v2_selects_clean_components_without_publishing_geometry() -> None:
    run = ComponentConsensusRefiner().refine_with_trace(_case("static_complete"))

    assert run.outcome.status == "insufficient_evidence"
    assert run.outcome.reasons == ("algorithm_stage_incomplete",)
    assert run.trace.stage == "component_selection_v2"
    assert run.outcome.diagnostics["frame_role_counts"] == {
        "geometry": 5,
        "pose_only": 0,
        "trajectory_only": 0,
    }
    for frame in run.trace.frames:
        component = frame.component
        assert component is not None
        assert component.status == "selected"
        assert component.frame_role is FrameRole.GEOMETRY
        assert component.selected_point_count == frame.count(EvidenceState.TARGET)
        assert 0 < component.selected_point_count < len(frame.roi_point_indices)
        assert component.resolution_stability_iou is not None
        assert component.resolution_stability_iou >= 0.65


@pytest.mark.parametrize(
    "case_id",
    ("nearby_clutter", "neighboring_tracks", "neighboring_tracks_neighbor"),
)
def test_v2_marks_merged_or_overwide_components_ambiguous(case_id: str) -> None:
    run = ComponentConsensusRefiner().refine_with_trace(_case(case_id))

    assert "insufficient_geometry_frames" in run.outcome.reasons
    for frame in run.trace.frames:
        component = frame.component
        assert component is not None
        assert component.status == "ambiguous"
        assert component.frame_role is FrameRole.TRAJECTORY_ONLY
        assert component.reason_codes == ("component_not_separable",)
        assert frame.count(EvidenceState.TARGET) == 0
        assert frame.count(EvidenceState.AMBIGUOUS) > 0


def test_v2_partial_observations_cannot_define_size() -> None:
    run = ComponentConsensusRefiner().refine_with_trace(_case("partial_visibility"))

    assert run.outcome.diagnostics["frame_role_counts"]["geometry"] == 0
    assert "insufficient_geometry_frames" in run.outcome.reasons
    assert all(
        frame.component is not None
        and frame.component.frame_role is not FrameRole.GEOMETRY
        for frame in run.trace.frames
    )


def test_v2_component_trace_is_deterministic_and_portable(tmp_path: Path) -> None:
    case = _case("moving_complete")
    first = ComponentConsensusRefiner().refine_with_trace(case)
    second = ComponentConsensusRefiner().refine_with_trace(case)

    assert first.trace.to_summary_dict() == second.trace.to_summary_dict()
    for left, right in zip(first.trace.frames, second.trace.frames, strict=True):
        np.testing.assert_array_equal(left.roi_point_indices, right.roi_point_indices)
        np.testing.assert_array_equal(left.point_states, right.point_states)

    manifest, _ = write_geometric_trace(tmp_path, first.trace)
    restored = read_geometric_trace(manifest)
    assert restored.to_summary_dict() == first.trace.to_summary_dict()
    for expected, actual in zip(first.trace.frames, restored.frames, strict=True):
        assert actual.component == expected.component
        np.testing.assert_array_equal(actual.point_states, expected.point_states)


def test_v2_settings_reject_inverted_role_thresholds() -> None:
    with pytest.raises(ValueError, match="geometry point threshold"):
        ComponentConsensusSettings(
            geometry_minimum_points=20,
            pose_minimum_points=30,
        )
    with pytest.raises(ValueError, match="stability_voxel_scale"):
        ComponentConsensusSettings(stability_voxel_scale=1.0)


def test_v2_review_names_component_selection_without_claiming_registration(
    tmp_path: Path,
) -> None:
    case = _case("static_complete")
    run = ComponentConsensusRefiner().refine_with_trace(case)

    bundle = build_review_bundle(
        case,
        run.outcome,
        tmp_path / "review",
        trace=run.trace,
        data_source="synthetic V2 component test",
        max_points_per_frame=300,
    )

    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    assert manifest["algorithm_stage"] == "component_selection_v2"
    assert manifest["frame_role_counts"] == {
        "geometry": 5,
        "pose_only": 0,
        "trajectory_only": 0,
    }
    assert manifest["has_registration_trace"] is False
    html = (bundle / "preview.html").read_text(encoding="utf-8")
    assert "V2 component selection" in html
    assert "selected object component" in html
    assert "no registration performed" in html
    assert "Frame roles" in html
    assert (bundle / "thumbnails" / "aggregate_evidence_top.png").is_file()
    assert (bundle / "thumbnails" / "aggregate_evidence_side.png").is_file()
