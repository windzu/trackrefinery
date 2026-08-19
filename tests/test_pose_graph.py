from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trackrefinery import (
    POSE_GRAPH_EXPERIMENT_CONTRACT,
    ComponentConsensusSettings,
    InferenceDataset,
    PoseGraphExperimentSettings,
    aggregate_geometry_components_pose_graph,
    run_controlled_recovery,
    select_object_components,
)


def _case():
    root = Path(__file__).parent / "fixtures" / "synthetic_v1" / "inference"
    return InferenceDataset.open(root).load_case("static_complete")


def _settings() -> ComponentConsensusSettings:
    return ComponentConsensusSettings(
        geometry_minimum_points=80,
        geometry_minimum_voxels=20,
        track_minimum_geometry_frames=2,
        maximum_outside_envelope_fraction=0.1,
    )


@pytest.mark.parametrize(
    ("variant", "expected_status", "expected_guard"),
    (
        ("point_to_point_pose_graph", "candidate", "accepted"),
        (
            "normal_aware_pose_graph",
            "insufficient_evidence",
            "insufficient_evidence",
        ),
    ),
)
def test_pose_graph_experiment_is_deterministic_and_anchored(
    tmp_path: Path, variant: str, expected_status: str, expected_guard: str
) -> None:
    case = _case()
    settings = _settings()
    components = select_object_components(case, settings)

    first = aggregate_geometry_components_pose_graph(
        case, components, settings, variant=variant
    )
    second = aggregate_geometry_components_pose_graph(
        case, components, settings, variant=variant
    )

    assert first.pose_graph.to_dict() == second.pose_graph.to_dict()
    assert first.trace.to_summary_dict() == second.trace.to_summary_dict()
    assert first.pose_graph.optimizer_success is True
    assert first.pose_graph.final_cost <= first.pose_graph.initial_cost + 1e-9
    aggregation = first.trace.anchored_aggregation
    assert aggregation is not None
    assert aggregation.status == expected_status
    anchor_index = next(
        index
        for index, frame in enumerate(case.frames)
        if frame.frame_id == aggregation.anchor_frame_id
    )
    anchor = first.trace.frames[anchor_index].registration
    assert anchor is not None
    if expected_status == "candidate":
        assert aggregation.accepted_frame_ids[0] == aggregation.anchor_frame_id
        assert anchor.status == "retained_coarse"
        assert anchor.translation_correction_m == 0.0
        assert anchor.yaw_correction_deg == 0.0
    else:
        assert aggregation.reason_codes == ("correction_bound_saturated",)
        assert anchor.status == "insufficient_evidence"

    output = first.pose_graph.write_json(tmp_path / f"{variant}.json")
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored["contract_version"] == POSE_GRAPH_EXPERIMENT_CONTRACT
    assert restored["variant"] == variant
    assert restored["guard_status"] == expected_guard


def test_pose_graph_rejects_low_overlap_single_bridge() -> None:
    case = _case()
    settings = _settings()
    components = select_object_components(case, settings)
    experiment = PoseGraphExperimentSettings(
        temporal_neighbor_count=1,
        connect_anchor_to_all=False,
        pair_minimum_overlap_fraction=0.12,
        bridge_minimum_overlap_fraction=1.0,
    )

    run = aggregate_geometry_components_pose_graph(
        case,
        components,
        settings,
        variant="point_to_point_pose_graph",
        experiment_settings=experiment,
    )

    aggregation = run.trace.anchored_aggregation
    assert aggregation is not None
    assert aggregation.status == "insufficient_evidence"
    assert aggregation.reason_codes == ("weak_partial_overlap_bridge",)
    assert run.pose_graph.optimizer_success is False
    assert run.pose_graph.optimizer_evaluations == 0
    assert run.pose_graph.guard_reason_codes == ("weak_partial_overlap_bridge",)


def test_controlled_recovery_accepts_pose_graph_backend() -> None:
    case = _case()
    settings = _settings()

    def backend(case, trace, settings):
        return aggregate_geometry_components_pose_graph(
            case,
            trace,
            settings,
            variant="normal_aware_pose_graph",
        ).trace

    run = run_controlled_recovery(
        case,
        settings=settings,
        aggregation_backend=backend,
        algorithm_variant="normal_aware_pose_graph",
    )

    assert run.report.algorithm_variant == "normal_aware_pose_graph"
    assert run.reference_output_trace.stage == "pose_graph_aggregation_v3_experiment"
    assert run.output_trace.stage == "pose_graph_aggregation_v3_experiment"
    assert np.isfinite(run.report.equivariant_output_translation_rms_m)
    assert np.isfinite(run.report.equivariant_output_yaw_rms_deg)


def test_pose_graph_settings_reject_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="pair_minimum_overlap_fraction"):
        PoseGraphExperimentSettings(pair_minimum_overlap_fraction=1.1)
    with pytest.raises(ValueError, match="bridge_minimum_overlap_fraction"):
        PoseGraphExperimentSettings(bridge_minimum_overlap_fraction=1.1)
