from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from trackrefinery import (
    ComponentConsensusSettings,
    FrameRole,
    InferenceDataset,
    ObservableCoreRefiner,
    ObservableCoreSettings,
    ObservableFrameDisposition,
    RefinementCase,
    build_review_bundle,
    read_geometric_trace,
    select_object_components,
    select_observable_core,
    write_geometric_trace,
)


def _case(case_id: str) -> RefinementCase:
    root = Path(__file__).parent / "fixtures" / "synthetic_v1" / "inference"
    return InferenceDataset.open(root).load_case(case_id)


def _settings(**overrides: object) -> ObservableCoreSettings:
    component_values = {
        "geometry_minimum_points": 80,
        "geometry_minimum_voxels": 20,
        "track_minimum_geometry_frames": 2,
        "maximum_outside_envelope_fraction": 1.0,
        **overrides,
    }
    return ObservableCoreSettings(
        component=ComponentConsensusSettings(**component_values)
    )


def test_selector_uses_one_connected_run_and_downgrades_isolated_geometry() -> None:
    case = _case("moving_complete")
    settings = _settings(geometry_minimum_relative_points=0.7)
    component_trace = select_object_components(case, settings.component)

    selection, selected_trace = select_observable_core(case, component_trace, settings)
    repeated, repeated_trace = select_observable_core(case, component_trace, settings)

    assert selection.status == "candidate"
    assert repeated.to_dict() == selection.to_dict()
    assert repeated_trace.to_summary_dict() == selected_trace.to_summary_dict()
    assert selection.core_frame_ids == tuple(
        frame.frame_id for frame in case.frames[:2]
    )
    assert selection.rejected_geometry_frame_ids == (case.frames[3].frame_id,)
    assert [run.frame_ids for run in selection.candidate_runs] == [
        tuple(frame.frame_id for frame in case.frames[:2]),
        (case.frames[3].frame_id,),
    ]
    assert [item.disposition for item in selection.frames] == [
        ObservableFrameDisposition.CORE_GEOMETRY,
        ObservableFrameDisposition.CORE_GEOMETRY,
        ObservableFrameDisposition.POSE_CANDIDATE,
        ObservableFrameDisposition.POSE_CANDIDATE,
        ObservableFrameDisposition.POSE_CANDIDATE,
    ]
    isolated = selected_trace.frames[3].component
    assert isolated is not None
    assert isolated.frame_role is FrameRole.POSE_ONLY
    assert "outside_selected_observable_core" in isolated.reason_codes


def test_selector_breaks_a_geometry_run_at_relative_timestamp_discontinuity() -> None:
    source = _case("static_complete")
    timestamps = (0, 100, 200, 1_200, 1_300)
    frames = tuple(
        replace(frame, timestamp_ns=timestamp)
        for frame, timestamp in zip(source.frames, timestamps, strict=True)
    )
    case = RefinementCase(source.case_id, frames, source.track)
    settings = _settings()
    component_trace = select_object_components(case, settings.component)

    selection, _ = select_observable_core(case, component_trace, settings)

    assert selection.reference_interval_ns == 100
    assert selection.maximum_connected_gap_ns == 250
    assert [run.frame_ids for run in selection.candidate_runs] == [
        tuple(frame.frame_id for frame in frames[:3]),
        tuple(frame.frame_id for frame in frames[3:]),
    ]
    assert selection.core_frame_ids == tuple(frame.frame_id for frame in frames[:3])


def test_selector_rejects_disconnected_geometry_below_minimum_run_length() -> None:
    case = _case("moving_complete")
    settings = _settings(
        geometry_minimum_relative_points=0.7,
        track_minimum_geometry_frames=3,
    )
    component_trace = select_object_components(case, settings.component)

    selection, selected_trace = select_observable_core(case, component_trace, settings)

    assert selection.status == "insufficient_evidence"
    assert selection.reason_codes == ("no_connected_geometry_core",)
    assert selection.core_frame_ids == ()
    assert all(
        frame.component is not None
        and frame.component.frame_role is not FrameRole.GEOMETRY
        for frame in selected_trace.frames
    )


def test_observable_core_settings_reject_invalid_relative_gap() -> None:
    with pytest.raises(ValueError, match="maximum_timestamp_gap_factor"):
        ObservableCoreSettings(maximum_timestamp_gap_factor=0.9)


def test_observable_core_refiner_aggregates_only_selected_core_and_is_portable(
    tmp_path: Path,
) -> None:
    case = _case("static_complete")
    run = ObservableCoreRefiner(_settings()).refine_with_trace(case)

    assert run.outcome.status == "insufficient_evidence"
    assert run.outcome.reasons == ("algorithm_stage_incomplete",)
    assert run.trace.stage == "observable_core_aggregation_v1"
    assert run.trace.anchored_aggregation is not None
    assert run.trace.anchored_aggregation.status == "candidate"
    observable_core = run.outcome.diagnostics["observable_core"]
    assert observable_core["status"] == "candidate"
    assert observable_core["core_frame_ids"] == tuple(
        frame.frame_id for frame in case.frames
    )

    manifest, _ = write_geometric_trace(tmp_path / "trace", run.trace)
    restored = read_geometric_trace(manifest)
    assert restored.to_summary_dict() == run.trace.to_summary_dict()


def test_review_bundle_exposes_observable_core_selection(tmp_path: Path) -> None:
    case = _case("moving_complete")
    settings = _settings(geometry_minimum_relative_points=0.7)
    run = ObservableCoreRefiner(settings).refine_with_trace(case)

    bundle = build_review_bundle(
        case,
        run.outcome,
        tmp_path / "review",
        trace=run.trace,
        data_source="synthetic observable-core selector test",
        max_points_per_frame=200,
    )

    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    assert manifest["algorithm_stage"] == "observable_core_aggregation_v1"
    assert manifest["observable_core_status"] == "candidate"
    assert manifest["observable_core_frame_ids"] == [
        frame.frame_id for frame in case.frames[:2]
    ]
    assert manifest["observable_core"]["rejected_geometry_frame_ids"] == [
        case.frames[3].frame_id
    ]
    html = (bundle / "preview.html").read_text(encoding="utf-8")
    assert "Observable-core aggregation" in html
