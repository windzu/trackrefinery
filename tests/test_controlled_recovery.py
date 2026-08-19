from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trackrefinery import (
    CONTROLLED_RECOVERY_CONTRACT,
    CONTROLLED_RECOVERY_REVIEW_CONTRACT,
    CONTROLLED_RECOVERY_SUITE_CONTRACT,
    ComponentConsensusRefiner,
    ComponentConsensusSettings,
    ControlledPerturbationProfile,
    InferenceDataset,
    build_controlled_recovery_bundle,
    build_controlled_recovery_suite,
    run_controlled_recovery,
)
from trackrefinery.cli import build_controlled_recovery_suite_main


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


def test_controlled_recovery_freezes_components_and_hides_reference_poses() -> None:
    case = _case()
    settings = _settings()
    baseline = ComponentConsensusRefiner(settings).refine_with_trace(case)
    profile = ControlledPerturbationProfile("test", 0.15, 2.0)

    first = run_controlled_recovery(
        case,
        profile=profile,
        settings=settings,
        component_trace=baseline.trace,
    )
    second = run_controlled_recovery(
        case,
        profile=profile,
        settings=settings,
        component_trace=baseline.trace,
    )

    assert first.component_trace.stage == "component_selection_v2"
    assert first.component_trace.anchored_aggregation is None
    assert first.component_trace.canonical_shape is None
    assert all(frame.registration is None for frame in first.component_trace.frames)
    assert first.output_trace.stage == "anchored_component_aggregation_v2"
    assert first.report.to_dict() == second.report.to_dict()
    assert (
        first.report.anchor_frame_id
        == baseline.trace.anchored_aggregation.anchor_frame_id
    )
    assert first.report.geometry_frame_count == 5
    assert first.report.perturbed_frame_count == 4
    assert first.report.to_dict()["contract_version"] == CONTROLLED_RECOVERY_CONTRACT
    assert first.report.to_dict()["reference_semantics"] == (
        "frozen_model_track_proxy_not_gold"
    )
    assert np.isclose(
        max(item.translation_m for item in first.perturbations),
        profile.maximum_translation_m,
    )
    assert np.isclose(
        max(abs(item.yaw_deg) for item in first.perturbations),
        profile.maximum_yaw_deg,
    )
    anchor = next(
        item
        for item in first.perturbations
        if item.frame_id == first.report.anchor_frame_id
    )
    assert anchor.phase == 0.0
    assert anchor.translation_m == 0.0
    assert anchor.yaw_deg == 0.0
    reference = {item.frame_id: item.coarse_box for item in case.track.observations}
    perturbed = {
        item.frame_id: item.coarse_box
        for item in first.perturbed_case.track.observations
    }
    assert perturbed[anchor.frame_id] == reference[anchor.frame_id]
    assert any(
        perturbed[item.frame_id] != reference[item.frame_id]
        for item in first.perturbations
        if item.frame_id != anchor.frame_id
    )


def test_controlled_recovery_review_has_four_same_point_views(
    tmp_path: Path,
) -> None:
    case = _case()
    settings = _settings()
    baseline = ComponentConsensusRefiner(settings).refine_with_trace(case)
    run = run_controlled_recovery(
        case,
        profile=ControlledPerturbationProfile("strong", 0.15, 2.0),
        settings=settings,
        component_trace=baseline.trace,
    )
    bundle = build_controlled_recovery_bundle(
        run,
        tmp_path / "suite" / "cases" / case.case_id / "strong",
        data_source="synthetic controlled test",
        max_points_per_frame=200,
    )
    suite = build_controlled_recovery_suite(tmp_path / "suite", [bundle])

    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    assert manifest["contract_version"] == CONTROLLED_RECOVERY_REVIEW_CONTRACT
    assert manifest["reference_semantics"] == "frozen_model_track_proxy_not_gold"
    with np.load(bundle / "aggregates.npz", allow_pickle=False) as archive:
        point_count = len(archive["reference_points_xyz"])
        assert len(archive["proxy_points_xyz"]) == point_count
        assert len(archive["input_points_xyz"]) == point_count
        assert len(archive["output_points_xyz"]) == point_count
        assert len(archive["frame_index"]) == point_count
    html = (bundle / "preview.html").read_text(encoding="utf-8")
    assert "REFERENCE" in html
    assert "INPUT" in html
    assert "OUTPUT" in html
    assert "PROXY is the frozen model track" in html
    assert "Equivariant translation RMS" in html
    assert (bundle / "thumbnails" / "comparison_top.png").is_file()
    assert (bundle / "thumbnails" / "comparison_side.png").is_file()
    suite_manifest = json.loads(
        (suite / "recovery-suite.json").read_text(encoding="utf-8")
    )
    assert suite_manifest["contract_version"] == CONTROLLED_RECOVERY_SUITE_CONTRACT
    assert suite_manifest["cases"][0]["case_id"] == case.case_id
    suite_html = (suite / "index.html").read_text(encoding="utf-8")
    assert 'class="case-tab active"' in suite_html
    assert "Known-error Stage 3 diagnostic" in suite_html
    assert "Neither is an input to the" in suite_html


def test_controlled_recovery_cli_builds_selected_profile(tmp_path: Path) -> None:
    root = Path(__file__).parent / "fixtures" / "synthetic_v1" / "inference"
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(_settings().to_dict()),
        encoding="utf-8",
    )
    status = build_controlled_recovery_suite_main(
        [
            "--inference-root",
            str(root),
            "--case-id",
            "static_complete",
            "--profile",
            "mild",
            "--algorithm-variant",
            "normal_aware_pose_graph",
            "--settings",
            str(settings_path),
            "--output",
            str(tmp_path / "recovery"),
            "--max-points-per-frame",
            "100",
        ]
    )

    assert status == 0
    manifest = json.loads(
        (tmp_path / "recovery" / "recovery-suite.json").read_text(encoding="utf-8")
    )
    assert len(manifest["cases"]) == 1
    assert [row["profile"]["name"] for row in manifest["cases"][0]["profiles"]] == [
        "mild"
    ]
    assert manifest["cases"][0]["profiles"][0]["algorithm_variant"] == (
        "normal_aware_pose_graph"
    )
