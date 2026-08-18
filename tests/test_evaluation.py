from __future__ import annotations

from pathlib import Path

import numpy as np

from trackrefinery import InsufficientEvidence, RefinedFramePose, RefinementSuccess
from trackrefinery.dataset import InferenceDataset
from trackrefinery.evaluation import AcceptanceThresholds, evaluate_case, evaluate_suite
from trackrefinery.serde import read_outcome, write_outcome
from trackrefinery.synthetic import generate_dataset
from trackrefinery.targets import GoldTarget, TargetDataset


def _success(target: GoldTarget) -> RefinementSuccess:
    return RefinementSuccess(
        track_id=target.track_id,
        canonical_size_lwh=target.canonical_size_lwh,
        frame_poses=tuple(
            RefinedFramePose(item.frame_id, item.pose) for item in target.frame_poses
        ),
        diagnostics={"fixture": "gold-copy-for-framework-test"},
    )


def _thresholds() -> AcceptanceThresholds:
    return AcceptanceThresholds(
        max_length_error_m=0.1,
        max_width_error_m=0.05,
        max_height_error_m=0.05,
        max_center_xy_p95_m=0.1,
        max_center_z_p95_m=0.05,
        max_yaw_p95_deg=2.0,
        max_center_xy_worst_m=0.2,
        max_center_z_worst_m=0.1,
        max_yaw_worst_deg=5.0,
        min_frame_bev_iou=0.85,
        min_frame_3d_iou=0.8,
    )


def test_perfect_fixture_output_passes_and_baseline_is_worse(tmp_path: Path) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    case = InferenceDataset.open(generated.inference_root).load_case("moving_complete")
    target = TargetDataset.open(generated.target_root).load_target(case.case_id)

    report = evaluate_case(case, _success(target), target, _thresholds())

    assert report.strict_pass is True
    assert report.refined is not None
    assert report.refined.dimension_abs_error_m == (0.0, 0.0, 0.0)
    assert report.refined.temporal_size_std_m == (0.0, 0.0, 0.0)
    assert report.refined.iou_3d_worst > 1 - 1e-12
    assert max(report.baseline.dimension_abs_error_m) > 0.1


def test_expected_unrefinable_case_rewards_explicit_insufficient_result(
    tmp_path: Path,
) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    case = InferenceDataset.open(generated.inference_root).load_case(
        "partial_visibility"
    )
    target = TargetDataset.open(generated.target_root).load_target(case.case_id)
    outcome = InsufficientEvidence(case.track.track_id, ("one_sided_visibility",))

    report = evaluate_case(case, outcome, target, _thresholds())

    assert report.strict_pass is True
    assert report.refined is None


def test_result_json_round_trip(tmp_path: Path) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    target = TargetDataset.open(generated.target_root).load_target("static_complete")
    path = tmp_path / "result.json"

    write_outcome(path, target.case_id, _success(target))
    case_id, restored = read_outcome(path)

    assert case_id == target.case_id
    assert isinstance(restored, RefinementSuccess)
    assert np.allclose(restored.canonical_size_lwh, target.canonical_size_lwh)


def test_suite_report_counts_success_and_explicit_insufficiency(tmp_path: Path) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    inference = InferenceDataset.open(generated.inference_root)
    targets = TargetDataset.open(generated.target_root)
    predictions = tmp_path / "predictions"
    for entry in inference.cases:
        target = targets.load_target(entry.case_id)
        outcome = (
            _success(target)
            if target.expected_refinable
            else InsufficientEvidence(target.track_id, ("fixture_insufficient",))
        )
        write_outcome(predictions / f"{entry.case_id}.json", entry.case_id, outcome)

    report = evaluate_suite(inference, targets, predictions, _thresholds())

    assert report.counts.total_cases == 7
    assert report.counts.strict_pass == 7
    assert report.counts.catastrophic_success == 0
    assert report.counts.unexpected_success == 0
    assert set(report.by_category) == {"car", "truck"}
