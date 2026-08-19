from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trackrefinery import (
    JointCuboidRefiner,
    RefinedFramePose,
    RefinementSuccess,
    write_geometric_trace,
    write_outcome,
)
from trackrefinery.cli import build_review_main
from trackrefinery.dataset import InferenceDataset
from trackrefinery.evaluation import evaluate_case
from trackrefinery.review import build_review_bundle
from trackrefinery.synthetic import generate_dataset
from trackrefinery.targets import TargetDataset


def test_review_bundle_contains_reproducible_and_interactive_views(
    tmp_path: Path,
) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    case = InferenceDataset.open(generated.inference_root).load_case("static_complete")
    target = TargetDataset.open(generated.target_root).load_target(case.case_id)
    outcome = RefinementSuccess(
        track_id=target.track_id,
        canonical_size_lwh=target.canonical_size_lwh,
        frame_poses=tuple(
            RefinedFramePose(item.frame_id, item.pose) for item in target.frame_poses
        ),
    )
    report = evaluate_case(case, outcome, target)

    bundle = build_review_bundle(
        case,
        outcome,
        tmp_path / "review",
        target=target,
        evaluation=report,
        max_points_per_frame=1_000,
    )

    expected = {
        "aggregate.npz",
        "bundle.json",
        "metrics.json",
        "preview.html",
        "result.json",
    }
    assert expected.issubset(path.name for path in bundle.iterdir())
    with np.load(bundle / "aggregate.npz", allow_pickle=False) as archive:
        assert archive["points_xyz"].shape[1] == 3
        assert len(archive["points_xyz"]) <= len(case.frames) * 1_000
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "trackrefinery-review-bundle-v1"
    assert manifest["has_evidence_trace"] is False
    html = (bundle / "preview.html").read_text(encoding="utf-8")
    assert "Object-frame aggregate" in html
    assert "Reviewer feedback" in html
    assert (bundle / "thumbnails" / "aggregate_top.png").is_file()


def test_review_bundle_renders_algorithm_evidence_trace(tmp_path: Path) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    case = InferenceDataset.open(generated.inference_root).load_case("nearby_clutter")
    run = JointCuboidRefiner().refine_with_trace(case)

    bundle = build_review_bundle(
        case,
        run.outcome,
        tmp_path / "review-trace",
        trace=run.trace,
        data_source="synthetic-v1 deterministic fixture",
        max_points_per_frame=300,
    )

    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    assert manifest["has_evidence_trace"] is True
    assert manifest["has_registration_trace"] is True
    assert manifest["has_cuboid_candidate"] is True
    assert manifest["data_source"] == "synthetic-v1 deterministic fixture"
    assert (bundle / "evidence_trace.json").is_file()
    assert (bundle / "evidence_masks.npz").is_file()
    assert (bundle / "canonical_shape.npz").is_file()
    assert (bundle / "thumbnails" / "aggregate_evidence_top.png").is_file()
    assert (bundle / "thumbnails" / "canonical_registration_top.png").is_file()
    with np.load(bundle / "aggregate.npz", allow_pickle=False) as archive:
        assert archive["evidence_state"].shape == (len(archive["points_xyz"]),)
        assert len(archive["points_xyz"]) <= len(case.frames) * 300
    html = (bundle / "preview.html").read_text(encoding="utf-8")
    assert "Current evidence classification" in html
    assert "Evidence trace" in html
    assert "registration candidate" in html
    assert "Canonical shape after alternating registration" in html
    assert "synthetic-v1 deterministic fixture" in html
    assert "Trace-only cuboid candidate" in html
    assert "cuboid candidate" in html


def test_review_cli_accepts_portable_evidence_trace(tmp_path: Path) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    case = InferenceDataset.open(generated.inference_root).load_case("static_complete")
    run = JointCuboidRefiner().refine_with_trace(case)
    result_path = tmp_path / "result.json"
    trace_dir = tmp_path / "trace"
    write_outcome(result_path, case.case_id, run.outcome)
    write_geometric_trace(trace_dir, run.trace)

    status = build_review_main(
        [
            "--inference-root",
            str(generated.inference_root),
            "--case-id",
            case.case_id,
            "--result",
            str(result_path),
            "--trace",
            str(trace_dir / "evidence_trace.json"),
            "--data-source",
            "synthetic-cli-fixture",
            "--output",
            str(tmp_path / "review-cli"),
        ]
    )

    assert status == 0
    assert (tmp_path / "review-cli" / "evidence_masks.npz").is_file()
    manifest = json.loads(
        (tmp_path / "review-cli" / "bundle.json").read_text(encoding="utf-8")
    )
    assert manifest["data_source"] == "synthetic-cli-fixture"
