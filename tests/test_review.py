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
from trackrefinery.cli import build_clip_review_suite_main, build_review_main
from trackrefinery.dataset import InferenceDataset
from trackrefinery.evaluation import evaluate_case
from trackrefinery.review import (
    build_clip_review_suite,
    build_review_bundle,
    build_review_suite,
)
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
        "gold_aggregate.npz",
        "metrics.json",
        "preview.html",
        "result.json",
    }
    assert expected.issubset(path.name for path in bundle.iterdir())
    with np.load(bundle / "aggregate.npz", allow_pickle=False) as archive:
        assert archive["points_xyz"].shape[1] == 3
        assert len(archive["points_xyz"]) <= len(case.frames) * 1_000
        aggregate_point_count = len(archive["points_xyz"])
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "trackrefinery-review-bundle-v1"
    assert manifest["has_evidence_trace"] is False
    assert manifest["has_gold_aligned_aggregate"] is True
    assert manifest["gold_aggregate_path"] == "gold_aggregate.npz"
    with np.load(bundle / "gold_aggregate.npz", allow_pickle=False) as archive:
        assert archive["points_xyz"].shape[1] == 3
        assert len(archive["points_xyz"]) == aggregate_point_count
    html = (bundle / "preview.html").read_text(encoding="utf-8")
    assert "Algorithm aggregate" in html
    assert "Annotation aggregate" in html
    assert "Annotation-pose-aligned aggregate (review only)" in html
    assert "onclick=\"showView('annotation', this)\"" in html
    assert "Reviewer feedback" in html
    assert (bundle / "thumbnails" / "aggregate_top.png").is_file()
    assert (bundle / "thumbnails" / "gold_aggregate_top.png").is_file()


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
    assert manifest["has_gold_aligned_aggregate"] is False
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
    assert "Annotation aggregate" not in html


def test_catalog_review_bundle_is_lightweight_and_explicitly_baseline(
    tmp_path: Path,
) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    case = InferenceDataset.open(generated.inference_root).load_case("static_complete")
    run = JointCuboidRefiner().refine_with_trace(case)

    bundle = build_review_bundle(
        case,
        run.outcome,
        tmp_path / "catalog",
        data_source="frozen model candidate",
        review_mode="model_candidate_baseline",
        detail_level="catalog",
        crop_scale=1.2,
        max_points_per_frame=100,
    )

    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    assert manifest["detail_level"] == "catalog"
    assert manifest["review_mode"] == "model_candidate_baseline"
    assert (bundle / "thumbnails" / "aggregate_top.png").is_file()
    assert (bundle / "thumbnails" / "aggregate_side.png").is_file()
    assert not (bundle / "thumbnails" / "aggregate_front.png").exists()
    html = (bundle / "preview.html").read_text(encoding="utf-8")
    assert "Model candidate baseline; refinement not run" in html
    assert "plotly" not in html.lower()


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


def test_review_suite_indexes_case_bundles_as_tabs(tmp_path: Path) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    inference = InferenceDataset.open(generated.inference_root)
    targets = TargetDataset.open(generated.target_root)
    suite_root = tmp_path / "suite"
    bundles = []
    for case_id in ("static_complete", "moving_complete"):
        case = inference.load_case(case_id)
        target = targets.load_target(case_id)
        run = JointCuboidRefiner().refine_with_trace(case)
        bundles.append(
            build_review_bundle(
                case,
                run.outcome,
                suite_root / "cases" / case_id,
                target=target,
                trace=run.trace,
                data_source="synthetic-v1 deterministic fixture",
                max_points_per_frame=200,
            )
        )

    output = build_review_suite(
        suite_root,
        bundles,
        title="Synthetic regression suite",
    )

    manifest = json.loads((output / "suite.json").read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "trackrefinery-review-suite-v1"
    assert [row["case_id"] for row in manifest["cases"]] == [
        "static_complete",
        "moving_complete",
    ]
    assert all(row["has_gold_aligned_aggregate"] for row in manifest["cases"])
    html = (output / "index.html").read_text(encoding="utf-8")
    assert "Synthetic regression suite" in html
    assert "static_complete" in html
    assert "moving_complete" in html
    assert html.count('class="case-tab') == 2
    assert html.count("<iframe") == 2


def test_clip_review_suite_groups_instances_as_cards_under_clip_tabs(
    tmp_path: Path,
) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    inference = InferenceDataset.open(generated.inference_root)
    suite_root = tmp_path / "clip-suite"
    grouped: dict[str, list[Path]] = {"clip-alpha": [], "clip-beta": []}
    for clip_id, case_ids in {
        "clip-alpha": ("static_complete", "moving_complete"),
        "clip-beta": ("partial_visibility",),
    }.items():
        for case_id in case_ids:
            case = inference.load_case(case_id)
            run = JointCuboidRefiner().refine_with_trace(case)
            grouped[clip_id].append(
                build_review_bundle(
                    case,
                    run.outcome,
                    suite_root / "clips" / clip_id / "instances" / case_id,
                    trace=run.trace,
                    data_source=clip_id,
                    review_mode=(
                        "source_annotation_reference"
                        if case_id == "partial_visibility"
                        else "algorithm_candidate"
                    ),
                    max_points_per_frame=100,
                )
            )

    output = build_clip_review_suite(
        suite_root,
        grouped,
        title="Real Clip review",
    )

    manifest = json.loads((output / "clip-suite.json").read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "trackrefinery-clip-review-suite-v1"
    assert [row["clip_id"] for row in manifest["clips"]] == [
        "clip-alpha",
        "clip-beta",
    ]
    assert [row["instance_count"] for row in manifest["clips"]] == [2, 1]
    assert manifest["clips"][1]["instances"][0]["review_mode"] == (
        "source_annotation_reference"
    )
    assert manifest["clips"][1]["instances"][0]["aggregate_top_label"] == (
        "Annotation alignment · TOP aggregate"
    )
    assert manifest["clips"][1]["instances"][0]["aggregate_side_label"] == (
        "Annotation alignment · SIDE aggregate"
    )
    assert manifest["clips"][0]["instances"][0]["review_mode"] == (
        "algorithm_candidate"
    )
    assert manifest["clips"][0]["instances"][0]["canonical_top_path"]
    html = (output / "index.html").read_text(encoding="utf-8")
    assert html.count('<button class="clip-tab') == 2
    assert html.count('<article class="instance-card ') == 3
    assert "Every image below is a multi-frame" in html
    assert "ALGORITHM CANDIDATE" in html
    assert "MODEL TRACK BASELINE" in html
    assert "ANNOTATION REFERENCE" in html
    assert 'data-review-mode="source_annotation_reference"' in html
    assert "REFINEMENT NOT RUN" in html

    status = build_clip_review_suite_main(
        [
            "--output",
            str(suite_root),
            "--clip-bundle",
            f"clip-alpha={grouped['clip-alpha'][0]}",
            "--clip-bundle",
            f"clip-alpha={grouped['clip-alpha'][1]}",
            "--clip-bundle",
            f"clip-beta={grouped['clip-beta'][0]}",
        ]
    )
    assert status == 0
