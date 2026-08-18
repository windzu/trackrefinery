from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trackrefinery import RefinedFramePose, RefinementSuccess
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
    html = (bundle / "preview.html").read_text(encoding="utf-8")
    assert "Object-frame aggregate" in html
    assert "Reviewer feedback" in html
    assert (bundle / "thumbnails" / "aggregate_top.png").is_file()
