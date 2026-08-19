"""Thin command-line adapters around importable TrackRefinery tooling."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from trackrefinery.adapters import export_x4d_clip_inference
from trackrefinery.component_consensus import (
    COMPONENT_CONSENSUS_CONFIG_SCHEMA_VERSION,
    POSE_GRAPH_VARIANTS,
    ComponentConsensusRefiner,
    ComponentConsensusSettings,
    aggregate_geometry_components_pose_graph,
)
from trackrefinery.controlled_recovery import (
    DEFAULT_CONTROLLED_PERTURBATION_PROFILES,
    run_controlled_recovery,
)
from trackrefinery.controlled_recovery_review import (
    build_controlled_recovery_bundle,
    build_controlled_recovery_suite,
)
from trackrefinery.dataset import InferenceDataset
from trackrefinery.evaluation import (
    AcceptanceThresholds,
    EvaluationReport,
    evaluate_case,
    evaluate_suite,
)
from trackrefinery.geometric import read_geometric_trace
from trackrefinery.refiner import validate_outcome
from trackrefinery.review import (
    REVIEW_DETAIL_LEVELS,
    REVIEW_MODES,
    build_clip_review_suite,
    build_review_bundle,
    build_review_suite,
    serve_review_bundle,
)
from trackrefinery.serde import read_outcome
from trackrefinery.targets import TargetDataset


def validate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate source-only TrackRefinery inference data."
    )
    parser.add_argument("inference_root", help="directory containing dataset.json")
    args = parser.parse_args(argv)
    dataset = InferenceDataset.open(args.inference_root)
    cases = dataset.validate()
    unique_frames = {
        (case.track.sequence_id, frame.frame_id): frame
        for case in cases
        for frame in case.frames
    }
    print(
        f"validated {dataset.dataset_id}: {len(dataset.sequences)} sequences, "
        f"{len(unique_frames)} shared frames, {len(cases)} single-instance cases, "
        f"{sum(len(frame.points_xyz) for frame in unique_frames.values())} points"
    )
    return 0


def validate_targets_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate separate gold targets.")
    parser.add_argument("target_root", help="directory containing targetset.json")
    args = parser.parse_args(argv)
    dataset = TargetDataset.open(args.target_root)
    targets = tuple(dataset.load_target(entry.case_id) for entry in dataset.entries)
    print(f"validated {dataset.dataset_id}: {len(targets)} gold targets")
    return 0


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate one refinement result.")
    _add_evaluation_arguments(parser)
    parser.add_argument("--out", required=True, help="output metrics JSON")
    args = parser.parse_args(argv)
    report = _load_evaluation(args)
    report.write_json(args.out)
    print(f"wrote evaluation report to {Path(args.out).resolve()}")
    return 0


def evaluate_suite_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a complete prediction set.")
    parser.add_argument("--inference-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--thresholds")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    inference = InferenceDataset.open(args.inference_root)
    targets = TargetDataset.open(args.target_root)
    thresholds = (
        AcceptanceThresholds.from_json(args.thresholds) if args.thresholds else None
    )
    report = evaluate_suite(
        inference,
        targets,
        args.predictions_root,
        thresholds,
    )
    report.write_json(args.out)
    print(f"wrote benchmark report to {Path(args.out).resolve()}")
    return 0


def build_review_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build one visual review bundle.")
    parser.add_argument("--inference-root", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--target-root")
    parser.add_argument("--thresholds")
    parser.add_argument("--trace")
    parser.add_argument("--data-source")
    parser.add_argument("--review-mode", choices=sorted(REVIEW_MODES))
    parser.add_argument("--detail-level", choices=sorted(REVIEW_DETAIL_LEVELS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-scale", type=float, default=1.8)
    parser.add_argument("--max-points-per-frame", type=int, default=8_000)
    args = parser.parse_args(argv)
    inference = InferenceDataset.open(args.inference_root)
    case = inference.load_case(args.case_id)
    result_case_id, outcome = read_outcome(args.result)
    if result_case_id != case.case_id:
        raise ValueError("result case_id does not match --case-id")
    validate_outcome(case, outcome)
    target = None
    report = None
    if args.target_root:
        targets = TargetDataset.open(args.target_root)
        if targets.dataset_id != inference.dataset_id:
            raise ValueError("inference and target dataset IDs do not match")
        target = targets.load_target(case.case_id)
        thresholds = (
            AcceptanceThresholds.from_json(args.thresholds) if args.thresholds else None
        )
        report = evaluate_case(case, outcome, target, thresholds)
    output = build_review_bundle(
        case,
        outcome,
        args.output,
        target=target,
        evaluation=report,
        trace=read_geometric_trace(args.trace) if args.trace else None,
        data_source=args.data_source or inference.dataset_id,
        review_mode=args.review_mode or "algorithm_candidate",
        detail_level=args.detail_level or "full",
        crop_scale=args.crop_scale,
        max_points_per_frame=args.max_points_per_frame,
    )
    print(f"wrote review bundle to {output}")
    return 0


def review_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a TrackRefinery review bundle.")
    parser.add_argument("bundle")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args(argv)
    serve_review_bundle(
        args.bundle,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
    )
    return 0


def build_review_suite_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a tabbed review suite index.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--title", default="TrackRefinery review suite")
    args = parser.parse_args(argv)
    output = build_review_suite(args.output, args.bundle, title=args.title)
    print(f"wrote review suite to {output}")
    return 0


def build_clip_review_suite_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a Clip-tabbed, all-instance review catalog."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--clip-bundle",
        action="append",
        required=True,
        metavar="CLIP_ID=BUNDLE_DIR",
    )
    parser.add_argument("--title", default="TrackRefinery real Clip review")
    args = parser.parse_args(argv)
    clips: dict[str, list[str]] = {}
    for value in args.clip_bundle:
        clip_id, separator, bundle = value.partition("=")
        if not separator or not clip_id or not bundle:
            raise ValueError("--clip-bundle must be CLIP_ID=BUNDLE_DIR")
        clips.setdefault(clip_id, []).append(bundle)
    output = build_clip_review_suite(args.output, clips, title=args.title)
    print(f"wrote Clip review suite to {output}")
    return 0


def build_controlled_recovery_suite_main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Build controlled Stage 3 recovery diagnostics."
    )
    parser.add_argument("--inference-root", required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument(
        "--profile", action="append", choices=("mild", "medium", "strong")
    )
    parser.add_argument(
        "--algorithm-variant",
        choices=("sequential_v2_1", *sorted(POSE_GRAPH_VARIANTS)),
        default="sequential_v2_1",
    )
    parser.add_argument("--data-source")
    parser.add_argument("--settings", help="component-consensus settings JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-points-per-frame", type=int, default=8_000)
    parser.add_argument("--title", default="TrackRefinery controlled pose recovery")
    args = parser.parse_args(argv)
    dataset = InferenceDataset.open(args.inference_root)
    settings = _load_component_consensus_settings(args.settings)
    profiles = {
        profile.name: profile for profile in DEFAULT_CONTROLLED_PERTURBATION_PROFILES
    }
    selected_profiles = (
        [profiles[name] for name in args.profile]
        if args.profile
        else list(DEFAULT_CONTROLLED_PERTURBATION_PROFILES)
    )
    output = Path(args.output).resolve()
    bundles: list[Path] = []
    aggregation_backend = None
    if args.algorithm_variant != "sequential_v2_1":

        def aggregation_backend(case, trace, settings):
            return aggregate_geometry_components_pose_graph(
                case,
                trace,
                settings,
                variant=args.algorithm_variant,
            ).trace

    for case_id in args.case_id:
        case = dataset.load_case(case_id)
        baseline = ComponentConsensusRefiner(settings).refine_with_trace(case)
        for profile in selected_profiles:
            run = run_controlled_recovery(
                case,
                profile=profile,
                settings=settings,
                component_trace=baseline.trace,
                aggregation_backend=aggregation_backend,
                algorithm_variant=args.algorithm_variant,
            )
            bundles.append(
                build_controlled_recovery_bundle(
                    run,
                    output
                    / "cases"
                    / case.case_id
                    / args.algorithm_variant
                    / profile.name,
                    data_source=args.data_source or dataset.dataset_id,
                    max_points_per_frame=args.max_points_per_frame,
                )
            )
    build_controlled_recovery_suite(output, bundles, title=args.title)
    print(
        f"wrote {len(bundles)} controlled recovery profiles for "
        f"{len(args.case_id)} cases to {output}"
    )
    return 0


def _load_component_consensus_settings(
    path: str | None,
) -> ComponentConsensusSettings:
    if path is None:
        return ComponentConsensusSettings()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("component-consensus settings must contain an object")
    schema = value.pop("schema_version", None)
    if schema != COMPONENT_CONSENSUS_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "component-consensus settings schema_version does not match this package"
        )
    return ComponentConsensusSettings(**value)


def export_x4d_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export one X-4D Dataset 0.17 Clip into source-only inputs."
    )
    parser.add_argument("--clip-dir", required=True)
    parser.add_argument("--annotation-document", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--role",
        required=True,
        choices=("development", "calibration", "test", "qualitative"),
    )
    parser.add_argument(
        "--source-kind",
        required=True,
        choices=("model_candidate", "source_annotation_reference"),
    )
    args = parser.parse_args(argv)
    exported = export_x4d_clip_inference(
        args.clip_dir,
        args.annotation_document,
        args.output,
        role=args.role,
        source_kind=args.source_kind,
    )
    print(
        f"exported {exported.clip_id}: {len(exported.case_ids)} cases, "
        f"{len(exported.lidar_channels)} LiDAR channels, "
        f"{exported.dropped_nonfinite_points} non-finite points dropped"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trackrefinery")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("inference_root")
    targets_parser = subparsers.add_parser("validate-targets")
    targets_parser.add_argument("target_root")
    evaluate_parser = subparsers.add_parser("evaluate")
    _add_evaluation_arguments(evaluate_parser)
    evaluate_parser.add_argument("--out", required=True)
    suite_parser = subparsers.add_parser("evaluate-suite")
    suite_parser.add_argument("--inference-root", required=True)
    suite_parser.add_argument("--target-root", required=True)
    suite_parser.add_argument("--predictions-root", required=True)
    suite_parser.add_argument("--thresholds")
    suite_parser.add_argument("--out", required=True)
    build_parser = subparsers.add_parser("build-review")
    build_parser.add_argument("--inference-root", required=True)
    build_parser.add_argument("--case-id", required=True)
    build_parser.add_argument("--result", required=True)
    build_parser.add_argument("--target-root")
    build_parser.add_argument("--thresholds")
    build_parser.add_argument("--trace")
    build_parser.add_argument("--data-source")
    build_parser.add_argument("--review-mode", choices=sorted(REVIEW_MODES))
    build_parser.add_argument("--detail-level", choices=sorted(REVIEW_DETAIL_LEVELS))
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--crop-scale", type=float, default=1.8)
    build_parser.add_argument("--max-points-per-frame", type=int, default=8_000)
    suite_parser = subparsers.add_parser("build-review-suite")
    suite_parser.add_argument("--output", required=True)
    suite_parser.add_argument("--bundle", action="append", required=True)
    suite_parser.add_argument("--title", default="TrackRefinery review suite")
    clip_suite_parser = subparsers.add_parser("build-clip-review-suite")
    clip_suite_parser.add_argument("--output", required=True)
    clip_suite_parser.add_argument("--clip-bundle", action="append", required=True)
    clip_suite_parser.add_argument("--title", default="TrackRefinery real Clip review")
    recovery_parser = subparsers.add_parser("build-controlled-recovery-suite")
    recovery_parser.add_argument("--inference-root", required=True)
    recovery_parser.add_argument("--case-id", action="append", required=True)
    recovery_parser.add_argument(
        "--profile", action="append", choices=("mild", "medium", "strong")
    )
    recovery_parser.add_argument(
        "--algorithm-variant",
        choices=("sequential_v2_1", *sorted(POSE_GRAPH_VARIANTS)),
        default="sequential_v2_1",
    )
    recovery_parser.add_argument("--data-source")
    recovery_parser.add_argument("--settings")
    recovery_parser.add_argument("--output", required=True)
    recovery_parser.add_argument("--max-points-per-frame", type=int, default=8_000)
    recovery_parser.add_argument(
        "--title", default="TrackRefinery controlled pose recovery"
    )
    x4d_parser = subparsers.add_parser("export-x4d")
    x4d_parser.add_argument("--clip-dir", required=True)
    x4d_parser.add_argument("--annotation-document", required=True)
    x4d_parser.add_argument("--output", required=True)
    x4d_parser.add_argument(
        "--role",
        required=True,
        choices=("development", "calibration", "test", "qualitative"),
    )
    x4d_parser.add_argument(
        "--source-kind",
        required=True,
        choices=("model_candidate", "source_annotation_reference"),
    )
    serve_parser = subparsers.add_parser("review")
    serve_parser.add_argument("bundle")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args(argv)
    forwarded = _forwarded_arguments(args)
    return {
        "validate": validate_main,
        "validate-targets": validate_targets_main,
        "evaluate": evaluate_main,
        "evaluate-suite": evaluate_suite_main,
        "build-review": build_review_main,
        "build-review-suite": build_review_suite_main,
        "build-clip-review-suite": build_clip_review_suite_main,
        "build-controlled-recovery-suite": build_controlled_recovery_suite_main,
        "export-x4d": export_x4d_main,
        "review": review_main,
    }[args.command](forwarded)


def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inference-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--thresholds")


def _load_evaluation(args: argparse.Namespace) -> EvaluationReport:
    inference = InferenceDataset.open(args.inference_root)
    targets = TargetDataset.open(args.target_root)
    if targets.dataset_id != inference.dataset_id:
        raise ValueError("inference and target dataset IDs do not match")
    case = inference.load_case(args.case_id)
    target = targets.load_target(args.case_id)
    result_case_id, outcome = read_outcome(args.result)
    if result_case_id != case.case_id:
        raise ValueError("result case_id does not match --case-id")
    thresholds = (
        AcceptanceThresholds.from_json(args.thresholds) if args.thresholds else None
    )
    return evaluate_case(case, outcome, target, thresholds)


def _forwarded_arguments(args: argparse.Namespace) -> list[str]:
    values = vars(args).copy()
    command = values.pop("command")
    forwarded: list[str] = []
    positional = {
        "validate": ("inference_root",),
        "validate-targets": ("target_root",),
        "review": ("bundle",),
    }.get(command, ())
    for name in positional:
        forwarded.append(str(values.pop(name)))
    for name, value in values.items():
        if value is None or value is False:
            continue
        flag = "--open" if name == "open_browser" else "--" + name.replace("_", "-")
        if value is True:
            forwarded.append(flag)
        elif isinstance(value, list):
            for item in value:
                forwarded.extend((flag, str(item)))
        else:
            forwarded.extend((flag, str(value)))
    return forwarded


if __name__ == "__main__":
    raise SystemExit(main())
