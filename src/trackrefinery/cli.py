"""Thin command-line adapters around importable TrackRefinery tooling."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from trackrefinery.dataset import InferenceDataset
from trackrefinery.evaluation import (
    AcceptanceThresholds,
    EvaluationReport,
    evaluate_case,
    evaluate_suite,
)
from trackrefinery.refiner import validate_outcome
from trackrefinery.review import build_review_bundle, serve_review_bundle
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
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--crop-scale", type=float, default=1.8)
    build_parser.add_argument("--max-points-per-frame", type=int, default=8_000)
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
        else:
            forwarded.extend((flag, str(value)))
    return forwarded


if __name__ == "__main__":
    raise SystemExit(main())
