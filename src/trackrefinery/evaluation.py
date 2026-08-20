"""Gold-target metrics for framework and algorithm development."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trackrefinery.contracts import (
    Box3D,
    InsufficientEvidence,
    PartialRefinementSuccess,
    RefinementCase,
    RefinementOutcome,
)
from trackrefinery.dataset import InferenceDataset
from trackrefinery.geometry import (
    angle_difference,
    rotation_geodesic,
    yaw_from_quaternion,
)
from trackrefinery.refiner import validate_outcome
from trackrefinery.serde import read_outcome
from trackrefinery.targets import GoldTarget, TargetDataset


@dataclass(frozen=True, slots=True)
class AcceptanceThresholds:
    max_length_error_m: float
    max_width_error_m: float
    max_height_error_m: float
    max_center_xy_p95_m: float
    max_center_z_p95_m: float
    max_yaw_p95_deg: float
    max_center_xy_worst_m: float
    max_center_z_worst_m: float
    max_yaw_worst_deg: float
    min_frame_bev_iou: float
    min_frame_3d_iou: float

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("acceptance thresholds must be finite")
        for name, value in values.items():
            if name.startswith("min_"):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1]")
            elif value < 0:
                raise ValueError(f"{name} must be non-negative")

    @classmethod
    def from_json(cls, path: str | Path) -> AcceptanceThresholds:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    frame_id: str
    center_xy_error_m: float
    center_z_error_m: float
    yaw_error_deg: float
    rotation_error_deg: float
    bev_iou: float
    iou_3d: float


@dataclass(frozen=True, slots=True)
class MetricSummary:
    canonical_size_lwh: tuple[float, float, float]
    dimension_abs_error_m: tuple[float, float, float]
    temporal_size_std_m: tuple[float, float, float]
    frames: tuple[FrameMetrics, ...]
    center_xy_median_m: float
    center_xy_p95_m: float
    center_xy_worst_m: float
    center_z_median_m: float
    center_z_p95_m: float
    center_z_worst_m: float
    yaw_median_deg: float
    yaw_p95_deg: float
    yaw_worst_deg: float
    bev_iou_median: float
    bev_iou_worst: float
    iou_3d_median: float
    iou_3d_worst: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    case_id: str
    track_id: str
    category: str | None
    expected_refinable: bool
    outcome_status: str
    input_frame_count: int
    authoritative_frame_count: int
    unsupported_frame_count: int
    authoritative_frame_ids: tuple[str, ...]
    unsupported_frame_ids: tuple[str, ...]
    baseline: MetricSummary
    refined: MetricSummary | None
    strict_pass: bool | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def evaluate_case(
    case: RefinementCase,
    outcome: RefinementOutcome,
    target: GoldTarget,
    thresholds: AcceptanceThresholds | None = None,
) -> EvaluationReport:
    """Compare frozen coarse and refined geometry with a separate gold target."""

    validate_outcome(case, outcome)
    _validate_target(case, target)
    all_target_poses = {
        item.frame_id: item.pose for item in target.frame_poses if item.evaluable
    }
    authoritative_ids = (
        tuple(item.frame_id for item in outcome.frame_poses)
        if not isinstance(outcome, InsufficientEvidence)
        else ()
    )
    unsupported_ids = (
        tuple(item.frame_id for item in outcome.unsupported_frames)
        if isinstance(outcome, PartialRefinementSuccess)
        else (
            tuple(item.frame_id for item in case.track.observations)
            if isinstance(outcome, InsufficientEvidence)
            else ()
        )
    )
    target_poses = (
        {
            frame_id: pose
            for frame_id, pose in all_target_poses.items()
            if frame_id in set(authoritative_ids)
        }
        if authoritative_ids
        else all_target_poses
    )
    has_authoritative_evaluable_frames = bool(target_poses)
    if not target_poses:
        target_poses = all_target_poses
    observations = {
        item.frame_id: item
        for item in case.track.observations
        if item.frame_id in target_poses
    }
    coarse_boxes = {item.frame_id: item.coarse_box for item in observations.values()}
    coarse_sizes = np.asarray(
        [item.size_lwh for item in coarse_boxes.values()], dtype=np.float64
    )
    baseline_size = tuple(float(value) for value in np.median(coarse_sizes, axis=0))
    baseline = _summarize(
        canonical_size=baseline_size,
        per_frame_boxes=coarse_boxes,
        target=target,
        target_poses=target_poses,
        temporal_sizes=coarse_sizes,
    )

    refined: MetricSummary | None = None
    strict_pass: bool | None
    reasons: tuple[str, ...] = ()
    if isinstance(outcome, InsufficientEvidence):
        strict_pass = not target.expected_refinable
        reasons = outcome.reasons
    else:
        if not has_authoritative_evaluable_frames:
            return EvaluationReport(
                case_id=case.case_id,
                track_id=case.track.track_id,
                category=case.track.category,
                expected_refinable=target.expected_refinable,
                outcome_status=outcome.status,
                input_frame_count=len(case.frames),
                authoritative_frame_count=len(authoritative_ids),
                unsupported_frame_count=len(unsupported_ids),
                authoritative_frame_ids=authoritative_ids,
                unsupported_frame_ids=unsupported_ids,
                baseline=baseline,
                refined=None,
                strict_pass=False,
                reasons=("no_authoritative_evaluable_frames",),
            )
        poses = {item.frame_id: item.pose for item in outcome.frame_poses}
        refined_boxes = {
            frame_id: Box3D(
                center=poses[frame_id].translation_xyz,
                size_lwh=outcome.canonical_size_lwh,
                orientation_xyzw=poses[frame_id].orientation_xyzw,
            )
            for frame_id in target_poses
        }
        repeated_size = np.repeat(
            np.asarray(outcome.canonical_size_lwh, dtype=np.float64)[None, :],
            len(refined_boxes),
            axis=0,
        )
        refined = _summarize(
            canonical_size=outcome.canonical_size_lwh,
            per_frame_boxes=refined_boxes,
            target=target,
            target_poses=target_poses,
            temporal_sizes=repeated_size,
        )
        if not target.expected_refinable:
            strict_pass = False
            reasons = ("target_marked_not_refinable",)
        elif thresholds is None:
            strict_pass = None
        else:
            strict_pass, reasons = _passes(refined, thresholds)
    return EvaluationReport(
        case_id=case.case_id,
        track_id=case.track.track_id,
        category=case.track.category,
        expected_refinable=target.expected_refinable,
        outcome_status=outcome.status,
        input_frame_count=len(case.frames),
        authoritative_frame_count=len(authoritative_ids),
        unsupported_frame_count=len(unsupported_ids),
        authoritative_frame_ids=authoritative_ids,
        unsupported_frame_ids=unsupported_ids,
        baseline=baseline,
        refined=refined,
        strict_pass=strict_pass,
        reasons=reasons,
    )


@dataclass(frozen=True, slots=True)
class BenchmarkCounts:
    total_cases: int
    successes: int
    full_successes: int
    partial_successes: int
    insufficient_evidence: int
    expected_refinable: int
    strict_pass: int
    strict_fail: int
    strict_unset: int
    unexpected_success: int
    missed_refinable: int
    catastrophic_success: int


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    dataset_id: str
    counts: BenchmarkCounts
    by_category: dict[str, BenchmarkCounts]
    cases: tuple[EvaluationReport, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def evaluate_suite(
    inference: InferenceDataset,
    targets: TargetDataset,
    predictions_root: str | Path,
    thresholds: AcceptanceThresholds | None = None,
) -> BenchmarkReport:
    """Evaluate one result per indexed case and aggregate strict outcome counts."""

    if inference.dataset_id != targets.dataset_id:
        raise ValueError("inference and target dataset IDs do not match")
    prediction_root = Path(predictions_root).resolve()
    reports: list[EvaluationReport] = []
    for entry in inference.cases:
        case = inference.load_case(entry.case_id)
        target = targets.load_target(entry.case_id)
        result_case_id, outcome = read_outcome(
            prediction_root / f"{entry.case_id}.json"
        )
        if result_case_id != entry.case_id:
            raise ValueError(f"prediction case_id mismatch for {entry.case_id!r}")
        reports.append(evaluate_case(case, outcome, target, thresholds))
    categories = sorted({report.category or "<unspecified>" for report in reports})
    return BenchmarkReport(
        dataset_id=inference.dataset_id,
        counts=_benchmark_counts(reports),
        by_category={
            category: _benchmark_counts(
                [
                    report
                    for report in reports
                    if (report.category or "<unspecified>") == category
                ]
            )
            for category in categories
        },
        cases=tuple(reports),
    )


def _benchmark_counts(reports: list[EvaluationReport]) -> BenchmarkCounts:
    success_statuses = {"success", "partial_success"}
    return BenchmarkCounts(
        total_cases=len(reports),
        successes=sum(report.outcome_status in success_statuses for report in reports),
        full_successes=sum(report.outcome_status == "success" for report in reports),
        partial_successes=sum(
            report.outcome_status == "partial_success" for report in reports
        ),
        insufficient_evidence=sum(
            report.outcome_status == "insufficient_evidence" for report in reports
        ),
        expected_refinable=sum(report.expected_refinable for report in reports),
        strict_pass=sum(report.strict_pass is True for report in reports),
        strict_fail=sum(report.strict_pass is False for report in reports),
        strict_unset=sum(report.strict_pass is None for report in reports),
        unexpected_success=sum(
            report.outcome_status in success_statuses and not report.expected_refinable
            for report in reports
        ),
        missed_refinable=sum(
            report.outcome_status == "insufficient_evidence"
            and report.expected_refinable
            for report in reports
        ),
        catastrophic_success=sum(
            report.outcome_status in success_statuses and report.strict_pass is False
            for report in reports
        ),
    )


def _validate_target(case: RefinementCase, target: GoldTarget) -> None:
    if target.case_id != case.case_id:
        raise ValueError("target case_id does not match the input case")
    if target.sequence_id != case.track.sequence_id:
        raise ValueError("target sequence_id does not match the input case")
    if target.track_id != case.track.track_id:
        raise ValueError("target track_id does not match the input track")
    input_frames = {item.frame_id for item in case.track.observations}
    target_frames = {item.frame_id for item in target.frame_poses}
    if not target_frames.issubset(input_frames):
        raise ValueError("target references frames outside the input track")


def _summarize(
    *,
    canonical_size: tuple[float, float, float],
    per_frame_boxes: dict[str, Box3D],
    target: GoldTarget,
    target_poses: dict[str, object],
    temporal_sizes: NDArray[np.float64],
) -> MetricSummary:
    target_size = np.asarray(target.canonical_size_lwh, dtype=np.float64)
    size = np.asarray(canonical_size, dtype=np.float64)
    frames: list[FrameMetrics] = []
    for frame_id, pose_value in target_poses.items():
        pose = pose_value
        gold_box = Box3D(
            center=pose.translation_xyz,
            size_lwh=target.canonical_size_lwh,
            orientation_xyzw=pose.orientation_xyzw,
        )
        box = per_frame_boxes[frame_id]
        delta = np.asarray(box.center) - np.asarray(gold_box.center)
        frames.append(
            FrameMetrics(
                frame_id=frame_id,
                center_xy_error_m=float(np.linalg.norm(delta[:2])),
                center_z_error_m=float(abs(delta[2])),
                yaw_error_deg=float(
                    np.degrees(
                        abs(
                            angle_difference(
                                yaw_from_quaternion(box.orientation_xyzw),
                                yaw_from_quaternion(gold_box.orientation_xyzw),
                            )
                        )
                    )
                ),
                rotation_error_deg=float(
                    np.degrees(
                        rotation_geodesic(
                            box.orientation_xyzw, gold_box.orientation_xyzw
                        )
                    )
                ),
                bev_iou=bev_iou(box, gold_box),
                iou_3d=iou_3d(box, gold_box),
            )
        )
    xy = np.asarray([item.center_xy_error_m for item in frames])
    z = np.asarray([item.center_z_error_m for item in frames])
    yaw = np.asarray([item.yaw_error_deg for item in frames])
    bev = np.asarray([item.bev_iou for item in frames])
    volume = np.asarray([item.iou_3d for item in frames])
    return MetricSummary(
        canonical_size_lwh=tuple(float(value) for value in size),
        dimension_abs_error_m=tuple(
            float(value) for value in np.abs(size - target_size)
        ),
        temporal_size_std_m=tuple(
            float(value) for value in np.std(temporal_sizes, axis=0)
        ),
        frames=tuple(frames),
        center_xy_median_m=float(np.median(xy)),
        center_xy_p95_m=float(np.quantile(xy, 0.95)),
        center_xy_worst_m=float(np.max(xy)),
        center_z_median_m=float(np.median(z)),
        center_z_p95_m=float(np.quantile(z, 0.95)),
        center_z_worst_m=float(np.max(z)),
        yaw_median_deg=float(np.median(yaw)),
        yaw_p95_deg=float(np.quantile(yaw, 0.95)),
        yaw_worst_deg=float(np.max(yaw)),
        bev_iou_median=float(np.median(bev)),
        bev_iou_worst=float(np.min(bev)),
        iou_3d_median=float(np.median(volume)),
        iou_3d_worst=float(np.min(volume)),
    )


def _passes(
    summary: MetricSummary, thresholds: AcceptanceThresholds
) -> tuple[bool, tuple[str, ...]]:
    length, width, height = summary.dimension_abs_error_m
    checks = {
        "length_error": length <= thresholds.max_length_error_m,
        "width_error": width <= thresholds.max_width_error_m,
        "height_error": height <= thresholds.max_height_error_m,
        "center_xy_p95": summary.center_xy_p95_m <= thresholds.max_center_xy_p95_m,
        "center_z_p95": summary.center_z_p95_m <= thresholds.max_center_z_p95_m,
        "yaw_p95": summary.yaw_p95_deg <= thresholds.max_yaw_p95_deg,
        "center_xy_worst": summary.center_xy_worst_m
        <= thresholds.max_center_xy_worst_m,
        "center_z_worst": summary.center_z_worst_m <= thresholds.max_center_z_worst_m,
        "yaw_worst": summary.yaw_worst_deg <= thresholds.max_yaw_worst_deg,
        "bev_iou_worst": summary.bev_iou_worst >= thresholds.min_frame_bev_iou,
        "iou_3d_worst": summary.iou_3d_worst >= thresholds.min_frame_3d_iou,
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return not reasons, reasons


def bev_iou(first: Box3D, second: Box3D) -> float:
    first_polygon = _bev_polygon(first)
    second_polygon = _bev_polygon(second)
    intersection = _polygon_area(_clip_polygon(first_polygon, second_polygon))
    first_area = first.size_lwh[0] * first.size_lwh[1]
    second_area = second.size_lwh[0] * second.size_lwh[1]
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def iou_3d(first: Box3D, second: Box3D) -> float:
    first_polygon = _bev_polygon(first)
    second_polygon = _bev_polygon(second)
    intersection_area = _polygon_area(_clip_polygon(first_polygon, second_polygon))
    first_min = first.center[2] - first.size_lwh[2] / 2
    first_max = first.center[2] + first.size_lwh[2] / 2
    second_min = second.center[2] - second.size_lwh[2] / 2
    second_max = second.center[2] + second.size_lwh[2] / 2
    overlap_height = max(0.0, min(first_max, second_max) - max(first_min, second_min))
    intersection = intersection_area * overlap_height
    first_volume = float(np.prod(first.size_lwh))
    second_volume = float(np.prod(second.size_lwh))
    union = first_volume + second_volume - intersection
    return float(intersection / union) if union > 0 else 0.0


def _bev_polygon(box: Box3D) -> NDArray[np.float64]:
    half_length, half_width = np.asarray(box.size_lwh[:2]) / 2
    local = np.asarray(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    yaw = yaw_from_quaternion(box.orientation_xyzw)
    rotation = np.asarray([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rotation.T + np.asarray(box.center[:2])


def _clip_polygon(
    subject: NDArray[np.float64], clip: NDArray[np.float64]
) -> NDArray[np.float64]:
    output = [point for point in subject]
    for index, edge_start in enumerate(clip):
        edge_end = clip[(index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _inside(current, edge_start, edge_end)
            previous_inside = _inside(previous, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, edge_start, edge_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, edge_start, edge_end)
                )
            previous = current
    return np.asarray(output, dtype=np.float64)


def _inside(
    point: NDArray[np.float64],
    edge_start: NDArray[np.float64],
    edge_end: NDArray[np.float64],
) -> bool:
    edge = edge_end - edge_start
    relative = point - edge_start
    return bool(edge[0] * relative[1] - edge[1] * relative[0] >= -1e-10)


def _line_intersection(
    first_start: NDArray[np.float64],
    first_end: NDArray[np.float64],
    second_start: NDArray[np.float64],
    second_end: NDArray[np.float64],
) -> NDArray[np.float64]:
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    denominator = _cross_2d(first_direction, second_direction)
    if abs(denominator) < 1e-12:
        return first_end
    offset = second_start - first_start
    t = _cross_2d(offset, second_direction) / denominator
    return first_start + t * first_direction


def _cross_2d(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _polygon_area(polygon: NDArray[np.float64]) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)
