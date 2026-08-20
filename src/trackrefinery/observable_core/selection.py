"""Deterministic connected-core selection over component evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from trackrefinery.contracts import RefinementCase
from trackrefinery.geometric.trace import (
    FrameEvidenceTrace,
    FrameRole,
    GeometricRefinementTrace,
    validate_geometric_trace,
)
from trackrefinery.observable_core.settings import ObservableCoreSettings


class ObservableFrameDisposition(str, Enum):
    """Provisional use of one frame before canonical sizing."""

    CORE_GEOMETRY = "core_geometry"
    POSE_CANDIDATE = "pose_candidate"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ObservableFrameQualification:
    frame_id: str
    provisional_role: FrameRole
    disposition: ObservableFrameDisposition
    reason_codes: tuple[str, ...]
    selected_point_count: int
    selected_voxel_count: int
    component_dominance: float | None
    resolution_stability_iou: float | None

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("qualification frame_id must be non-empty")
        object.__setattr__(self, "provisional_role", FrameRole(self.provisional_role))
        object.__setattr__(
            self, "disposition", ObservableFrameDisposition(self.disposition)
        )
        reasons = tuple(self.reason_codes)
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("qualification reason codes must be non-empty strings")
        if self.disposition is ObservableFrameDisposition.CORE_GEOMETRY and reasons:
            raise ValueError("core geometry qualification cannot contain reasons")
        if self.selected_point_count < 0 or self.selected_voxel_count < 0:
            raise ValueError("qualification counts must be non-negative")
        for name, value in (
            ("component_dominance", self.component_dominance),
            ("resolution_stability_iou", self.resolution_stability_iou),
        ):
            if value is not None and (
                not np.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "provisional_role": self.provisional_role.value,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "selected_point_count": self.selected_point_count,
            "selected_voxel_count": self.selected_voxel_count,
            "component_dominance": self.component_dominance,
            "resolution_stability_iou": self.resolution_stability_iou,
        }


@dataclass(frozen=True, slots=True)
class ObservableCoreRunTrace:
    frame_ids: tuple[str, ...]
    start_index: int
    end_index: int
    total_selected_points: int
    total_selected_voxels: int
    minimum_component_dominance: float
    minimum_resolution_stability_iou: float
    selected: bool = False

    def __post_init__(self) -> None:
        frame_ids = tuple(self.frame_ids)
        if not frame_ids or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("observable run requires unique non-empty frame IDs")
        if self.start_index < 0 or self.end_index < self.start_index:
            raise ValueError("observable run indices are invalid")
        if self.end_index - self.start_index + 1 != len(frame_ids):
            raise ValueError("observable run indices must be contiguous")
        if self.total_selected_points < 1 or self.total_selected_voxels < 1:
            raise ValueError("observable run requires positive evidence counts")
        for name, value in (
            ("minimum_component_dominance", self.minimum_component_dominance),
            (
                "minimum_resolution_stability_iou",
                self.minimum_resolution_stability_iou,
            ),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        object.__setattr__(self, "frame_ids", frame_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_ids": list(self.frame_ids),
            "start_index": self.start_index,
            "end_index": self.end_index,
            "frame_count": len(self.frame_ids),
            "total_selected_points": self.total_selected_points,
            "total_selected_voxels": self.total_selected_voxels,
            "minimum_component_dominance": self.minimum_component_dominance,
            "minimum_resolution_stability_iou": (self.minimum_resolution_stability_iou),
            "selected": self.selected,
        }


@dataclass(frozen=True, slots=True)
class ObservableCoreSelection:
    status: str
    reason_codes: tuple[str, ...]
    reference_interval_ns: int
    maximum_connected_gap_ns: int
    minimum_core_frames: int
    core_frame_ids: tuple[str, ...]
    rejected_geometry_frame_ids: tuple[str, ...]
    candidate_runs: tuple[ObservableCoreRunTrace, ...]
    frames: tuple[ObservableFrameQualification, ...]

    def __post_init__(self) -> None:
        if self.status not in {"candidate", "insufficient_evidence"}:
            raise ValueError("observable core status is unsupported")
        reasons = tuple(self.reason_codes)
        core_ids = tuple(self.core_frame_ids)
        rejected_ids = tuple(self.rejected_geometry_frame_ids)
        runs = tuple(self.candidate_runs)
        frames = tuple(self.frames)
        if self.reference_interval_ns < 1 or self.maximum_connected_gap_ns < 1:
            raise ValueError("observable core time intervals must be positive")
        if self.maximum_connected_gap_ns < self.reference_interval_ns:
            raise ValueError("maximum connected gap cannot be below reference interval")
        if self.minimum_core_frames < 1:
            raise ValueError("minimum_core_frames must be positive")
        frame_ids = [frame.frame_id for frame in frames]
        if not frames or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("observable core frames must be non-empty and unique")
        if set(core_ids) & set(rejected_ids):
            raise ValueError("core and rejected geometry frame IDs must be disjoint")
        if (set(core_ids) | set(rejected_ids)) - set(frame_ids):
            raise ValueError("observable core references an unknown frame")
        selected_runs = [run for run in runs if run.selected]
        if self.status == "candidate":
            if reasons or len(selected_runs) != 1:
                raise ValueError("observable core candidate requires one selected run")
            if core_ids != selected_runs[0].frame_ids:
                raise ValueError("selected run and core frame IDs must match")
            if len(core_ids) < self.minimum_core_frames:
                raise ValueError("observable core candidate is below minimum length")
        elif not reasons or selected_runs or core_ids:
            raise ValueError(
                "insufficient observable core requires reasons and no core"
            )
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "core_frame_ids", core_ids)
        object.__setattr__(self, "rejected_geometry_frame_ids", rejected_ids)
        object.__setattr__(self, "candidate_runs", runs)
        object.__setattr__(self, "frames", frames)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "reference_interval_ns": self.reference_interval_ns,
            "maximum_connected_gap_ns": self.maximum_connected_gap_ns,
            "minimum_core_frames": self.minimum_core_frames,
            "core_frame_ids": list(self.core_frame_ids),
            "rejected_geometry_frame_ids": list(self.rejected_geometry_frame_ids),
            "candidate_runs": [run.to_dict() for run in self.candidate_runs],
            "frames": [frame.to_dict() for frame in self.frames],
        }


def select_observable_core(
    case: RefinementCase,
    component_trace: GeometricRefinementTrace,
    settings: ObservableCoreSettings | None = None,
) -> tuple[ObservableCoreSelection, GeometricRefinementTrace]:
    """Select one strongest contiguous run of provisional geometry frames."""

    resolved = settings or ObservableCoreSettings()
    validate_geometric_trace(case, component_trace)
    if component_trace.config_sha256 != resolved.component.sha256:
        raise ValueError("component trace and observable-core settings do not match")
    reference_interval_ns = _reference_interval_ns(case)
    maximum_gap_ns = max(
        reference_interval_ns,
        round(reference_interval_ns * resolved.maximum_timestamp_gap_factor),
    )
    provisional_geometry = tuple(
        index
        for index, frame in enumerate(component_trace.frames)
        if frame.component is not None
        and frame.component.status == "selected"
        and frame.component.frame_role is FrameRole.GEOMETRY
    )
    runs = _geometry_runs(
        case,
        component_trace,
        provisional_geometry,
        maximum_gap_ns,
    )
    eligible = [
        run for run in runs if len(run.frame_ids) >= resolved.minimum_core_frames
    ]
    selected = max(eligible, key=_run_selection_key) if eligible else None
    selected_ids = () if selected is None else selected.frame_ids
    selected_set = set(selected_ids)
    rejected_ids = tuple(
        case.frames[index].frame_id
        for index in provisional_geometry
        if case.frames[index].frame_id not in selected_set
    )
    marked_runs = tuple(
        replace(run, selected=selected is not None and run.frame_ids == selected_ids)
        for run in runs
    )
    qualifications = _qualifications(component_trace, selected_set)
    selection = ObservableCoreSelection(
        status="candidate" if selected is not None else "insufficient_evidence",
        reason_codes=() if selected is not None else ("no_connected_geometry_core",),
        reference_interval_ns=reference_interval_ns,
        maximum_connected_gap_ns=maximum_gap_ns,
        minimum_core_frames=resolved.minimum_core_frames,
        core_frame_ids=selected_ids,
        rejected_geometry_frame_ids=rejected_ids,
        candidate_runs=marked_runs,
        frames=qualifications,
    )
    pruned_frames = tuple(
        _apply_core_disposition(frame, selected_set) for frame in component_trace.frames
    )
    return selection, replace(
        component_trace,
        stage="observable_core_selection_v1",
        frames=pruned_frames,
    )


def _reference_interval_ns(case: RefinementCase) -> int:
    intervals = np.diff(
        np.asarray([frame.timestamp_ns for frame in case.frames], dtype=np.int64)
    )
    return max(1, round(float(np.median(intervals))))


def _geometry_runs(
    case: RefinementCase,
    trace: GeometricRefinementTrace,
    geometry_indices: tuple[int, ...],
    maximum_gap_ns: int,
) -> tuple[ObservableCoreRunTrace, ...]:
    raw_runs: list[list[int]] = []
    for index in geometry_indices:
        if (
            not raw_runs
            or index != raw_runs[-1][-1] + 1
            or case.frames[index].timestamp_ns
            - case.frames[raw_runs[-1][-1]].timestamp_ns
            > maximum_gap_ns
        ):
            raw_runs.append([index])
        else:
            raw_runs[-1].append(index)
    return tuple(_run_trace(case, trace, indices) for indices in raw_runs)


def _run_trace(
    case: RefinementCase,
    trace: GeometricRefinementTrace,
    indices: list[int],
) -> ObservableCoreRunTrace:
    components = [trace.frames[index].component for index in indices]
    if any(
        component is None
        or component.component_dominance is None
        or component.resolution_stability_iou is None
        for component in components
    ):
        raise AssertionError("geometry run is missing component metrics")
    return ObservableCoreRunTrace(
        frame_ids=tuple(case.frames[index].frame_id for index in indices),
        start_index=indices[0],
        end_index=indices[-1],
        total_selected_points=sum(
            component.selected_point_count
            for component in components
            if component is not None
        ),
        total_selected_voxels=sum(
            component.selected_voxel_count
            for component in components
            if component is not None
        ),
        minimum_component_dominance=min(
            component.component_dominance
            for component in components
            if component is not None and component.component_dominance is not None
        ),
        minimum_resolution_stability_iou=min(
            component.resolution_stability_iou
            for component in components
            if component is not None and component.resolution_stability_iou is not None
        ),
    )


def _run_selection_key(run: ObservableCoreRunTrace) -> tuple[object, ...]:
    return (
        len(run.frame_ids),
        run.total_selected_points,
        run.total_selected_voxels,
        run.minimum_resolution_stability_iou,
        run.minimum_component_dominance,
        -run.start_index,
    )


def _qualifications(
    trace: GeometricRefinementTrace,
    selected_ids: set[str],
) -> tuple[ObservableFrameQualification, ...]:
    rows: list[ObservableFrameQualification] = []
    for frame in trace.frames:
        component = frame.component
        if component is None:
            raise AssertionError("observable-core selection requires component traces")
        if frame.frame_id in selected_ids:
            disposition = ObservableFrameDisposition.CORE_GEOMETRY
            reasons: tuple[str, ...] = ()
        elif component.status == "selected" and component.frame_role in {
            FrameRole.GEOMETRY,
            FrameRole.POSE_ONLY,
        }:
            disposition = ObservableFrameDisposition.POSE_CANDIDATE
            reasons = tuple(
                dict.fromkeys(
                    (
                        *component.reason_codes,
                        *(
                            ("outside_selected_observable_core",)
                            if component.frame_role is FrameRole.GEOMETRY
                            else ()
                        ),
                    )
                )
            )
        else:
            disposition = ObservableFrameDisposition.UNSUPPORTED
            reasons = component.reason_codes or ("component_not_usable",)
        rows.append(
            ObservableFrameQualification(
                frame_id=frame.frame_id,
                provisional_role=component.frame_role,
                disposition=disposition,
                reason_codes=reasons,
                selected_point_count=component.selected_point_count,
                selected_voxel_count=component.selected_voxel_count,
                component_dominance=component.component_dominance,
                resolution_stability_iou=component.resolution_stability_iou,
            )
        )
    return tuple(rows)


def _apply_core_disposition(
    frame: FrameEvidenceTrace, selected_ids: set[str]
) -> FrameEvidenceTrace:
    component = frame.component
    if component is None:
        raise AssertionError("observable-core selection requires component traces")
    if (
        component.frame_role is FrameRole.GEOMETRY
        and frame.frame_id not in selected_ids
    ):
        return replace(
            frame,
            component=replace(
                component,
                frame_role=FrameRole.POSE_ONLY,
                reason_codes=tuple(
                    dict.fromkeys(
                        (*component.reason_codes, "outside_selected_observable_core")
                    )
                ),
            ),
        )
    return frame
