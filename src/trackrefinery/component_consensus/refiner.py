"""Stage-gated entry point for V2 component-consensus refinement."""

from __future__ import annotations

from trackrefinery.component_consensus.aggregation import aggregate_geometry_components
from trackrefinery.component_consensus.components import select_object_components
from trackrefinery.component_consensus.settings import ComponentConsensusSettings
from trackrefinery.contracts import (
    InsufficientEvidence,
    RefinementCase,
    RefinementOutcome,
)
from trackrefinery.geometric.trace import (
    FrameRole,
    GeometricRefinementRun,
    GeometricRefinementTrace,
    validate_geometric_trace,
)
from trackrefinery.refiner import TrackRefiner, validate_outcome


class ComponentConsensusRefiner(TrackRefiner):
    """V2 backend, currently gated after anchored component aggregation."""

    def __init__(self, settings: ComponentConsensusSettings | None = None) -> None:
        self.settings = settings or ComponentConsensusSettings()

    def refine_with_trace(self, case: RefinementCase) -> GeometricRefinementRun:
        outcome, trace = self._execute(case)
        validate_outcome(case, outcome)
        validate_geometric_trace(case, trace)
        return GeometricRefinementRun(outcome, trace)

    def _refine(self, case: RefinementCase) -> RefinementOutcome:
        outcome, _ = self._execute(case)
        return outcome

    def _execute(
        self, case: RefinementCase
    ) -> tuple[RefinementOutcome, GeometricRefinementTrace]:
        component_trace = select_object_components(case, self.settings)
        role_counts = {role.value: 0 for role in FrameRole}
        reasons = ["algorithm_stage_incomplete"]
        for frame in component_trace.frames:
            if frame.component is None:
                raise AssertionError("V2 component trace is missing")
            role_counts[frame.component.frame_role.value] += 1
            if frame.component.status != "selected":
                reasons.append(f"component_not_separable:{frame.frame_id}")
        dense_track_supported = (
            role_counts[FrameRole.GEOMETRY.value]
            >= self.settings.track_minimum_geometry_frames
        )
        trace = (
            aggregate_geometry_components(case, component_trace, self.settings)
            if dense_track_supported
            else component_trace
        )
        aggregation = trace.anchored_aggregation
        aggregation_supported = (
            aggregation is not None and aggregation.status == "candidate"
        )
        if not dense_track_supported:
            reasons.append("dense_track_out_of_scope")
            reasons.append("insufficient_geometry_frames")
        elif not aggregation_supported:
            reasons.append("component_alignment_failed")
            if aggregation is not None:
                reasons.extend(aggregation.reason_codes)
        outcome = InsufficientEvidence(
            track_id=case.track.track_id,
            reasons=tuple(dict.fromkeys(reasons)),
            diagnostics={
                "algorithm_version": trace.algorithm_version,
                "config_schema_version": trace.config_schema_version,
                "config_sha256": trace.config_sha256,
                "stage": trace.stage,
                "development_scope": "dense_instances_only",
                "dense_track_supported": dense_track_supported,
                "anchored_aggregation_supported": aggregation_supported,
                "track_minimum_geometry_frames": (
                    self.settings.track_minimum_geometry_frames
                ),
                "frame_role_counts": role_counts,
                "anchored_aggregation": (
                    None if aggregation is None else aggregation.to_dict()
                ),
                "frames": [frame.to_summary_dict() for frame in trace.frames],
            },
        )
        return outcome, trace
