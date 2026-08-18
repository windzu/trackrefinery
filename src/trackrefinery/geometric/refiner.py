"""First deterministic backend entry point with explicit stage gating."""

from __future__ import annotations

from trackrefinery.contracts import (
    InsufficientEvidence,
    RefinementCase,
    RefinementOutcome,
)
from trackrefinery.geometric.evidence import select_initial_evidence
from trackrefinery.geometric.settings import GeometricRefinementSettings
from trackrefinery.geometric.trace import (
    GeometricRefinementRun,
    GeometricRefinementTrace,
    validate_geometric_trace,
)
from trackrefinery.refiner import TrackRefiner, validate_outcome


class JointCuboidRefiner(TrackRefiner):
    """Deterministic joint refiner, currently gated at initial evidence stage."""

    def __init__(self, settings: GeometricRefinementSettings | None = None) -> None:
        self.settings = settings or GeometricRefinementSettings()

    def refine_with_trace(self, case: RefinementCase) -> GeometricRefinementRun:
        """Return the public outcome together with point-level development trace."""

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
        trace = select_initial_evidence(case, self.settings)
        frame_summaries = [frame.to_summary_dict() for frame in trace.frames]
        ground_missing = [
            frame.frame_id for frame in trace.frames if frame.ground_plane is None
        ]
        reasons = ["algorithm_stage_incomplete"]
        if ground_missing:
            reasons.append("ground_support_unavailable")
        outcome = InsufficientEvidence(
            track_id=case.track.track_id,
            reasons=tuple(reasons),
            diagnostics={
                "algorithm_version": trace.algorithm_version,
                "config_schema_version": trace.config_schema_version,
                "config_sha256": trace.config_sha256,
                "stage": trace.stage,
                "frame_count": len(trace.frames),
                "total_point_state_counts": trace.total_counts,
                "ground_support_missing_frame_ids": ground_missing,
                "frames": frame_summaries,
            },
        )
        return outcome, trace
