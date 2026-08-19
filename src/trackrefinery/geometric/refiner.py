"""First deterministic backend entry point with explicit stage gating."""

from __future__ import annotations

from trackrefinery.contracts import (
    InsufficientEvidence,
    RefinementCase,
    RefinementOutcome,
)
from trackrefinery.geometric.envelope import fit_alternating_envelope
from trackrefinery.geometric.evidence import select_initial_evidence
from trackrefinery.geometric.registration import register_canonical_shape
from trackrefinery.geometric.settings import GeometricRefinementSettings
from trackrefinery.geometric.trace import (
    GeometricRefinementRun,
    GeometricRefinementTrace,
    validate_geometric_trace,
)
from trackrefinery.refiner import TrackRefiner, validate_outcome


class JointCuboidRefiner(TrackRefiner):
    """Deterministic joint refiner, gated before final observability checks."""

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
        initial_trace = select_initial_evidence(case, self.settings)
        registration_trace = register_canonical_shape(
            case, initial_trace, self.settings
        )
        trace = fit_alternating_envelope(case, registration_trace, self.settings)
        frame_summaries = [frame.to_summary_dict() for frame in trace.frames]
        ground_missing = [
            frame.frame_id for frame in trace.frames if frame.ground_plane is None
        ]
        reasons = ["algorithm_stage_incomplete"]
        if ground_missing:
            reasons.append("ground_support_unavailable")
        failed_registration = [
            frame
            for frame in trace.frames
            if frame.registration is None or frame.registration.status != "registered"
        ]
        if any(
            frame.registration is not None
            and "insufficient_target_points" in frame.registration.reason_codes
            for frame in failed_registration
        ):
            reasons.append("insufficient_target_points")
        if (
            trace.canonical_shape is None
            or not trace.canonical_shape.converged
            or trace.cuboid_fit is None
            or not trace.cuboid_fit.converged
        ):
            reasons.append("optimization_not_converged")
        if trace.cuboid_fit is not None and trace.cuboid_fit.reason_codes:
            reasons.extend(
                reason
                for reason in trace.cuboid_fit.reason_codes
                if reason not in reasons
            )
        reasons.extend(
            f"pose_unobservable:{frame.frame_id}" for frame in failed_registration
        )
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
                "registered_frame_count": len(trace.frames) - len(failed_registration),
                "canonical_shape": (
                    None
                    if trace.canonical_shape is None
                    else trace.canonical_shape.to_summary_dict()
                ),
                "cuboid_fit": (
                    None if trace.cuboid_fit is None else trace.cuboid_fit.to_dict()
                ),
                "frames": frame_summaries,
            },
        )
        return outcome, trace
