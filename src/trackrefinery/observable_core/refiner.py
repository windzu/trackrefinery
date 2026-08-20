"""Stage-gated deterministic observable-core refiner."""

from __future__ import annotations

from dataclasses import replace

from trackrefinery.component_consensus import (
    aggregate_geometry_components,
    select_object_components,
)
from trackrefinery.contracts import (
    InsufficientEvidence,
    RefinementCase,
    RefinementOutcome,
)
from trackrefinery.geometric.trace import (
    GeometricRefinementRun,
    GeometricRefinementTrace,
    validate_geometric_trace,
)
from trackrefinery.observable_core.selection import select_observable_core
from trackrefinery.observable_core.settings import (
    OBSERVABLE_CORE_ALGORITHM_VERSION,
    OBSERVABLE_CORE_CONFIG_SCHEMA_VERSION,
    ObservableCoreSettings,
)
from trackrefinery.refiner import TrackRefiner, validate_outcome


class ObservableCoreRefiner(TrackRefiner):
    """MVP backend, gated after connected-core aggregation."""

    def __init__(self, settings: ObservableCoreSettings | None = None) -> None:
        self.settings = settings or ObservableCoreSettings()

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
        component_trace = select_object_components(case, self.settings.component)
        selection, selected_trace = select_observable_core(
            case, component_trace, self.settings
        )
        if selection.status == "candidate":
            trace = aggregate_geometry_components(
                case, selected_trace, self.settings.component
            )
            trace = replace(trace, stage="observable_core_aggregation_v1")
        else:
            trace = selected_trace
        trace = replace(
            trace,
            algorithm_version=OBSERVABLE_CORE_ALGORITHM_VERSION,
            config_schema_version=OBSERVABLE_CORE_CONFIG_SCHEMA_VERSION,
            config_sha256=self.settings.sha256,
            settings_json=self.settings.canonical_json(),
        )
        aggregation = trace.anchored_aggregation
        aggregation_supported = (
            aggregation is not None and aggregation.status == "candidate"
        )
        reasons = ["algorithm_stage_incomplete"]
        if selection.status != "candidate":
            reasons.extend(selection.reason_codes)
        elif not aggregation_supported:
            reasons.append("observable_core_alignment_failed")
            if aggregation is not None:
                reasons.extend(aggregation.reason_codes)
        outcome = InsufficientEvidence(
            track_id=case.track.track_id,
            reasons=tuple(dict.fromkeys(reasons)),
            diagnostics={
                "algorithm_version": OBSERVABLE_CORE_ALGORITHM_VERSION,
                "config_schema_version": OBSERVABLE_CORE_CONFIG_SCHEMA_VERSION,
                "config_sha256": self.settings.sha256,
                "stage": trace.stage,
                "observable_core": selection.to_dict(),
                "anchored_aggregation_supported": aggregation_supported,
                "anchored_aggregation": (
                    None if aggregation is None else aggregation.to_dict()
                ),
            },
        )
        return outcome, trace
