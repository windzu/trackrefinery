"""Deterministic geometric-refinement backend and development trace API."""

from trackrefinery.geometric.envelope import fit_alternating_envelope
from trackrefinery.geometric.evidence import select_initial_evidence
from trackrefinery.geometric.refiner import JointCuboidRefiner
from trackrefinery.geometric.registration import register_canonical_shape
from trackrefinery.geometric.settings import (
    GEOMETRIC_ALGORITHM_VERSION,
    GEOMETRIC_CONFIG_SCHEMA_VERSION,
    EnvelopeFittingSettings,
    EvidenceSelectionSettings,
    GeometricRefinementSettings,
    RegistrationSettings,
)
from trackrefinery.geometric.trace import (
    EVIDENCE_TRACE_CONTRACT,
    AggregateSharpnessTrace,
    AnchoredAggregationTrace,
    CanonicalShapeTrace,
    CuboidFitTrace,
    EvidenceState,
    FrameComponentTrace,
    FrameEvidenceTrace,
    FrameRegistrationTrace,
    FrameRole,
    GeometricRefinementRun,
    GeometricRefinementTrace,
    GroundPlaneEstimate,
    read_geometric_trace,
    validate_geometric_trace,
    write_geometric_trace,
)

__all__ = [
    "EVIDENCE_TRACE_CONTRACT",
    "GEOMETRIC_ALGORITHM_VERSION",
    "GEOMETRIC_CONFIG_SCHEMA_VERSION",
    "AggregateSharpnessTrace",
    "AnchoredAggregationTrace",
    "CanonicalShapeTrace",
    "CuboidFitTrace",
    "EnvelopeFittingSettings",
    "EvidenceSelectionSettings",
    "EvidenceState",
    "FrameComponentTrace",
    "FrameEvidenceTrace",
    "FrameRegistrationTrace",
    "FrameRole",
    "GeometricRefinementRun",
    "GeometricRefinementSettings",
    "GeometricRefinementTrace",
    "GroundPlaneEstimate",
    "JointCuboidRefiner",
    "RegistrationSettings",
    "fit_alternating_envelope",
    "read_geometric_trace",
    "register_canonical_shape",
    "select_initial_evidence",
    "validate_geometric_trace",
    "write_geometric_trace",
]
