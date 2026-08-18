"""Deterministic geometric-refinement backend and development trace API."""

from trackrefinery.geometric.evidence import select_initial_evidence
from trackrefinery.geometric.refiner import JointCuboidRefiner
from trackrefinery.geometric.settings import (
    GEOMETRIC_ALGORITHM_VERSION,
    GEOMETRIC_CONFIG_SCHEMA_VERSION,
    EvidenceSelectionSettings,
    GeometricRefinementSettings,
)
from trackrefinery.geometric.trace import (
    EVIDENCE_TRACE_CONTRACT,
    EvidenceState,
    FrameEvidenceTrace,
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
    "EvidenceSelectionSettings",
    "EvidenceState",
    "FrameEvidenceTrace",
    "GeometricRefinementRun",
    "GeometricRefinementSettings",
    "GeometricRefinementTrace",
    "GroundPlaneEstimate",
    "JointCuboidRefiner",
    "read_geometric_trace",
    "select_initial_evidence",
    "validate_geometric_trace",
    "write_geometric_trace",
]
