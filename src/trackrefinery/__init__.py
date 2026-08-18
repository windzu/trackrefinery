"""Public API for single-instance, multi-frame 3D refinement."""

from trackrefinery.contracts import (
    Box3D,
    FrameCloud,
    InstanceTrack,
    InsufficientEvidence,
    ObservationKind,
    Pose3D,
    RefinedFramePose,
    RefinementCase,
    RefinementOutcome,
    RefinementSuccess,
    TrackObservation,
)
from trackrefinery.dataset import InferenceDataset
from trackrefinery.evaluation import (
    AcceptanceThresholds,
    BenchmarkReport,
    EvaluationReport,
    evaluate_case,
    evaluate_suite,
)
from trackrefinery.geometric import (
    EVIDENCE_TRACE_CONTRACT,
    GEOMETRIC_ALGORITHM_VERSION,
    GEOMETRIC_CONFIG_SCHEMA_VERSION,
    EvidenceSelectionSettings,
    EvidenceState,
    FrameEvidenceTrace,
    GeometricRefinementRun,
    GeometricRefinementSettings,
    GeometricRefinementTrace,
    GroundPlaneEstimate,
    JointCuboidRefiner,
    read_geometric_trace,
    select_initial_evidence,
    validate_geometric_trace,
    write_geometric_trace,
)
from trackrefinery.refiner import TrackRefiner, validate_outcome
from trackrefinery.review import build_review_bundle
from trackrefinery.serde import read_outcome, write_outcome
from trackrefinery.targets import GoldFramePose, GoldTarget, TargetDataset

__all__ = [
    "EVIDENCE_TRACE_CONTRACT",
    "GEOMETRIC_ALGORITHM_VERSION",
    "GEOMETRIC_CONFIG_SCHEMA_VERSION",
    "AcceptanceThresholds",
    "BenchmarkReport",
    "Box3D",
    "EvaluationReport",
    "EvidenceSelectionSettings",
    "EvidenceState",
    "FrameCloud",
    "FrameEvidenceTrace",
    "GeometricRefinementRun",
    "GeometricRefinementSettings",
    "GeometricRefinementTrace",
    "GoldFramePose",
    "GoldTarget",
    "GroundPlaneEstimate",
    "InferenceDataset",
    "InstanceTrack",
    "InsufficientEvidence",
    "JointCuboidRefiner",
    "ObservationKind",
    "Pose3D",
    "RefinedFramePose",
    "RefinementCase",
    "RefinementOutcome",
    "RefinementSuccess",
    "TargetDataset",
    "TrackObservation",
    "TrackRefiner",
    "build_review_bundle",
    "evaluate_case",
    "evaluate_suite",
    "read_geometric_trace",
    "read_outcome",
    "select_initial_evidence",
    "validate_geometric_trace",
    "validate_outcome",
    "write_geometric_trace",
    "write_outcome",
]

__version__ = "0.1.0.dev0"
