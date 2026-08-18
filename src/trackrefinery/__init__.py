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
from trackrefinery.refiner import TrackRefiner, validate_outcome
from trackrefinery.review import build_review_bundle
from trackrefinery.serde import read_outcome, write_outcome
from trackrefinery.targets import GoldFramePose, GoldTarget, TargetDataset

__all__ = [
    "AcceptanceThresholds",
    "BenchmarkReport",
    "Box3D",
    "EvaluationReport",
    "FrameCloud",
    "GoldFramePose",
    "GoldTarget",
    "InferenceDataset",
    "InstanceTrack",
    "InsufficientEvidence",
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
    "read_outcome",
    "validate_outcome",
    "write_outcome",
]

__version__ = "0.1.0.dev0"
