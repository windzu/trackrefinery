# Python API Guide

The pip package is the primary integration surface. CLI commands are thin
development-tool adapters around the same importable functions.

## Implement a backend

```python
from trackrefinery import (
    InsufficientEvidence,
    RefinementCase,
    RefinementOutcome,
    TrackRefiner,
)


class MyRefiner(TrackRefiner):
    def _refine(self, case: RefinementCase) -> RefinementOutcome:
        # The real algorithm belongs here. This example deliberately makes no
        # geometry claim.
        return InsufficientEvidence(
            track_id=case.track.track_id,
            reasons=("backend_not_implemented",),
        )
```

`TrackRefiner.refine()` calls `_refine()` and then enforces that the result
belongs to the input track. A successful backend must return exactly one pose
for every input observation in the same order.

## Inspect the deterministic backend

The first backend currently implements initial point-evidence selection,
ground estimation, robust per-frame upright registration, and persistent
canonical shape aggregation. Until cuboid fitting, alternating reassignment,
and success gates are implemented, it intentionally returns
`algorithm_stage_incomplete` rather than publishing provisional geometry:

```python
from trackrefinery import JointCuboidRefiner, write_geometric_trace

run = JointCuboidRefiner().refine_with_trace(case)
assert run.outcome.status == "insufficient_evidence"
write_geometric_trace("traces/case", run.trace)

assert run.trace.canonical_shape is not None
for frame in run.trace.frames:
    # These are trace-only candidates, not released refinement results.
    print(frame.registration.status)
```

Settings are immutable, content-addressed, and internal to the backend:

```python
from trackrefinery import (
    EvidenceSelectionSettings,
    GeometricRefinementSettings,
    JointCuboidRefiner,
    RegistrationSettings,
)

settings = GeometricRefinementSettings(
    evidence=EvidenceSelectionSettings(roi_margin_xyz_m=(1.1, 1.0, 0.55)),
    registration=RegistrationSettings(canonical_support_radius_m=0.15),
)
backend = JointCuboidRefiner(settings)
print(settings.sha256)
```

The required caller contract still supplies full frame clouds. These settings
are versioned algorithm configuration, not a caller-owned crop request.
Install `trackrefinery[geometric]` before invoking this backend; importing the
base package and using its contracts remains NumPy-only.

## Load source-only input

```python
from trackrefinery import InferenceDataset

dataset = InferenceDataset.open("benchmark/inference")
case = dataset.load_case("scene-001_vehicle-0042")
result = MyRefiner().refine(case)
```

`FrameCloud` objects are immutable. When several cases reference one sequence,
the loader reuses the same in-memory frame and point-array objects.
`InferenceDataset` has no target-loading API.

## Serialize a result

```python
from trackrefinery import read_outcome, write_outcome

write_outcome("predictions/run-001/case.json", case.case_id, result)
case_id, restored = read_outcome("predictions/run-001/case.json")
```

## Evaluate explicitly

Evaluation requires a separately opened target root:

```python
from trackrefinery import TargetDataset, evaluate_case

targets = TargetDataset.open("benchmark/targets")
gold = targets.load_target(case.case_id)
report = evaluate_case(case, result, gold)
```

No default correction tolerance is embedded in the package. Once calibrated,
an `AcceptanceThresholds` value makes the strict track pass/fail decision
versionable and reproducible.

`evaluate_suite()` consumes one `<case_id>.json` prediction for every indexed
case and reports strict passes, unexpected successes, missed refinable tracks,
and catastrophic successes overall and by category.

## Generate a visual review bundle

Install the optional renderer and call the same library function used by the
CLI:

```bash
python -m pip install 'trackrefinery[review]'
```

```python
from trackrefinery import build_review_bundle

build_review_bundle(
    case,
    run.outcome,
    "review/run-001/case",
    target=gold,
    trace=run.trace,
)
```

The bundle contains fixed aggregate artifacts, the provisional
`canonical_shape.npz`, orthographic thumbnails, metrics, optional evidence-mask
sidecars, and a self-contained interactive HTML view. With a trace, initial
target, ambiguous, background, and ground points can be toggled independently;
candidate registration poses are explicitly labeled and never presented as a
refined result. Serve it with
`trackrefinery-review review/run-001/case --open`.

The CLI accepts the same sidecar through
`trackrefinery-build-review --trace traces/case/evidence_trace.json ...`.
