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
    result,
    "review/run-001/case",
    target=gold,
    evaluation=report,
)
```

The bundle contains fixed aggregate artifacts, orthographic thumbnails,
metrics, and a self-contained interactive HTML view. Serve it with
`trackrefinery-review review/run-001/case --open`.
