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
belongs to the input track. `RefinementSuccess` returns exactly one pose for
every input observation in the same order. `PartialRefinementSuccess` returns
authoritative poses for a supported subset and explicitly accounts for every
other input frame with `UnsupportedFrame`; the two ordered lists must partition
the input.

```python
from trackrefinery import (
    PartialRefinementSuccess,
    RefinedFramePose,
    RefinedFrameRole,
    UnsupportedFrame,
)

return PartialRefinementSuccess(
    track_id=case.track.track_id,
    canonical_size_lwh=(4.72, 1.86, 1.61),
    frame_poses=(RefinedFramePose("frame-004", pose, RefinedFrameRole.GEOMETRY),),
    unsupported_frames=(UnsupportedFrame("frame-003", ("sparse_track_tail",)),),
)
```

The example is schematic: a real result must account for every input frame.

## Inspect the legacy V1 backend

`JointCuboidRefiner` implements the rejected V1 experiment: initial
point-evidence selection, ground estimation, all-frame upright registration,
canonical shape aggregation, evidence reassignment, and visible-envelope
fitting. It is retained to reproduce the real-data regression and intentionally
returns `algorithm_stage_incomplete` rather than publishing provisional
geometry. It is not the accepted V2 implementation:

```python
from trackrefinery import JointCuboidRefiner, write_geometric_trace

run = JointCuboidRefiner().refine_with_trace(case)
assert run.outcome.status == "insufficient_evidence"
write_geometric_trace("traces/case", run.trace)

assert run.trace.canonical_shape is not None
assert run.trace.cuboid_fit is not None
print(run.trace.cuboid_fit.canonical_size_lwh)  # trace-only, not released
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

## Inspect V2 component selection and anchored aggregation

The accepted backend currently stops after selecting one object component per
frame, assigning provisional frame roles, and sequentially aggregating reliable
geometry frames. A green `target` state in the
portable evidence enum means `selected object component` for this V2 stage;
orange means a competing or inseparable component. Stage 3 produces provisional
alignment traces but does not estimate dimensions or publish a refinement:

```python
from trackrefinery import ComponentConsensusRefiner, write_geometric_trace

run = ComponentConsensusRefiner().refine_with_trace(case)
assert run.outcome.status == "insufficient_evidence"
assert run.trace.stage == "anchored_component_aggregation_v2"
print(run.trace.anchored_aggregation.to_dict())
for frame in run.trace.frames:
    decision = frame.component
    print(frame.frame_id, decision.status, decision.frame_role, frame.registration)
write_geometric_trace("traces/v2-component-case", run.trace)
```

The review bundle renders explicit `selected object component`, `competing /
inseparable component`, `other ROI points`, and `removed ground` labels. Dense
supported tracks also receive a same-selected-point BEFORE/AFTER view of the
input-track alignment and provisional anchored alignment.

## Validate known-error Stage 3 recovery

`run_controlled_recovery()` is an evaluator-side diagnostic. It reuses frozen
component indices, injects deterministic smooth drift into non-anchor geometry
poses, and runs aggregation against that perturbed case. The original model
poses remain a proxy reference outside the algorithm run. The evaluator also
runs the same backend on the unperturbed frozen components so injected-error
recovery can be measured independently of a repeatable natural-input
correction. The controlled drift is applied around that unperturbed algorithm
output so the declared perturbation, rather than a hidden sum of natural and
artificial corrections, is what exercises the production correction envelope.

```python
from trackrefinery import (
    DEFAULT_CONTROLLED_PERTURBATION_PROFILES,
    build_controlled_recovery_bundle,
    run_controlled_recovery,
)

recovery = run_controlled_recovery(
    case,
    profile=DEFAULT_CONTROLLED_PERTURBATION_PROFILES[-1],
    component_trace=run.trace,
)
print(recovery.report.translation_rms_reduction_fraction)
print(recovery.report.yaw_rms_reduction_fraction)
print(recovery.report.equivariant_translation_rms_reduction_fraction)
print(recovery.report.equivariant_yaw_rms_reduction_fraction)
build_controlled_recovery_bundle(
    recovery,
    "review/recovery/strong",
    data_source="frozen model-track proxy",
)
```

To evaluate the V3 sidecar without changing the default refiner, pass an
explicit backend:

```python
from trackrefinery import aggregate_geometry_components_pose_graph


def normal_pose_graph(case, trace, settings):
    return aggregate_geometry_components_pose_graph(
        case,
        trace,
        settings,
        variant="normal_aware_pose_graph",
    ).trace


recovery = run_controlled_recovery(
    case,
    profile=DEFAULT_CONTROLLED_PERTURBATION_PROFILES[-1],
    component_trace=run.trace,
    aggregation_backend=normal_pose_graph,
    algorithm_variant="normal_aware_pose_graph",
)
```

The default profiles are 5 cm / 0.5 degrees, 10 cm / 1 degree, and 15 cm /
2 degrees. `build_controlled_recovery_suite()` indexes their bundles with one
tab per source case. The CLI equivalent is:

```bash
trackrefinery-build-controlled-recovery-suite \
  --inference-root my-data/inference \
  --case-id scene-001_vehicle-0042 \
  --algorithm-variant normal_aware_pose_graph \
  --output review/recovery
```

## Inspect the experimental Stage 4 canonical cuboid

Stage 4 accepts only an observable V3 pose-graph trace. It returns a trace and
an experiment sidecar, not a public refinement success:

```python
from trackrefinery import fit_observable_canonical_cuboid

size_run = fit_observable_canonical_cuboid(case, stage3_trace)
print(size_run.canonical_cuboid.status)
print(size_run.canonical_cuboid.reason_codes)
print(size_run.canonical_cuboid.provisional_size_lwh)
size_run.canonical_cuboid.write_json("traces/case/canonical-cuboid.json")
```

An accepted experiment applies one common center/yaw transform to every Stage
3 geometry pose and exposes one common size in `size_run.trace.cuboid_fit`. A
rejected run may retain a provisional size in the sidecar for diagnosis, but
its trace contains no materialized candidate dimensions. The estimator does
not read coarse dimensions, category priors, sensor rays, or annotations.

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
from trackrefinery import (
    build_clip_review_suite,
    build_review_bundle,
    build_review_suite,
)

build_review_bundle(
    case,
    run.outcome,
    "review/run-001/case",
    target=gold,
    trace=run.trace,
    data_source="synthetic-v1",  # or the real dataset/Clip identifier
)

build_review_suite(
    "review/run-001",
    [
        "review/run-001/cases/static_complete",
        "review/run-001/cases/moving_complete",
    ],
    title="Regression run 001",
)

build_clip_review_suite(
    "review/real-clips",
    {
        "clip-001": [
            "review/real-clips/clips/clip-001/instances/vehicle-01",
            "review/real-clips/clips/clip-001/instances/vehicle-02",
        ],
        "clip-002": [
            "review/real-clips/clips/clip-002/instances/vehicle-05",
        ],
    },
)
```

The bundle contains fixed aggregate artifacts, the provisional
`canonical_shape.npz`, orthographic thumbnails, metrics, optional evidence-mask
sidecars, and a self-contained interactive HTML view. With a trace, current
target, ambiguous, background, and ground points can be toggled independently;
candidate registration poses and the visible-envelope cuboid are explicitly
labeled and never presented as a refined result. The declared data source is
shown on the page to distinguish generated fixtures from real Clips. Serve it
with `trackrefinery-review review/run-001/case --open`.

The suite index presents all case bundles as top-level tabs. A case with a
separate target includes an `Annotation aggregate` tab made from the same
display points aligned by gold poses. This is a review-only artifact and does
not change the source-only inference contract.

The real-Clip catalog uses prominent mode badges and filters to distinguish
algorithm candidates, frozen model-track baselines, and annotation references.
All card images are multi-frame aggregates and name their alignment source.
Algorithm cards lead with top and side A/B figures built from identical point
indices, frame colors, and plot axes: the left side uses frozen input-track
poses and the right side uses algorithm poses. Evidence classification and the
cross-frame-supported canonical shape remain separately labeled views.

The CLI accepts the same sidecar through
`trackrefinery-build-review --trace traces/case/evidence_trace.json ...`.

## Export an X-4D Clip into source-only inputs

Install the optional adapter alongside a Dataset 0.17 compatible Devkit:

```bash
python -m pip install 'trackrefinery[x4d,review]'
```

```python
from trackrefinery import export_x4d_clip_inference

exported = export_x4d_clip_inference(
    "/data/clips/20260817_G150-002_000",
    "/data/candidates/centerpoint-offline-geometry-v3.json",
    "/data/trackrefinery/inference/20260817_G150-002_000",
    role="qualitative",
    source_kind="model_candidate",
)
```

The adapter discovers all metadata-declared LiDAR channels, transforms each
current-keyframe cloud into `meta.annotation_frame_id`, preserves exact
`uint64` point times and sensor provenance, records filtered non-finite point
counts, and writes one shared frame sequence plus one track file per instance.
It accepts either native annotation-v3 JSON or a protocol-v2 candidate
envelope. The export step and algorithm step remain separate: the refiner opens
only the generated `inference/` root and has no path to native annotations or
evaluation targets.
