# TrackRefinery

TrackRefinery is a single-instance, multi-frame 3D object refinement library.
It consumes full per-frame point clouds, one already-associated detection
track, and exact frame poses, then estimates:

- one stable, canonical size for each rigid object instance;
- authoritative refined poses for the supported observed frames;
- diagnostics stating whether the available evidence supports the result.

The caller does not choose how far to enlarge and crop each detection. Evidence
selection, multi-frame alignment, canonical geometry fitting, and per-frame
pose refinement are the library's job. Detection, tracking/association,
multi-LiDAR fusion, and automatic-label release remain upstream or downstream
responsibilities.

## Status

The framework is usable as an installable Python package. It contains the
validated public contracts, source/target-isolated datasets, deterministic
fixtures, quantitative evaluation, and visual review bundles.

Real-data review has accepted a deterministic
[observable-core MVP](docs/observable-core-refinement-v1.md). It estimates one
canonical size from a connected reliable subset, refines only poses that pass
fixed-geometry gates, and explicitly marks sparse or contradictory frames as
unsupported. `success` retains all-frame authority; `partial_success` exposes
the narrower supported-frame authority without copying coarse tails into the
result.

The [object-centric foundation exploration](docs/object-centric-foundation-exploration-v1.md)
is deferred to future coverage expansion for incomplete or unobservable cases.
Direct trajectory box regression remains only a negative-control baseline. No
learned backend is release-qualified yet.

[Component-Consensus Geometric Refinement V2](docs/geometric-refinement-v2.md)
is retained as a frozen comparison and diagnostic direction. It does not act
as a silent success fallback. The `ComponentConsensusRefiner` implements its
first three gated stages:
bounded ROI and ground handling, deterministic 3D component extraction,
cross-resolution selection stability, conservative merged-component rejection,
`geometry` / `pose_only` / `trajectory_only` frame roles, and anchored
geometry-frame aggregation. Corrections are bounded and individually
rejectable; a regressing proposal retains the exact coarse pose. It returns
`insufficient_evidence` until canonical size, fixed-size pose refinement, and
final acceptance gates are implemented.

The normal-aware Stage 3 pose graph and observable Stage 4 canonical-cuboid
estimator are available as explicit experiment APIs. Stage 4 requires repeated
support for all physical boundaries, records leave-one-frame-out and
resolution stability, and rejects a missing opposing face instead of using a
model/category size fallback. These experiments still do not publish
`RefinementSuccess`; reviewed-target calibration and fixed-shape per-frame pose
refinement remain required.

The current default is a dense-first MVP profile. Only same-track frames with
at least 1,000 selected component points and strong relative spatial support
may define geometry, and a track needs five such frames. Sparse tracks remain
visible in diagnostics as `dense_track_out_of_scope`; they are not silently
fed into the future size estimator. A dense component that materially exits a
loose coarse-box envelope is rejected as inseparable instead of being cropped
to manufacture a clean-looking target.

The existing `JointCuboidRefiner` implements the rejected
[V1 experimental design](docs/geometric-refinement-v1.md). It is retained as a
trace-only regression baseline and always returns `insufficient_evidence`;
real-data review showed that its local registration residual could improve
while the aggregate vehicle geometry became worse.

## Install and import

```bash
python -m pip install .
# Include the deterministic geometric backend:
python -m pip install '.[geometric]'
# Include visual review generation:
python -m pip install '.[geometric,review]'
# Include the optional X-4D Dataset 0.17 development adapter:
python -m pip install '.[geometric,review,x4d]'
```

```python
from trackrefinery import InferenceDataset, TrackRefiner

dataset = InferenceDataset.open("my-data/inference")
case = dataset.load_case("scene-001_vehicle-0042")

# A concrete backend subclasses TrackRefiner and implements _refine(case).
result = my_refiner.refine(case)
```

The legacy V1 backend can be used to reproduce and inspect its evidence,
provisional poses, persistent canonical shape, and fitted cuboid candidate
without producing a false refinement:

```python
from trackrefinery import JointCuboidRefiner, build_review_bundle

run = JointCuboidRefiner().refine_with_trace(case)
build_review_bundle(
    case,
    run.outcome,
    "review/case",
    trace=run.trace,
    data_source="synthetic-v1",  # or the real dataset/Clip identifier
)
```

The V2 component and anchored-aggregation stages are independently inspectable
without claiming a refinement result:

```python
from trackrefinery import ComponentConsensusRefiner

run = ComponentConsensusRefiner().refine_with_trace(case)
assert run.trace.stage == "anchored_component_aggregation_v2"
print(run.trace.anchored_aggregation.to_dict())
for frame in run.trace.frames:
    print(
        frame.frame_id,
        frame.component.status,
        frame.component.frame_role,
        frame.registration,
    )
```

Stage 3 correction capability can be tested without pretending that an
already-good natural track contains a large error. The controlled-recovery
utility freezes component selection, keeps the anchor unchanged, injects a
known smooth pose drift into other geometry frames, and reports how much of
that drift the aggregation stage removes:

```python
from trackrefinery import (
    DEFAULT_CONTROLLED_PERTURBATION_PROFILES,
    build_controlled_recovery_bundle,
    run_controlled_recovery,
)

recovery = run_controlled_recovery(
    case,
    profile=DEFAULT_CONTROLLED_PERTURBATION_PROFILES[-1],  # 15 cm / 2 degrees
    component_trace=run.trace,
)
build_controlled_recovery_bundle(
    recovery,
    "review/recovery/strong",
    data_source="frozen detector track",
)
```

An accepted V3 trace can be passed to the Stage 4 sizing experiment:

```python
from trackrefinery import fit_observable_canonical_cuboid

size_run = fit_observable_canonical_cuboid(case, stage3_trace)
print(size_run.canonical_cuboid.status)
print(size_run.canonical_cuboid.provisional_size_lwh)
```

The value is an experimental diagnostic, not a released annotation result.

Its REFERENCE view is only the frozen model-track proxy used by the evaluator;
it is not reviewed gold, and its poses are not available to the perturbed
algorithm run.

Multiple case bundles can be placed under one directory and exposed through a
single tabbed review page with `build_review_suite()`. Real-data catalogs use
`build_clip_review_suite()` so each outer tab is one Clip and all of its
instances are tiled inside the tab. When a physically
separate gold target is supplied, each case also contains an annotation-pose-
aligned aggregate for review-only comparison; it is never exposed to the
refinement backend.

`trackrefinery-build-controlled-recovery-suite` generates the default mild,
medium, and strong profiles for one or more dense cases and writes a separate
case-tabbed diagnostic suite. These controlled cases are not inserted into the
real Clip inventory as if they were source Clips.

The base package depends only on NumPy. Registration uses SciPy through the
`geometric` extra, review rendering uses the `review` extra, and the Dataset
0.17 adapter uses the `x4d` extra. Importing the core contracts does not require
SciPy, Plotly, Matplotlib, or X-4D.

## Quick start

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/trackrefinery-generate-synthetic \
  --output tests/fixtures/synthetic_v1
.venv/bin/trackrefinery-validate \
  tests/fixtures/synthetic_v1/inference
.venv/bin/trackrefinery-validate-targets \
  tests/fixtures/synthetic_v1/targets
.venv/bin/pytest
```

The generated fixture contains static, moving, partially observed, outlier,
clutter, and neighboring-object cases. Real or licensed data stays under the
ignored `.data/` directory and is never redistributed.

See [architecture](docs/architecture.md), the
[data contract](docs/data-contract-v1.md), and the
[development-data plan](docs/development-data.md). Evaluation and visual
feedback are specified in
[evaluation-and-review](docs/evaluation-and-review.md). Importable entry points
are summarized in the [Python API guide](docs/python-api.md).

## Non-goals

- detection, association, or online tracking;
- sensor calibration or coordinate fusion;
- Clip-level multi-object orchestration and annotation release policy;
- an annotation UI;
- core/runtime coupling to X-4D, MMDetection3D, or a proprietary dataset;
  ecosystem-specific development adapters remain optional.

## License

Apache-2.0.
