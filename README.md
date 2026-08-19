# TrackRefinery

TrackRefinery is a single-instance, multi-frame 3D object refinement library.
It consumes full per-frame point clouds, one already-associated detection
track, and exact frame poses, then estimates:

- one stable, canonical size for each rigid object instance;
- a refined pose for every observed frame;
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

The accepted algorithm direction is
[Component-Consensus Geometric Refinement V2](docs/geometric-refinement-v2.md):
extract one object component per usable frame, aggregate only reliable frames,
fit one canonical size, and refine every frame pose with that size fixed. It
does not use learned priors, sensor rays, free-space, or occupancy modeling.
The `ComponentConsensusRefiner` now implements its first gated stage: bounded
ROI and ground handling, deterministic 3D component extraction, cross-resolution
selection stability, conservative merged-component rejection, and provisional
`geometry` / `pose_only` / `trajectory_only` frame roles. It returns
`insufficient_evidence` until aggregation, canonical size, fixed-size pose
refinement, and acceptance gates are implemented.

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

The V2 component stage is independently inspectable without claiming a
refinement result:

```python
from trackrefinery import ComponentConsensusRefiner

run = ComponentConsensusRefiner().refine_with_trace(case)
assert run.trace.stage == "component_selection_v2"
for frame in run.trace.frames:
    print(frame.frame_id, frame.component.status, frame.component.frame_role)
```

Multiple case bundles can be placed under one directory and exposed through a
single tabbed review page with `build_review_suite()`. Real-data catalogs use
`build_clip_review_suite()` so each outer tab is one Clip and all of its
instances are tiled inside the tab. When a physically
separate gold target is supplied, each case also contains an annotation-pose-
aligned aggregate for review-only comparison; it is never exposed to the
refinement backend.

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
