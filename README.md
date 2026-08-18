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
fixtures, quantitative evaluation, and visual review bundles. No geometry
refinement algorithm is included yet. The first backend is specified as a
deterministic, non-learned joint cuboid and per-frame pose optimizer in the
[Geometric Refinement V1 design](docs/geometric-refinement-v1.md).

## Install and import

```bash
python -m pip install .
# Include visual review generation:
python -m pip install '.[review]'
```

```python
from trackrefinery import InferenceDataset, TrackRefiner

dataset = InferenceDataset.open("my-data/inference")
case = dataset.load_case("scene-001_vehicle-0042")

# A concrete backend subclasses TrackRefiner and implements _refine(case).
result = my_refiner.refine(case)
```

The base package depends only on NumPy. Review rendering is an optional extra;
importing the core package does not require Plotly or Matplotlib.

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
- direct dependencies on X-4D, MMDetection3D, or a proprietary dataset.

## License

Apache-2.0.
