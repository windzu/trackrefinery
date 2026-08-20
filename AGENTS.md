# TrackRefinery Agent Instructions

TrackRefinery is an independent, open-source library for single-instance,
multi-frame 3D object refinement. Given scene point clouds, one already
associated detection track, and exact frame poses, it estimates one canonical
instance size and authoritative refined object poses for the supported input
observations.

## Product boundary

- Keep the core independent of X-4D, X-Points, MMDetection3D, ROS, and any one
  dataset format. Integrations belong in adapters.
- Refine exactly one already-associated object track per logical request, not
  isolated frames or a complete multi-object scene.
- Treat detector/tracker output as evidence, never as ground truth.
- Consume full per-frame point clouds and choose the enlarged target evidence
  region inside the library. The caller does not own a crop-margin policy.
- Use one canonical size per rigid instance and refine per-frame poses against
  that geometry.
- Prefer a deterministic observable core over weak whole-track coverage.
  Every input frame must be explicitly classified as authoritative geometry,
  authoritative pose-only, or unsupported; never present an inherited coarse
  pose as refined.
- When evidence cannot support the required accuracy, return an explicit
  insufficient-evidence outcome instead of presenting copied or guessed boxes
  as a successful refinement. Release policy belongs to the caller.

## Data and safety

- Inputs and evaluation targets must be stored separately. Production loaders
  must never expose targets to a refinement backend.
- Do not commit private X-4D clips, customer data, model weights, or licensed
  third-party datasets. Store local material under `.data/`.
- Commit only generated synthetic fixtures and tiny assets whose licenses are
  documented.
- Preserve integer nanosecond timestamps and explicit coordinate-frame names.
- Each frame's points and coarse detection are expressed in that frame's
  declared annotation frame. Exact annotation-frame-to-world poses are part of
  the input. Sensor fusion and calibration are adapter responsibilities.

## Development

```bash
python -m pip install -e '.[dev]'
trackrefinery-generate-synthetic --output tests/fixtures/synthetic_v1
ruff format --check .
ruff check .
pytest
```

Architecture or contract changes must update the corresponding file under
`docs/` in the same pull request. Use short-lived `agent/*` branches and merge
through a pull request after CI passes.
