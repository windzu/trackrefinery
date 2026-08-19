# Architecture

## Goal and unit of work

TrackRefinery refines one already-associated rigid-object track. Its input is
the sequence of full scene point clouds containing that object, the object's
coarse detection in each input frame, and exact frame poses. Its output is one
canonical `(length, width, height)` plus a refined object pose for every input
observation.

The public algorithm unit is one instance track. A batch runner may share the
same immutable frame-cloud objects across many tracks, but that is an execution
optimization rather than a change to the refinement contract.

```text
upstream detector and tracker
  -> full frame clouds + one associated detection track + frame poses
  -> TrackRefinery extracts one component per usable frame
  -> reliable-frame aggregation and one canonical size
  -> fixed-size per-frame pose refinement
  -> canonical size + refined poses + diagnostics
  -> caller decides whether and how to publish the result
```

## Boundaries

The core begins after synchronization, calibration, per-frame multi-sensor
fusion, detection, and track association. In every frame, all selected LiDAR
points and the coarse detection use that frame's declared annotation frame.
The input also carries the exact transform from that annotation frame to a
shared sequence/world frame. This keeps the core independent of vehicle layout,
topic names, frame rate, and bag format without discarding the pose information
required for temporal aggregation.

The caller supplies full frame clouds, not a pre-cropped object tensor and not
a crop-margin setting. TrackRefinery owns enlarged evidence-region selection,
target/background separation, multi-frame canonicalization, geometry fitting,
and pose refinement. It may expose internal evidence masks for diagnostics, but
those masks are not required input.

The core does not read annotations from the input tree. Evaluation targets live
under a separate tree and are opened only by benchmark code. This prevents a
backend from accidentally consuming labels when a labeled clip is reused for
inference.

Detection, association, track lifecycle, gap filling, head/tail extrapolation,
sensor fusion, calibration, and Clip candidate construction are outside the
library. X-4D/MMDetection3D integration is an adapter. Automatic release is a
caller policy, not a TrackRefinery decision.

Each successful rigid-instance result has one canonical size shared exactly by
all refined frame poses. A future articulated-object extension must be explicit
rather than weakening this invariant. If the evidence cannot support the
required accuracy, the operation returns a typed insufficient-evidence outcome
with diagnostics; it must not silently return coarse boxes as refined output.

## Extension points

- `FrameCloudStore`: immutable, shareable full-frame point evidence.
- `TrackRefiner`: single-instance geometry and pose refinement.
- `DatasetAdapter`: source-specific construction of frame clouds, detections,
  and frame poses.
- `ResultAdapter`: translation into a caller's box/annotation representation.
- `Evaluator`: development-only comparison with physically separate targets.

## Algorithm boundary

The public framework remains backend-neutral. The accepted first production
direction is the deterministic, non-learned
[Component-Consensus Geometric Refinement V2](geometric-refinement-v2.md). It
extracts one object component per usable frame, prevents weak frames from
defining canonical geometry, performs track-anchored aggregation, estimates
one size, and then refines every frame pose while holding that size fixed.

The existing `JointCuboidRefiner` implementation is the rejected V1
experimental baseline. It performs coarse-box evidence selection, synchronous
cross-frame registration, canonical aggregation, and alternating envelope
fitting, but deliberately returns `insufficient_evidence`. Real review showed
that its local residual could improve while the aggregate geometry regressed.
It remains available only to reproduce that failure and compare V2 against a
frozen baseline; it is not an in-progress production backend.

V2 keeps the public full-frame input and success/insufficient outcome. It does
not use sensor-origin rays, free-space, occupancy, learned priors, or
per-sensor processing. Optional portable metadata remains in the contract for
other backends and diagnostics but is not consumed by V2.

Future experimental backends may use a different implementation without
changing the public full-frame input or validated success/insufficient outcome.
They must be named explicitly and must not weaken the public success semantics.

## Integration direction

MMDetection3D may call TrackRefinery once per associated track while reusing its
already loaded full-frame point tensors. X-4D continues to supply and receive a
complete Clip through the existing preannotation protocol. The core package
does not import either project and never introduces a second X-4D service
protocol.

The optional `trackrefinery.adapters.x4d` development adapter may depend on the
published Devkit. It resolves every metadata-declared current-keyframe LiDAR
channel, transforms points into the Clip's annotation frame, preserves exact
point time and sensor provenance, and freezes the result into the portable
source-only inference layout. The export process may read a frozen candidate
or a review-only annotation reference to construct coarse track files, but the
algorithm process opens only the exported inference root. This two-step path
keeps native annotations and evaluation targets out of the refiner even when a
locally synchronized Clip physically contains them.
