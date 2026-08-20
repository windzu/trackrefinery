# Architecture

## Goal and unit of work

TrackRefinery refines one already-associated rigid-object track. Its input is
the sequence of full scene point clouds containing that object, the object's
coarse detection in each input frame, and exact frame poses. Its output is one
canonical `(length, width, height)` plus authoritative refined object poses for
the supported input observations.

The public algorithm unit is one instance track. A batch runner may share the
same immutable frame-cloud objects across many tracks, but that is an execution
optimization rather than a change to the refinement contract.

```text
upstream detector and tracker
  -> full frame clouds + one associated detection track + frame poses
  -> TrackRefinery extracts one component per usable frame
  -> observable-core aggregation and one canonical size
  -> fixed-size pose refinement on supported frames
  -> canonical size + authoritative poses + unsupported-frame reasons
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

Each full or partial successful rigid-instance result has one canonical size
shared exactly by all authoritative refined frame poses. A full success covers
every input frame. A partial success exactly partitions the input into
authoritative and unsupported frames; coarse poses may remain a caller-side
display fallback but are never TrackRefinery output. A future articulated-object
extension must be explicit rather than weakening this invariant. If the
evidence cannot support any useful core at the required accuracy, the operation
returns a typed insufficient-evidence outcome with diagnostics.

## Extension points

- `FrameCloudStore`: immutable, shareable full-frame point evidence.
- `TrackRefiner`: single-instance geometry and pose refinement.
- `DatasetAdapter`: source-specific construction of frame clouds, detections,
  and frame poses.
- `ResultAdapter`: translation into a caller's box/annotation representation.
- `Evaluator`: development-only comparison with physically separate targets.

## Algorithm boundary

The public framework remains backend-neutral. Real review showed that claiming
whole-track corrections mixes strong central evidence with sparse, weak tails.
The accepted MVP is therefore the deterministic
[Observable-Core Refinement V1](observable-core-refinement-v1.md): estimate a
canonical size from a connected reliable subset, refine only supported poses,
and make every unsupported frame explicit. Direct trajectory
`delta_log_lwh` regression from the superseded
[learned V1 plan](learned-refinement-plan-v1.md) remains only a negative-control
baseline.

The representation-first
[object-centric foundation exploration](object-centric-foundation-exploration-v1.md)
is deferred to coverage expansion for incomplete or unobservable cases. It is
not on the deterministic MVP critical path.

[Component-Consensus Geometric Refinement V2](geometric-refinement-v2.md) and
its V3/V4 experiments remain frozen comparison and diagnostic backends. They
do not silently act as a successful fallback for a learned backend.

The existing `JointCuboidRefiner` implementation is the rejected V1
experimental baseline. It performs coarse-box evidence selection, synchronous
cross-frame registration, canonical aggregation, and alternating envelope
fitting, but deliberately returns `insufficient_evidence`. Real review showed
that its local residual could improve while the aggregate geometry regressed.
It remains available only to reproduce that failure and compare V2 against a
frozen baseline; it is not an in-progress production backend.

V2 keeps the public full-frame input and validated full/partial/insufficient
outcome. It does
not use sensor-origin rays, free-space, occupancy, learned priors, or
per-sensor processing. Optional portable metadata remains in the contract for
other backends and diagnostics but is not consumed by V2.

The V2 implementation currently stops after anchored aggregation of reliable
geometry-frame components. Overwide components that likely merge a neighbor or
clutter remain ambiguous rather than becoming object evidence. Candidate
alignment is bounded, sequential, and individually rejectable; an unnecessary
or regressing correction retains the exact coarse pose. The backend is
stage-gated and cannot return success until canonical sizing, fixed-size pose
refinement, and final acceptance are implemented.

Before sizing, the anchored stage must pass a separate known-error recovery
gate. That evaluator freezes selected components, runs the candidate on the
unperturbed real model track, and injects deterministic non-anchor XY/yaw drift
around that natural output. The perturbed run receives only its perturbed
coarse poses. Reports retain both equivariant recovery to the unperturbed
algorithm output and absolute drift to the original non-gold model-track
proxy. This is an isolated test of Stage 3 registration, not an alternate
runtime input contract, annotation target, or end-to-end crop test.

The accepted Stage 3 redesign is documented in
[`stage3-pose-graph-v3.md`](stage3-pose-graph-v3.md). It separates observable
pairwise local measurements from one robust global pose-graph solve, uses exact
timestamps and frame poses for trajectory consistency, and keeps graph
diagnostics in an experimental sidecar until the promotion gate passes.

The experimental Stage 4 sizing implementation is documented in
[`stage4-observable-canonical-cuboid-v1.md`](stage4-observable-canonical-cuboid-v1.md).
It consumes only an accepted Stage 3 trace, estimates common upright axes and
cross-frame-supported physical boundaries, and rejects missing opposing faces
instead of substituting model/category dimensions. It applies one common
center/yaw gauge transform to all participating poses. The API and review
sidecar are implemented, but it still cannot publish `RefinementSuccess`;
reviewed-target calibration and fixed-shape per-frame pose refinement remain
gated work.

The current production-path implementation is `ObservableCoreRefiner`. It
reuses V2 component measurements, forms maximal adjacent geometry-frame runs,
breaks runs at relative timestamp discontinuities, deterministically chooses
one strongest run, and downgrades disconnected geometry frames to pose
candidates before aggregation. Its trace stages are
`observable_core_selection_v1` and `observable_core_aggregation_v1`. It cannot
publish a full or partial success until canonical-size stability and fixed-size
pose gates are implemented.

Future experimental backends may use a different implementation without
changing the public full-frame input or validated full/partial/insufficient outcome.
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
