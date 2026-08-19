# Component-Consensus Geometric Refinement V2

Status: accepted replacement design; component extraction, frame roles, and
provisional anchored geometry-frame aggregation implemented

## Decision

V2 replaces the V1 all-frame registration and alternating-envelope design.
It remains deterministic and non-learned, but deliberately solves a smaller
problem:

```text
full frame clouds + one associated coarse track + exact frame poses
  -> extract one object component in each usable frame
  -> align only reliable components into a sharp aggregate
  -> fit one canonical size from reliable aggregate evidence
  -> keep that size fixed while refining every frame pose
  -> accept only a demonstrable same-instance improvement
```

The backend does not perform scene understanding. It receives a track that has
already been detected and associated, and uses that track as a bounded spatial
seed. It must not turn refinement into another detector, tracker, occupancy
model, or sensor simulation problem.

The V1 implementation remains available temporarily as a trace-only regression
baseline. It is not the foundation for V2 and must not be extended by adding
more thresholds to its unanchored all-frame ICP loop.

## Output invariant and success meaning

The public contract does not change:

- exactly one canonical `(length, width, height)` for the rigid instance;
- one refined pose for every input observation;
- the exact same canonical dimensions materialized in every output frame;
- an explicit `insufficient_evidence` result when the claimed accuracy is not
  supported.

For this project, `success` means that the result is suitable for direct
preannotation review: a reviewer should not need to correct its size or any
released frame pose. It does not mean that an internal residual decreased.

## Explicit non-goals

V2 does not use or require:

- sensor-origin rays, ray casting, free-space, occupancy, or visibility
  simulation;
- a neural network, trained weights, CAD template, or category dimension
  prior;
- a caller-provided crop or crop margin;
- per-sensor processing after the caller has fused points into the annotation
  frame;
- track association, ID repair, head/tail extension, or new observations;
- a monolithic optimizer that moves all frames and the size at the same time.

Point timestamps and sensor provenance remain legal portable input metadata,
but the V2 backend does not consume them. Frame timestamps and exact
annotation-frame poses are required and are consumed.

## Why V1 is rejected

The implemented V1 candidate uses expanded coarse-box membership as initial
target ownership, independently initializes frames with PCA, synchronously
registers every frame against a mixed aggregate with point-to-plane residuals,
retains points with weak cross-frame proximity, and then fits quantile tails.
This objective can lower local surface residuals by aligning ground,
background, or partial vehicle planes while making the physical vehicle
envelope worse.

The same-instance real-data comparison exposed exactly that failure: the
registration candidate reduced per-frame centroid XY RMS while increasing the
aggregate width and XY envelope area. Internal ICP RMSE was therefore
anti-correlated with the product objective. No parameter calibration can make
that objective a reliable release criterion.

V2 changes the decomposition instead of tuning V1:

1. determine the object component before registration;
2. prevent weak frames from defining canonical geometry;
3. use anchored aggregation rather than simultaneous gauge-free movement;
4. estimate size once, before per-frame pose refinement;
5. keep size fixed during every pose update;
6. compare the candidate against the frozen coarse baseline using the same
   points and reject regressions.

## Coordinates and state

For frame `i`:

- `P_i` is the full fused cloud in annotation frame `A_i`;
- `W_i = T_world_from_annotation[i]` is the exact input pose;
- `C_i = T_annotation_from_coarse_object[i]` is the coarse box pose;
- `B_i^0 = W_i C_i` is the coarse object pose in the shared world frame;
- `S_i` is the selected non-ground object component;
- `B_i` is the refined object pose in the shared world frame.

The instance state is one canonical size `s = (l, w, h)` and the ordered poses
`B_1 ... B_N`. Internally V2 optimizes upright translation and yaw. Ground
support determines vertical placement and roll/pitch handling; it is not mixed
into horizontal component registration.

Exact timestamps are used in trajectory residuals. The implementation must not
assume a fixed Clip rate. Final poses are transformed back into each frame's
declared annotation frame.

## Stage 1: bounded ROI, ground removal, and component extraction

The library owns a finite ROI around every coarse box. The ROI is only a search
boundary; points are never declared to be object evidence merely because they
fall inside the box or its margin.

Within the ROI, V2:

1. estimates a robust local ground plane and removes its supported points;
2. voxelizes the remaining points at a deterministic metric resolution;
3. builds spatially connected components;
4. retains component candidates that intersect a conservative inner seed
   around the coarse object location;
5. selects one component using deterministic coarse-track consistency, center
   proximity, vertical extent, component dominance, and neighboring-component
   separation.

The detector box is a localization seed, not a target mask and not a dimension
target. A candidate that cannot be separated from a neighbor remains
ambiguous. V2 returns insufficient evidence rather than merging components or
allowing all ROI points to enter registration.

For the dense-first profile, a selected connected component also has to remain
inside a deliberately loose coarse-box envelope. The current envelope adds
`(0.35, 0.25, 0.25)` metres to the box half-extents in local length, width, and
height, and permits at most 2% of selected points outside it. Exceeding that
limit does not crop away the inconvenient points: the entire frame becomes
`component_not_separable` with `component_exits_coarse_envelope`. This catches
dense components connected to a wall, curb, or neighboring object without
pretending the retained interior is known to be pure.

This stage produces a selected point index set and component diagnostics for
each frame. Those exact indices are preserved through baseline/candidate visual
comparison so an apparent improvement cannot be manufactured by changing the
displayed points.

## Stage 2: frame reliability

Frames do not have equal authority. Each selected component is assigned one of
three deterministic roles:

- `geometry`: sufficiently dense and spatially distributed to participate in
  canonical aggregation and size estimation;
- `pose_only`: sufficient for alignment against a fixed canonical shape, but
  not allowed to affect size;
- `trajectory_only`: insufficient for geometric alignment; a pose may only be
  interpolated from reliable neighboring frames with bounded uncertainty.

Reliability uses only measurable properties of the selected component and the
coarse track: point count, occupied-voxel count, horizontal and vertical
spread, component dominance in the ROI, separation from the nearest competing
component, and stability under deterministic voxel resolutions. Detector score
may be reported but is not sufficient to grant a geometry role.

A track needs multiple geometry frames with complementary spatial support.
Low-quality frames cannot vote on canonical dimensions. If any required output
frame is neither geometrically refinable nor safely bracketed by reliable
trajectory estimates, the whole result is insufficient evidence.

### Dense-first MVP scope

The first implemented profile intentionally supports point-rich instances
before sparse targets. A `geometry` frame must pass both absolute evidence
requirements and same-track relative requirements. The relative reference is
the configured upper quantile of selected-component point counts and robust
axis spreads from that one track; it is not a category-size prior.

The current default profile requires at least 1,000 selected points, 100
occupied component voxels, at least 20% of the track's 80th-percentile point
support, at least 65% of its 80th-percentile spread on every axis, and five
qualifying geometry frames in the track. Frames that have a valid component
but miss this geometry gate remain `pose_only` when their evidence permits.
Tracks below the five-frame gate report `dense_track_out_of_scope` and do not
advance to canonical aggregation during this MVP.

This is a development-scope gate, not a claim that 999 points are physically
insufficient. The thresholds are versioned in
`trackrefinery-component-consensus-settings-v3`; sparse-track support will be a
separate calibration and implementation task after dense-track refinement is
usable.

## Stage 3: anchored component aggregation

V2 aggregates only selected components from geometry frames. It never aligns
raw ROI points.

The highest-quality temporally central geometry frame establishes the initial
canonical coordinate gauge. Other geometry frames are added in deterministic
quality order. Each candidate frame is aligned to the fixed aggregate built
from already accepted frames, not while all existing frame poses move at once.

Alignment may use a robust, trimmed point/voxel correspondence implementation,
but it is constrained by:

- a bounded correction from the coarse world trajectory;
- timestamp-aware velocity, acceleration, and yaw-continuity residuals;
- a minimum overlap/support requirement with the accepted aggregate;
- explicit rejection when the correction is underconstrained or produces a
  less sharp same-point aggregate.

The aggregate is updated only after a frame alignment is accepted. Rejected
frames cannot deform the canonical shape and are downgraded to `pose_only` or
`trajectory_only`.

This is not a claim that ICP is the algorithm. Nearest-neighbor registration is
an interchangeable local alignment primitive inside a track-anchored process.
Its residual is diagnostic, never the definition of success.

### Implemented Stage 3 profile

The first implementation uses a deterministic trimmed point-to-point local
primitive over 8 cm voxel representatives. It optimizes only horizontal
translation and yaw; it never changes vertical placement or any accepted
frame. The anchor is the temporally most central frame among frames within 80%
of the best measured component quality. Remaining geometry frames are attempted
in deterministic quality order against the fixed aggregate accumulated so far.

Every proposed correction is bounded to 0.25 m and 4 degrees. A material
correction must improve both absolute and relative trimmed RMSE, must not widen
any 1--99% aggregate axis by more than 2 cm, and must not reduce multi-frame
voxel concentration by more than 0.005. A frame whose baseline already has
sufficient overlap may retain its exact coarse pose when the proposed movement
is unnecessary or fails those non-regression checks. Insufficiently overlapping
frames are rejected rather than allowed to deform the aggregate.

After sequential aggregation, a track-level check repeats the same-axis and
voxel-concentration comparison and verifies timestamp-aware correction velocity,
acceleration, and yaw-rate bounds using the exact annotation-frame-to-world
poses. A track-level regression discards all proposed movement and retains the
coarse alignment. These thresholds are versioned development calibration, not
physical claims or public caller parameters.

The Stage 3 output is still a provisional trace and always returns
`insufficient_evidence`: no canonical dimensions have been fitted and no
annotation result is released. Review A/B views use only identical selected
component points from geometry frames; ROI background and ground cannot make a
registration candidate look better or worse.

## Stage 4: one canonical size

Canonical size is estimated exactly once from the accepted geometry-frame
aggregate. Pose-only and trajectory-only frames cannot change it.

The first implementation uses deterministic robust envelope fitting in the
canonical axes:

- horizontal limits come from persistently supported aggregate tails rather
  than raw extrema;
- bottom placement comes from consistent local ground support;
- height comes from supported upper evidence across geometry frames;
- isolated voxels and evidence unique to one frame are excluded;
- dimensions are checked by leaving out each geometry frame in turn.

No method without a learned prior can recover a completely unobserved physical
side. Therefore a dimension is accepted only when complementary observations
and the leave-one-frame-out fits constrain it within the versioned tolerance.
Otherwise the result is `unobservable_length`, `unobservable_width`, or
`unobservable_height` rather than an inferred default.

The fitted cuboid may introduce one common canonical-center offset. That offset
is applied consistently to all frame poses; it is not an opportunity to apply
independent per-frame size corrections.

## Stage 5: fixed-size per-frame pose refinement

After Stage 4, canonical dimensions are immutable. Every geometry or pose-only
frame is registered against the fixed canonical component/shape using its
selected points, with the coarse world trajectory and timestamp-aware temporal
terms as constraints.

The pose objective is conceptually:

```text
selected-component alignment to fixed canonical shape
+ bounded correction from coarse world pose
+ velocity/acceleration/yaw continuity
```

It contains no dimension variable. A difficult frame can no longer improve its
local residual by widening the instance.

Trajectory-only frames use interpolation only when surrounded by reliable
poses and when the propagated uncertainty stays below the output tolerance.
They do not alter the canonical shape, size, or reliable neighboring poses.

## Candidate acceptance and regression protection

V2 does not publish a result merely because every numerical stage returned a
value. A candidate must pass all of the following:

- one component is separable in every geometry and pose-only frame;
- enough geometry frames were accepted to constrain all three dimensions;
- canonical dimensions are stable under geometry-frame leave-one-out;
- every per-frame correction is bounded and the refined world trajectory is
  temporally consistent;
- selected-point registration residuals improve on the coarse baseline;
- same-point aggregate sharpness improves and no horizontal axis becomes more
  smeared;
- the fixed-size invariant holds exactly in all materialized boxes;
- no required frame is unobservable or extrapolated beyond its tolerance;
- all values are finite and the run is deterministic.

Baseline comparison always uses identical selected point indices, colors,
plot axes, and frame set. Aggregate sharpness must include axis-wise spread and
voxel concentration; one scalar nearest-neighbor or ICP RMSE is not sufficient.

With reviewed targets, the stronger evaluation contract still applies:
canonical dimension errors and every-frame pose errors must pass the agreed
correction tolerances. Without reviewed targets, the backend may demonstrate
non-regression and produce a development candidate, but it cannot be declared
release-qualified from qualitative aggregate sharpness alone.

Initial stable rejection families are:

```text
unsupported_object_geometry
ground_support_unavailable
component_not_separable:<frame_id>
insufficient_component_points:<frame_id>
insufficient_geometry_frames
component_alignment_failed:<frame_id>
trajectory_unobservable:<frame_id>
unobservable_length
unobservable_width
unobservable_height
unstable_canonical_size
aggregate_regression
trajectory_regression
optimization_not_converged
```

Thresholds are versioned calibration data and must be chosen on reviewed
development/calibration cases, never by inspecting locked test targets.

## Diagnostics and review artifacts

Every run records:

- selected component and rejected component indices per frame;
- frame role and the measurements that produced it;
- accepted aggregation order and per-frame alignment correction;
- coarse-versus-candidate trajectory residuals;
- canonical dimensions and leave-one-frame-out estimates;
- axis-wise same-point aggregate spread and voxel concentration before/after;
- fixed-size per-frame pose results;
- final acceptance or rejection reasons.

The existing Clip catalog remains the primary qualitative surface. For each
algorithm instance it must show the same selected points aligned by coarse,
candidate, and reviewed target poses where available. The default card should
lead with coarse-versus-candidate top and side views, frame roles, canonical
dimensions, and the exact rejection reason.

## Implementation sequence

1. **Implemented:** preserve V1 as a named legacy trace baseline and prevent it
   from being mistaken for the current backend.
2. **Partly implemented:** add V2 trace contracts. Component decisions, frame
   roles, anchored alignment decisions, and aggregate sharpness are present;
   dimension-stability fields arrive with their owning stage.
3. **Implemented, pending real-catalog review:** deterministic ground removal,
   3D component selection, resolution-stability measurement, and conservative
   merged-component rejection.
4. **Implemented, pending five-instance real review:** frame-role
   classification and anchored geometry-frame aggregation, retaining baseline
   poses when a candidate alignment regresses.
5. Implement canonical size fitting and leave-one-frame-out stability.
6. Implement fixed-size per-frame pose refinement and bounded trajectory-only
   interpolation.
7. Calibrate rejection thresholds on reviewed development/calibration tracks.
8. Run frozen blind evaluation and X-Points correction-time review before any
   MMDetection3D integration is called usable.

Each implementation stage must produce a reviewable real-data artifact. Clean
synthetic recovery is a regression test, not evidence that the stage is useful.
