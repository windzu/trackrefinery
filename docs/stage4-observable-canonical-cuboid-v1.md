# Stage 4: Observable Canonical Cuboid

Status: experimental implementation complete; reviewed-target calibration and
promotion pending

## Decision

Stage 4 estimates one common canonical frame and one canonical cuboid only
after Stage 3 has produced an observable, accepted relative alignment:

```text
accepted Stage 3 geometry poses and frozen selected components
  -> common upright axis estimate from repeated local surface geometry
  -> per-frame robust outer frontiers
  -> cross-frame supported opposing boundaries
  -> ground-supported bottom and observed upper boundary
  -> leave-one-frame-out and resolution stability checks
  -> canonical size plus one common pose transform, or insufficient evidence
```

The estimator is deterministic and non-learned. It does not use a category
dimension prior, detector box dimensions, annotations, CAD templates, sensor
rays, free-space, or an occupancy model. The selected components are already
the target evidence delivered by Stages 1--3; Stage 4 solves only the common
axis, center, and physical envelope that this evidence actually supports.

The public V2 refiner remains stage-gated. The first implementation is an
explicit experiment API and sidecar, and must not publish a
`RefinementSuccess` until the sizing and later fixed-shape pose gates pass.

## Input boundary

Stage 4 consumes the original `RefinementCase` and one matching Stage 3 trace.
The trace must have:

- a candidate anchored aggregation;
- at least the configured number of geometry frames;
- a finite provisional pose for every participating geometry frame;
- the exact frozen selected-component indices from Stage 2;
- a persistent canonical shape from Stage 3.

Stage 4 must reject a disconnected, retained-coarse-on-regression, or otherwise
insufficient Stage 3 run. It does not rerun association, component selection,
or pairwise registration, and it cannot make an unobservable Stage 3 result
observable by fitting a box around it.

Annotation targets remain evaluator-only. A source annotation or frozen model
track may be displayed as an explicitly non-gold reference, but it is never an
algorithm input and is not used to tune one instance at runtime.

## Canonical-frame state

Let `p_r` be a selected point in the provisional Stage 3 registration frame.
Stage 4 estimates one upright transform `T_r_from_c` from the final canonical
cuboid frame to that registration frame:

```text
p_c = inverse(T_r_from_c) p_r
```

`T_r_from_c` contains a common XY center offset, a ground/top-derived Z center,
and a common yaw. It has no roll or pitch. The final cuboid is centered at the
origin of `c`, is axis aligned there, and has one `(length, width, height)`.

For every Stage 3 participating frame, the Stage 4 pose candidate is:

```text
T_annotation_from_c = T_annotation_from_r T_r_from_c
```

Thus Stage 4 applies exactly one rigid common-gauge change. It cannot alter
relative frame-to-frame alignment or assign different dimensions per frame.
Stage 5 may later refine every frame pose against the fixed cuboid/shape, but it
must keep the Stage 4 dimensions byte-identical.

## Common upright axes

Minimum-area rectangles and point-cloud PCA are not accepted axis estimators.
Both can rotate toward a partial roof, one long visible side, or uneven point
density while reducing their scalar objective.

The experimental estimator instead uses local surface normals from the
persistent Stage 3 shape:

1. estimate deterministic local covariance normals at the frozen voxel
   resolution;
2. keep low-variation, near-horizontal normals belonging to approximately
   vertical object surfaces;
3. treat normal sign as irrelevant and combine orientations modulo 90 degrees;
4. weight repeated geometry by frame-support count;
5. choose the axis hypothesis nearest the Stage 3 gauge, within the configured
   common-yaw correction bound.

The fourfold circular coherence of the retained normals is the yaw
observability metric. Insufficient normal support, weak coherence, a correction
at the configured bound, or instability under leave-one-frame-out and voxel
resolution changes returns insufficient evidence. The coarse box supplies the
nearby axis convention only; its dimensions and center are not fit priors.

## Cross-frame supported boundaries

One aggregate percentile is not a physical boundary. Its meaning changes with
frame point count, duplicated views, range, and the number of accumulated
frames. Raw minimum/maximum values are even less safe.

For each participating frame and each candidate canonical axis, Stage 4 first
computes lower and upper robust frontiers from that frame alone. The tail uses
a versioned minimum point count with a capped fractional fallback, so neither a
very dense frame nor a sparse frame silently owns the estimate.

The track boundary is then the outer frontier supported by at least the
versioned minimum number and fraction of geometry frames. A frame supports a
face only when the boundary band contains enough low-variation local surface
points, their sign-invariant normals agree with the candidate face normal, and
the points have measurable tangential span. An isolated return or the cut end
of a different surface cannot establish a face merely because it is extreme.

Length and width require supported opposing boundaries. Stage 4 never mirrors
one visible side, substitutes a detector dimension, or applies a category
default to invent the missing side.

## Height and ground

The component stage deliberately removes ground, so the lowest retained object
return is not the physical cuboid bottom. Stage 4 transforms every accepted
frame's already-estimated ground plane into the Stage 3 object frame and uses a
robust cross-frame consensus for the bottom boundary. It uses no ray tracing.

The upper boundary is estimated from per-frame observed upper frontiers using
the same repeated-support rule as the horizontal faces. Height is observable
only when:

- enough accepted frames contain valid ground estimates;
- transformed ground heights agree within the versioned stability limit;
- enough frames support the upper frontier;
- the resulting height is positive and stable under leave-one-frame-out and
  resolution perturbation.

## Stability and rejection

Every candidate is refit after removing each geometry frame in turn. It is also
refit at the configured neighboring voxel resolutions. The sidecar records the
maximum change in size, center, and common yaw across both families.

A candidate is rejected when any of the following holds:

- Stage 3 is not an accepted observable candidate;
- fewer than the minimum geometry frames participate;
- common yaw lacks coherent surface-normal support or reaches its bound;
- any opposing horizontal face lacks repeated, spatially extended support;
- ground or upper-face support is missing;
- any dimension, center, or yaw is non-finite or non-positive;
- leave-one-frame-out size, center, or yaw exceeds its stability limit;
- neighboring-resolution size or yaw exceeds its stability limit;
- the candidate correction exceeds the independent center/yaw safety bounds.

An insufficient result retains a provisional estimate only in the development
sidecar for diagnosis. It must not materialize that estimate as a cuboid
candidate or resemble a released annotation in review output.

## Experimental output and trace

The first API returns:

- a geometric trace expressed in the final centered canonical frame when the
  candidate passes, or an insufficient cuboid trace when it does not;
- one sidecar with the exact experiment settings and status;
- `T_r_from_c`, canonical size, lower/upper boundaries, and common yaw for an
  accepted candidate;
- per-face supporting frame/point counts and tangential-span diagnostics;
- normal count, coherence, and yaw hypotheses;
- ground-frame count and ground-height dispersion;
- leave-one-frame-out and resolution-perturbation summaries;
- stable reason codes for every rejection.

The accepted candidate is still not a public refinement success. The sidecar
is separate from the stable evidence-trace contract while thresholds and
diagnostic fields are being calibrated.

## Evaluation and review

Synthetic evaluation isolates Stage 4 by supplying exact or controlled-good
relative poses in the evaluator. It must cover:

- biased common center, yaw, and detector dimensions;
- all six supported faces and a sloped but consistent ground plane;
- one missing horizontal face, which must be rejected;
- an isolated boundary outlier, which must not change the fitted cuboid;
- a single critical viewpoint whose removal changes a dimension, which must be
  rejected by leave-one-frame-out stability;
- point-order and world-gauge invariance.

Real development review uses the frozen five-track dense slice from
`20260817_G150-002_000`. All five remain visible. The four Stage 3 observable
tracks receive Stage 4 candidates or explicit Stage 4 rejection; the weak
Stage 3 bridge remains rejected before sizing. The page displays, for the same
selected points:

```text
BEFORE  Stage 3 aggregate with frozen model-track dimensions
AFTER   centered canonical aggregate with the Stage 4 cuboid, when accepted
```

Each card shows input/candidate dimensions, common center/yaw change, face
support, stability maxima, and rejection reasons. This Clip has no reviewed
geometry target, so those results are qualitative and must not be reported as
accuracy.

Promotion requires a physically separate reviewed target set. The primary
metric remains precision among reported successes: a successful cuboid must
require no human size correction. Coverage is secondary, and catastrophic
success is a release blocker. Final tolerances are set with the annotation
owner after blinded X-Points review; they are not inferred from the current
model boxes.

## First implementation result

`fit_observable_canonical_cuboid()` implements this contract as the explicit
`observable_canonical_cuboid_v4_experiment` stage. Every numeric gate is an
immutable `CanonicalCuboidExperimentSettings` field; the implementation has no
category table, model-dimension fallback, or hidden dense/sparse branch. The
dense-first scope comes from the preceding component/Stage 3 gate.

The exact-alignment synthetic fixture recovers all three canonical dimensions
within 6 cm and deterministically rejects the fixture with a removed length
face. The review bundle serializes the complete experiment sidecar and shows
input-to-fit dimensions, common center/yaw correction, six boundary-support
counts, stability maxima, and reason codes.

On the frozen five-track dense slice from `20260817_G150-002_000`, four tracks
reach Stage 4 after Stage 3. One track supports every required boundary and
produces a `6.135 x 2.489 x 3.055 m` experimental candidate from a
`6.338 x 2.704 x 3.183 m` model box. Its maximum leave-one-frame-out dimension
change is 1.6 cm and neighboring-voxel-resolution change is 1.8 cm. The other
three tracks are rejected for missing repeated support on one or more opposing
length/width faces. The fifth remains rejected by Stage 3's weak-bridge gate.

This result establishes observability-aware behavior, not metric accuracy. The
Clip has no physically separate reviewed target, so neither the accepted
candidate nor the rejection coverage qualifies the estimator for release.
