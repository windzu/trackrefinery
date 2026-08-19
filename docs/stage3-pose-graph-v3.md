# Stage 3 V3: Observable Pairwise Registration and Pose Graph

Status: experimental implementation complete; four dense tracks pass controlled
recovery and one is rejected as graph-underconstrained; default promotion pending

## Decision

The next Stage 3 candidate replaces order-dependent frame-to-growing-aggregate
registration with two explicit layers:

```text
frozen selected geometry components
  -> observable pairwise SE(2) measurements
  -> one robust anchored pose-graph solve
  -> same-point aggregate and non-regression checks
```

This is a redesign of relative multi-frame alignment, not another round of
threshold tuning on the sequential V2.1 implementation. The existing anchored
implementation remains frozen as the experiment baseline until V3 passes the
controlled-recovery and natural-input gates.

V3 remains deterministic and non-learned. It does not use sensor rays,
free-space, occupancy, category dimensions, a CAD model, or annotation targets.
It consumes the exact selected components, coarse observations, timestamps,
and annotation-frame-to-world poses already present in the accepted input and
Stage 2 trace.

## Why the sequential candidate is insufficient

The first controlled suite shows that V2.1 improves translation RMS in 13 of
15 case/profile runs but yaw RMS in only 9 of 15. A strong-profile track
regresses in yaw. The local point-to-point residual can rotate a long vehicle
side or slide along it while still reducing nearest-neighbor distance.

Sequential aggregation compounds that ambiguity:

- one partial surface is registered against a reference whose composition
  depends on earlier frame order;
- an early biased correction changes every later local objective;
- one-way nearest-neighbor correspondences permit partial-surface collapse;
- a scalar residual does not state which of `x`, `y`, or `yaw` was observable;
- timestamp-aware trajectory terms currently reject only after registration;
  they do not help reconcile competing measurements.

The failure is therefore structural. Lowering the step size or changing a trim
fraction cannot make an unobservable yaw measurement trustworthy.

## State and gauge

For geometry frame `i`, let `p_i` be a selected point expressed in the input
coarse-object frame and let `X_i` be the upright correction from that frame to
the shared canonical gauge:

```text
q = X_i p_i
```

`X_i` contains only `(x, y, yaw)`. The chosen anchor has `X_anchor = identity`.
The candidate world object pose remains compatible with the existing Stage 3
contract:

```text
B_i_candidate = B_i_coarse inverse(X_i)
```

where `B_i_coarse` is constructed using the exact input
annotation-frame-to-world pose. Z, roll, and pitch are unchanged in Stage 3.

The anchor removes the global gauge only. A center or yaw bias shared by every
input frame is not recoverable from relative registration. Stage 4 owns the
common canonical center, orientation, and size; Stage 5 then refines each frame
against that fixed shape.

## Graph construction

Every geometry frame is one node. Candidate edges are deterministic and
bounded in number:

1. connect consecutive geometry frames in timestamp order;
2. connect the next configurable temporal neighbors on each side;
3. optionally connect nonlocal keyframes only when their identity-initialized
   selected components have enough symmetric overlap;
4. never build a complete all-pairs graph merely because a track is long.

An edge is retained only when it contains measurable geometric information.
The retained graph must connect every participating geometry frame to the
anchor. A disconnected frame loses geometry authority; if fewer than the
required number remain, the track returns insufficient evidence.
The first experimental implementation is intentionally stricter and rejects
the whole Stage 3 candidate when any geometry frame is disconnected, so a
batch report cannot silently omit a difficult frame from its denominator.

Connectivity alone is not observability. Removing each retained edge identifies
graph bridges. A bridge with less than the versioned majority-overlap threshold
cannot be the sole authority for the relative pose of two frame clusters; the
track returns `weak_partial_overlap_bridge`. This prevents a locally attractive
partial-view yaw from moving an otherwise self-consistent cluster without a
geometric cycle that can contradict it.

The graph builder does not assume a frame rate. Temporal neighborhoods use
ordered observations, while all trajectory residuals use exact integer
nanosecond timestamps.

## Pairwise local measurement

For edge `(i, j)`, the local estimator measures `M_j_from_i`, mapping points in
coarse frame `i` into coarse frame `j`. The graph consistency relation is:

```text
X_i = X_j M_j_from_i
```

The accepted normal-aware direction is an upright local-registration front end:

- deterministic voxel representatives from the frozen Stage 2 component trace;
- local covariance normals estimated from neighboring selected-component
  points, with no sensor-origin orientation requirement;
- maximum metric distance and sign-invariant normal compatibility gates;
- symmetric source-to-target and target-to-source residuals;
- mutual-nearest-neighbor preference so a partial surface cannot collapse onto
  a small target patch;
- robust loss and trimmed residual support;
- bounded hypotheses around the coarse initialization, including a small yaw
  grid to avoid one local minimum deciding the edge.

The first implementation uses one frozen voxel resolution and symmetric
point-to-plane residuals. A coarse-to-fine schedule and Generalized ICP remain
interchangeable front-end experiments, not public contracts and not the global
solution by themselves.

### Observability

Each accepted correspondence contributes a Jacobian in scaled `(x, y, yaw)`
coordinates. Yaw is scaled by the component's robust horizontal radius so a
radian represents comparable surface displacement rather than an arbitrary
unit.

The weighted normal matrix is eigen-decomposed. Eigen-directions below the
versioned information threshold are set to zero instead of being promoted into
false precision. An edge may therefore constrain only two directions, for
example lateral translation and yaw on a long side. Complementary views may
supply the missing direction in the global graph.

An edge is rejected when it has insufficient correspondence support or
symmetric overlap, no observable direction, an excessive bounded correction,
or a non-improving symmetric residual. Rank, eigenvalues, condition, overlap,
residuals, and the chosen initialization hypothesis are recorded.

## Global pose-graph objective

The anchor is fixed and all other node corrections are solved together. The
objective contains four families:

```text
observable pairwise edge residuals
+ weak bounded coarse-correction priors
+ timestamp-aware candidate-world acceleration/yaw-acceleration residuals
+ robust loss on every non-anchor term
```

Pairwise residuals use each edge's positive-semidefinite information matrix;
unobservable directions contribute zero weight. Priors prevent a weakly
connected graph from drifting to correction limits but must not override
well-supported geometric edges.

Trajectory residuals are computed after converting candidate object poses into
the shared world frame using each exact input frame pose. Finite differences use
the exact timestamps and softly penalize candidate linear acceleration and yaw
acceleration. Constant linear velocity and constant yaw rate therefore incur no
penalty; nonconstant motion remains possible through the robust soft term. The
objective does not assume a stationary object or fixed Clip rate. Experiments
also show that trajectory plausibility cannot resolve a multi-second gap when
two cluster yaws are both physically smooth, so it never substitutes for the
bridge observability gate.

The solve uses deterministic initialization, bounds, ordering, and robust
least squares. It must return the same result and diagnostics for identical
inputs. A failed convergence, disconnected graph, bound saturation, or
trajectory violation returns insufficient evidence.

## Aggregate and acceptance

Only after the graph solve are selected geometry components transformed by the
node corrections and aggregated. The candidate is compared with the frozen
coarse baseline using identical point indices and frame colors.

Stage 3 remains provisional. It retains or rejects candidate corrections and
never publishes canonical dimensions. Acceptance requires:

- connected observable graph and deterministic convergence;
- bounded per-node translation/yaw corrections;
- timestamp-aware trajectory limits;
- no resolution-significant robust-axis expansion, no footprint-area or
  voxel-concentration regression, and no cross-frame residual regression;
- improved symmetric cross-frame residual;
- no frame whose local evidence is contradicted by most incident graph edges.

The per-axis guard is resolution-aware. A sub-voxel change in one axis may be
the geometric consequence of removing rotational smear and is not rejected
when the total XY footprint, cross-frame residual, and concentration all
improve. The allowance is derived from half the configured sharpness voxel and
is still capped by footprint-area non-regression; it is not a category or
instance-size tolerance.

Leave-one-edge-family and alternate-anchor solves are development diagnostics.
They must agree within the versioned stability tolerance before V3 is promoted,
but they do not change the fixed production gauge.

## Experiment matrix

The first implementation exposes three frozen experiment variants over the
same component trace and graph edges:

| Variant | Local measurements | Global solve | Purpose |
|---|---|---|---|
| `sequential_v2_1` | current point-to-point against growing aggregate | no | frozen baseline |
| `point_to_point_pose_graph` | pairwise trimmed point-to-point | yes | isolate order/global-consistency benefit |
| `normal_aware_pose_graph` | symmetric normal-aware point-to-plane with partial information | yes | test correspondence and observability benefit |

No variant may change component indices, anchor, perturbation, displayed
points, frame colors, or axes within one comparison.

Use three of the five current dense tracks for development and keep two as a
small holdout against obvious per-instance overfitting. This split is only a
controlled diagnostic because all five come from one Clip; it is not a public
accuracy test.

## Promotion gate

The controlled suite is a correction-capability diagnostic, not gold accuracy.
It reports both absolute error to the frozen model-track proxy and equivariant
error to the same algorithm's unperturbed output. The initial engineering gate
for the strong 15 cm / 2 degree profile applies to the equivariant family:

- at least 80% translation and yaw RMS recovery on every development and
  holdout track;
- equivariant output translation RMS at most 0.02 m and yaw RMS at most 0.25
  degrees;
- no negative recovery in mild or medium profiles;
- no unavailable geometry frame hidden from the denominator.

Absolute proxy metrics remain mandatory report fields and must not be hidden or
used as the promotion recovery score. Natural unperturbed tracks must
simultaneously pass the existing same-point non-regression checks and visual
review. This prevents a consistently wrong unperturbed/perturbed pair from
passing through equivariance alone. These values are a Stage 3 engineering
gate, not final annotation tolerances. Final release still requires physically
separate reviewed targets and X-Points correction-time evaluation.

If none of the pairwise pose-graph variants passes, Stage 4 remains blocked.
The next response is to inspect rejected/weak edge evidence and revise the
local measurement model, not to relax the gate until the experiment appears
successful.

## Trace and implementation boundary

During experimentation, graph diagnostics use a separate versioned sidecar so
the accepted evidence-trace contract is not repeatedly changed. The sidecar
contains:

- node/frame IDs, anchor, input and candidate corrections;
- every attempted edge and its status;
- estimator variant, correspondence counts, overlap, residuals;
- observable rank, scaled information eigenvalues, and information matrix;
- optimizer termination, cost families, and trajectory diagnostics;
- aggregate baseline/candidate sharpness and rejection reasons.

Only after one variant passes the frozen gate will its stable fields be folded
into the production trace and replace the sequential Stage 3 implementation.
