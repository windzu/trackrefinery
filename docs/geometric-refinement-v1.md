# Deterministic Geometric Refinement V1

Status: accepted algorithm design; evidence, registration, visible-envelope,
and alternating trace stages implemented

## Decision

The first TrackRefinery backend is a deterministic, non-learned geometric
optimizer. It uses no neural network, trained weights, training dataset, CAD
template, or category-conditioned dimension prior. The optional input category
is not consumed by this backend.

"Non-learned" does not prohibit an explicit geometric state. The requested
output is itself a rigid cuboid, so the backend estimates one cuboid and its
motion using physical geometry and temporal constraints:

- one canonical `(length, width, height)` for the complete rigid instance;
- one object pose for every input frame;
- target, ambiguous, background, and ground evidence assignments;
- diagnostics proving that the successful result was observable and stable.

The backend must return `insufficient_evidence` when those claims cannot be
supported. A copied detector box, median detector size, class default, or
unobserved extrapolation is never a successful refinement.

The implementation class is provisionally named `JointCuboidRefiner`.

## V1 scope

V1 targets rigid, approximately upright road vehicles whose annotation can be
represented by one cuboid for the entire track. Pedestrians, cyclists,
articulated vehicles, objects with changing geometry, and tracks with an ID
switch are outside the first backend's success domain. They receive explicit
unsupported or insufficient-evidence reason codes; the public data contract
does not need to change.

The public output remains a full `Pose3D`. Internally, V1 decomposes pose into:

- horizontal translation and yaw;
- vertical placement;
- roll and pitch supported by a robust local ground plane.

Free roll and pitch are not optimized from sparse object returns alone. When a
required support plane is unavailable or inconsistent, the backend must not
claim a fully refined pose.

## Evidence and coordinate use

For every frame, points and the coarse box first remain in the declared
annotation frame. Exact `T_world_from_annotation` values place coarse poses in
the shared world frame and allow motion hypotheses to be evaluated without a
fixed-frame-rate or fixed-vehicle assumption. Final poses are transformed back
to each frame's annotation frame.

The backend consumes all fused per-frame LiDAR points supplied by the adapter.
It does not select a sensor channel and does not ask the caller for an object
crop or crop margin. Point timestamps, sensor provenance, and sensor origins
are used when present, but absence is represented as reduced observability
rather than filled with guessed metadata.

Coarse boxes serve only as bounded localization and initialization evidence.
The backend builds an adaptive, finite region of interest from the union of
the coarse and current candidate boxes. The region must be large enough to
recover a biased box, but it must remain bounded so a bad initialization cannot
turn one-instance refinement into scene-wide segmentation.

## Why a one-shot aggregate fit is insufficient

The existing experimental MMDetection3D postprocessor transforms cropped
points using the coarse per-frame boxes, voxelizes the aggregate, estimates
quantile extents, and applies one common center/yaw correction to the track.
That approach has three structural failure modes:

1. Per-frame pose errors smear the aggregate and inflate its dimensions.
2. Ground, neighboring objects, and background returns can become extrema.
3. A common correction cannot repair independent pose errors in each frame.

PCA, a minimum-area rectangle, raw extrema, and a single L-shape fit have the
same one-shot limitation under partial visibility. They remain useful only as
initial hypotheses or comparison baselines.

## State and hypotheses

For a track with frames `i = 1..N`, the joint state contains:

- positive canonical dimensions `s = (l, w, h)`;
- per-frame world poses `T_world_from_object[i]`;
- per-frame ground support where observable;
- latent point-evidence assignments and robust weights.

Dimensions are parameterized in log space during optimization. Angle
differences use circular arithmetic. All motion terms use exact timestamp
deltas.

The solver evaluates a small deterministic hypothesis set instead of trusting
one local optimum:

- bounded yaw offsets around the coarse track;
- the 180-degree heading ambiguity;
- length/width axis exchange;
- static and dynamic trajectory hypotheses;
- bounded evidence-region scales.

The static hypothesis shares one world object pose across frames. The dynamic
hypothesis has per-frame poses with robust velocity, acceleration, and heading
continuity terms. Model selection uses evidence fit, stability, and a fixed
complexity penalty; no learned motion classifier is required.

## Evidence assignment

Each optimization round assigns region-of-interest points to four states:

- `target`: sufficiently supported object evidence;
- `ambiguous`: plausible target or background evidence that must not drive an
  outer boundary;
- `background`: evidence inconsistent with the current object hypothesis;
- `ground`: locally estimated supporting surface.

Initial target seeds come from the coarse box and spatial components that
intersect its supported interior or visible boundary. Assignment then combines:

- robust spatial connectivity at range-aware resolution;
- consistency after transforming observations into the current object frame;
- support persistence across frames or sensors;
- agreement with visible cuboid boundaries;
- separation from the local ground surface;
- LiDAR free-space checks when per-point sensor origins are available.

Ambiguity is first-class. A point shared with a neighboring object, a connected
ground patch, or a transient return is not forced into the target merely to
increase point count.

## Separate registration from cuboid sizing

Most returns from a vehicle do not lie on its annotation cuboid. Bonnet,
windows, roof curvature, wheel wells, and recessed surfaces are legitimately
inside the box. Therefore V1 must not minimize the distance from every target
point to a cuboid face; that objective would shrink the box onto interior
surfaces.

Two different evidence objectives are required:

1. **Shape registration** uses robust point-to-point or point-to-local-surface
   correspondence between frame evidence and the accumulated canonical object
   shape. It estimates per-frame pose.
2. **Cuboid envelope fitting** uses containment, supported outer-envelope
   points, visible faces, ground contact, and free-space constraints. It
   estimates the one canonical size.

Interior target points support registration and containment but do not pull a
cuboid face inward. Only observable outer-envelope evidence may tighten a
face.

### Implemented registration and visible-envelope stages

The current implementation keeps registration separate from the size estimate:

- points begin in each coarse-box-local frame, while coarse object poses are
  composed through exact `T_world_from_annotation` transforms before candidate
  poses are returned to the annotation frame;
- bounded planar principal-axis initialization supplies an orientation seed
  only; it is not treated as a fitted cuboid or a final pose claim;
- deterministic voxel representatives and robust point-to-local-surface
  residuals jointly refine horizontal translation, vertical translation, and
  yaw across frames;
- every frame is matched against the other frames rather than its own points,
  preventing self-correspondence from falsely reporting a zero residual;
- canonical points are retained only when another configured number of frames
  supports them within the configured spatial radius.
- cross-frame persistence and the current candidate envelope deterministically
  reassign target, ambiguous, and background evidence between rounds;
- supported outer tails estimate horizontal faces, the robust ground plane
  anchors the lower face, and the supported upper tail estimates height;
- registration, evidence reassignment, and envelope fitting alternate to fixed
  convergence and iteration limits without category-conditioned priors.

The resulting per-frame poses, canonical points, and one canonical cuboid are
development candidates inside the trace. They are not copied into
`RefinementSuccess`. Sparse or degenerate frames carry explicit reason codes,
and the entire backend remains gated until hypothesis selection and all
observability and stability checks exist. Face support counts are stored in
`(x_min, x_max, y_min, y_max, z_min, z_max)` order.

SciPy spatial indexing is loaded only by this stage through the optional
`geometric` package extra. The public contracts and dataset tools retain their
NumPy-only installation path.

## Joint objective

The conceptual robust objective is:

```text
registration residual
+ visible-envelope and containment residual
+ LiDAR free-space violation
+ ground/contact residual
+ temporal trajectory residual
+ weak bounded initialization residual
```

The initialization term prevents implausible jumps when evidence is noisy but
cannot establish an unobserved dimension. Category-specific expected sizes are
not part of the objective. Broad numerical bounds exist only to keep the
solver finite and reject impossible output.

Sensor-origin rays provide asymmetric information: free space before a return
can reject a candidate near face, while an unseen far face remains
unconstrained. The optimizer must preserve that asymmetry rather than centering
a guessed size around a visible surface.

## Alternating solver

Each hypothesis follows a deterministic alternating procedure:

1. Validate inputs, transform coarse poses to world, and estimate local ground.
2. Select the current evidence region and assign point states.
3. With canonical shape and dimensions fixed, refine every frame pose against
   the accumulated shape plus temporal constraints.
4. Transform high-confidence evidence into the current object frame.
5. With poses fixed, update the canonical cuboid from supported outer
   envelopes, visible faces, ground contact, and free-space evidence.
6. Reassign evidence using the updated state.
7. Repeat until state change and objective reduction satisfy fixed convergence
   criteria, or until a fixed iteration limit is reached.
8. Compare all hypotheses using the same quality and stability measures.

Correspondences and robust weights are recomputed between optimization rounds,
not frozen from the coarse boxes. Stable voxel representatives, point ordering,
tie-breaking, and any sampling are deterministic.

The first implementation should use NumPy plus SciPy optimization and spatial
indexing behind an optional `geometric` package extra. Open3D is not a core
runtime dependency. The base contracts, dataset reader, evaluator, and review
tools remain usable with the existing lightweight installation.

## Observability and success gate

Convergence alone is not success. A candidate must prove all of the following:

- each dimension is constrained by the required opposing-side or equivalent
  geometric evidence;
- height has consistent ground/bottom and upper-envelope support;
- view direction and surface coverage meet versioned thresholds;
- target/background separation is not dominated by ambiguous points;
- all per-frame poses have sufficient registration support and bounded
  residuals;
- multi-start hypotheses converge to one compatible result;
- leave-one-frame-out and supported sensor-dropout fits remain within bounds;
- the numerical system is sufficiently conditioned for every claimed degree
  of freedom;
- the result changes the coarse initialization only where direct evidence
  supports that change.

The result contract requires a refined pose for every input observation.
Therefore one unobservable required frame makes the whole V1 result
`insufficient_evidence`; V1 does not publish a partially successful track.

Initial stable reason-code families are:

```text
unsupported_object_geometry
insufficient_target_points
insufficient_view_coverage
unobservable_length
unobservable_width
unobservable_height
ground_support_unavailable
pose_unobservable:<frame_id>
excessive_background_ambiguity
hypothesis_disagreement
unstable_frame_dropout
unstable_sensor_dropout
ill_conditioned_solution
optimization_not_converged
```

`algorithm_stage_incomplete` is a development-only gate used while the backend
can produce diagnostics but the joint optimizer is not yet capable of a valid
success. It is removed from normal outcomes once all mandatory V1 stages are
implemented.

Threshold values are calibration data, not universal truths. They must be
versioned and selected on a calibration split without tuning on the test split.

## Diagnostics and review trace

Every run, including insufficient outcomes, records JSON-safe summary
diagnostics:

- algorithm and configuration version;
- evaluated hypotheses and selection reason;
- objective values and convergence history;
- point-state counts per frame;
- view, face, ground, and sensor coverage;
- per-frame registration residuals and pose corrections;
- canonical dimension support intervals;
- dropout stability and conditioning measures;
- final success or rejection reason codes.

Development runs additionally produce an optional trace sidecar containing
region-of-interest point indices and point-state masks aligned with each input
frame. Large masks do not belong in `result.json`; the review bundle references
them and renders selected, ambiguous, background, and ground evidence using
distinct colors.

## Evaluation requirements

The backend is evaluated against the frozen coarse input and physically
separate reviewed targets. Required test families include:

- exact and noisy static vehicles;
- moving vehicles with irregular timestamps;
- biased per-frame centers and yaw;
- partial and opposing view coverage;
- sparse returns and complete occlusion;
- ground slope and height ambiguity;
- neighboring vehicles, background structures, and outliers;
- multiple LiDAR origins and sensor dropout;
- frame-order, point-order, world-frame, and sensor-order invariance.

Primary success is strict successful-track precision. Coverage is secondary.
The implementation is not ready for X-4D release testing merely because it
improves mean IoU; successful tracks must pass dimension and every-frame pose
tolerances and require no geometry correction in blinded X-Points review.

## Delivery sequence

1. **Implemented:** add versioned geometric settings, trace types, and a
   backend that returns explicit insufficient evidence until a hypothesis
   passes its gate.
2. **Implemented:** add deterministic initial evidence selection and ground
   estimation with review-mask visualization. Point ownership remains an
   initialization and will be recomputed by later optimization rounds.
3. **Implemented:** add deterministic robust per-frame registration and
   cross-frame-supported canonical shape aggregation. Candidate poses remain
   trace-only.
4. **Implemented:** add visible-envelope cuboid fitting and the alternating
   evidence/registration/envelope loop. Candidate dimensions remain trace-only.
5. Add multi-hypothesis selection, observability, and dropout stability gates.
6. Calibrate on reviewed development/calibration tracks, freeze thresholds,
   and run blind test and X-Points review.
7. Integrate the published package into MMDetection3D only after the standalone
   backend meets the acceptance contract.

## Research references

- Pang, Li, and Wang, *Model-free Vehicle Tracking and State Estimation in
  Point Cloud Sequences*: optimization-based registration, accumulated shape,
  correspondence, and motion priors. The task differs from TrackRefinery, and
  the published repository is a research reference rather than imported code:
  <https://arxiv.org/abs/2103.06028> and
  <https://github.com/tusen-ai/LiDAR_SOT>.
- Zhang et al., *Efficient L-Shape Fitting for Vehicle Detection Using Laser
  Scanners*: useful search-based visible-edge initialization, not a complete
  multi-frame refinement solution:
  <https://publications.ri.cmu.edu/efficient-l-shape-fitting-for-vehicle-detection-using-laser-scanners>.
- Ma et al., *DetZero*: motivates decomposing constant instance geometry from
  time-varying trajectory attributes. Its learned refiner is explicitly outside
  this V1 decision: <https://arxiv.org/abs/2306.06023>.

No third-party source code is copied or vendored by this design.
