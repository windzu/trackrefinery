# Development Data

## What constitutes one case

One benchmark case is one already-associated instance track. It references
shared full-frame point clouds, supplies the instance's coarse detection in
each frame and the exact frame poses, and has a physically separate reviewed
target containing one canonical size and per-frame reference poses.

The benchmark never uses GT boxes to crop input points. Real coarse boxes must
come from a frozen detector/tracker profile. Synthetic coarse boxes may be
generated from known truth with recorded perturbations for controlled tests.

## Tier 1: committed synthetic corpus

The deterministic corpus is safe to commit and must exercise the complete
contract rather than only object crops:

- moving ego with stationary and moving objects;
- exact and non-uniform frame timestamps;
- intentionally biased coarse size, center, height, and yaw;
- complementary front/side/rear observations;
- one-sided and sparse evidence that must report insufficient evidence;
- ground, background outliers, neighboring objects, and overlapping crops;
- duplicate spatial evidence from multiple upstream channels;
- non-default annotation-frame names and exact world round trips.

Synthetic data proves coordinate math, optimization behavior, and regressions.
It cannot qualify real-world geometry accuracy.

## Tier 2: local X-4D corpus

Private operational data lives under ignored `.data/`. The synchronized native
Clips are immutable source material, not direct benchmark inputs. An exporter
constructs source-only frame clouds and fixed model track inputs separately
from reviewed targets.

The current local seed inventory is:

| Clip | Initial purpose | Current limitation |
|---|---|---|
| `20260714_G150-004_000` | adapter and smoke test | only one labeled instance |
| `20260720_G150-004_000` | early calibration | eight labeled instances |
| `20180830_n008_scene-0757_sweeps_000` | multi-category comparison | one physical test scene |
| `20260817_G150-002_000` | qualitative failure review | no reviewed geometry target |

These Clips are enough to build and exercise the pipeline, but not enough to
claim accuracy. The first useful real corpus should contain at least several
independent Clips and deliberately cover:

- near/mid/far range and dense/sparse returns;
- full, partial, and one-sided visibility;
- moving and stationary vehicles with moving ego;
- close neighbors, occlusion, ground slopes, and background structures;
- common rigid vehicle categories and unusual dimensions;
- high-quality coarse tracks and visibly biased coarse tracks.

Every real case records source Clip identity, source revision, point
materialization identity, detector checkpoint, tracker/profile identity, and
exporter version. Coarse predictions are immutable benchmark input; rerunning a
new detector creates a different input suite.

The current local catalog has four real Clip tabs and 119 instance cards. The
81 tracks for `20260817_G150-002_000` come from the frozen CenterPoint
`offline-geometry-v3` candidate and are qualitative because that Clip has no
reviewed target. The other 38 tracks come from current source annotations and
are review-only alignment references: they are neither frozen detector inputs
nor reviewed gold. Consequently, no current Clip yet supports a qualified
coarse-versus-refined-versus-gold comparison. The next benchmark preparation
step is to freeze model candidates for the three annotated Clips and review
matched target tracks independently in X-Points.

### Rejected V1 baseline observation

The same-point A/B bundle for the first real V1 candidate in
`20260817_G150-002_000` contains 30,239 selected target points. V1 registration
reduced per-frame centroid XY RMS from `0.654 m` to `0.411 m`, but changed the
1--99% target envelope from `4.077 x 2.246 m` to `4.100 x 2.479 m` and increased
its XY area from `9.158 m^2` to `10.164 m^2`.

This is a recorded qualitative design failure, not an accuracy result: the Clip
has no reviewed target. It demonstrates that V1's local registration metric can
improve while the visible aggregate becomes wider and less vehicle-like. V2
must reproduce this same-point case as a non-regression test and must reject a
candidate with this behavior.

### V2 Stage 3 dense review slice

The first anchored-aggregation review slice contains the five dense-supported
model tracks in `20260817_G150-002_000`. They contribute 17, 17, 11, 6, and 5
geometry frames respectively. All geometry frames retained authority; proposed
movements that were unnecessary or failed a local improvement check kept their
exact coarse pose.

Across this slice, accepted non-zero corrections stayed below 0.069 m and 1.36
degrees. Trimmed cross-frame RMSE decreased for every track. Voxel concentration
increased materially for three tracks and changed by less than 0.0004 for the
other two, within the explicit non-regression allowance. Fixed same-point top
and side A/B review shows small sharpening rather than the V1 envelope
expansion. This remains qualitative development evidence because the Clip has
no physically separate reviewed targets; it does not qualify pose accuracy or
canonical dimensions.

The visible gain is small because these five frozen model tracks are already
closely aligned. They establish that Stage 3 preserves good dense inputs, not
that it can repair materially biased poses. The next gate is the controlled
pose-recovery suite defined in `evaluation-and-review.md`: reuse the exact
selected real components, inject deterministic 5/10/15 cm and 0.5/1/2 degree
non-anchor drift, and show reference/input/output shared-axis aggregates. The
original model-track poses are an evaluator-side proxy reference, never a gold
target or Stage 3 input. Component indices are deliberately frozen before
perturbation because this benchmark isolates registration from crop selection.

### First controlled-recovery result

The first complete suite applies all three profiles to the same five dense
tracks. Translation RMS improves in 13 of 15 case/profile runs, while yaw RMS
improves in only 9 of 15. Under the strongest 15 cm / 2 degree profile, the
per-track translation RMS reduction is 20.2%, 6.7%, 32.6%, 44.4%, and 33.7%.
Yaw RMS reduction is -13.4%, 37.4%, 85.8%, 7.4%, and 37.3% respectively. The
negative result is a real regression, not hidden by an average.

This establishes limited local correction capability but fails the Stage 3
gate: one track regresses in strong-profile yaw, small perturbations can be
dominated by the model track's own residual bias, and most injected error
remains after aggregation. The reference/input/output views make the successful
and failed cases visible with identical points and axes. Stage 4 sizing must
remain blocked while Stage 3 registration is redesigned and rerun against this
frozen suite.

## Gold-target preparation

Existing annotation is a candidate target, not automatically gold. Selected
tracks need a dedicated X-Points review in which one canonical instance size
and all evaluated frame poses are corrected together. The reviewer also marks:

- frames with sufficient visual/point evidence for pose evaluation;
- tracks whose physical extent cannot be established from available data;
- ambiguous category, articulation, truncation, or association problems;
- whether a result at the final tolerance would require any manual correction.

Gold targets are versioned and never passed to the refiner.

## Tier 3: reproducible public benchmarks

Provide adapters and split manifests for user-downloaded nuScenes and Argoverse
2 Sensor data. Do not redistribute third-party sensor assets. Public labels
still require filtering for tracks whose geometry is consistent enough to act
as a refinement target.

## Split discipline

- `development`: algorithm debugging; results may guide implementation;
- `calibration`: diagnostic/insufficient-evidence thresholds only;
- `test`: locked comparison after a version is frozen;
- `qualitative`: visual failures without metric claims.

Split by physical sequence, never by track or frame. A test failure may be
added to a future development split only in a new benchmark version; the
current locked result remains recorded.
