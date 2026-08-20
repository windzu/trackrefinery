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

### Stage 3 V3 pose-graph result

The first V3 normal-aware pose-graph run uses the same five component traces
and adds an explicit unperturbed-algorithm reference. Controlled drift is
injected around that reference so the equivariant metric isolates recovery of
the declared error; absolute error to the frozen model-track proxy remains in
every report and view.

Four tracks have redundant observable graphs. At the strong profile their
equivariant translation recovery is 98.8%, 96.3%, 92.2%, and 93.6%, leaving
1.2--7.6 mm RMS. Yaw recovery is 98.7%, 97.1%, 98.1%, and 99.4%, leaving
0.010--0.037 degrees RMS. All mild and medium runs for these tracks have
positive recovery; one mild translation run is 78.4%, while the strong-profile
promotion minimum is exceeded by all four.

The six-geometry-frame track is not counted as a success. Its right-hand frame
cluster is internally consistent but connects to the anchor cluster through
one partial-overlap bridge. That bridge cannot disambiguate the cluster yaw,
and a nine-second observation gap makes both local minima trajectory-plausible.
V3 returns `weak_partial_overlap_bridge`, registers no frame, and the controlled
suite reports all five non-anchor frames unavailable. This is the desired
failure mode: four releasable candidates and one explicit rejection, not an
average that hides an incorrect pose.

The generated development suite is under
`.data/real-clip-review/site/recovery-v3/`. It contains PROXY, unperturbed
REFERENCE, perturbed INPUT, and OUTPUT views with identical points, colors, and
axes. The assets remain private development data and are not committed.

The combined suite contains all 45 case/profile/variant bundles. Strong-profile
equivariant recovery percentages are:

| Track suffix | Sequential XY / yaw | P2P graph XY / yaw | Normal graph XY / yaw |
|---|---:|---:|---:|
| `83bdbc95` | 16.2 / 5.0 | 46.0 / 49.5 | 98.8 / 98.7 |
| `8a60c8b6` | 34.3 / 81.7 | 63.8 / 49.2 | 96.3 / 97.1 |
| `fe8dfe0c` | 37.7 / 73.4 | 51.9 / 70.6 | 92.2 / 98.1 |
| `2a9844b6` | 36.4 / 34.9 | 53.9 / 44.3 | 93.6 / 99.4 |
| `33acbf1b` | 46.4 / 20.7 | unavailable | unavailable |

The rejected row is not a regression hidden by the table: both graph variants
refuse to infer a cluster yaw through a weak bridge, while the sequential
baseline still emits a low-recovery candidate. This is the confidence behavior
required for automatic-label release.

### Stage 4 observable canonical cuboid result

Stage 4 runs only on the V3 normal-aware trace and uses no model dimensions,
category priors, sensor rays, or reviewed annotations. It requires repeated
surface-normal support for both length faces, both width faces, the top, and a
consistent transformed ground boundary. All thresholds are recorded in the
experiment sidecar.

On the same five-track slice, `83bdbc95` is the only sizing candidate. Its
model median size is `6.338 x 2.704 x 3.183 m`; the observable fit is
`6.135 x 2.489 x 3.055 m`. All six boundaries are supported by at least 11 of
17 geometry frames, leave-one-frame-out dimension change is at most 1.6 cm,
and neighboring-resolution change is at most 1.8 cm. The fixed side/top views
show a coherent truck silhouette and a tighter box, but there is no reviewed
target, so this is qualitative evidence rather than an accuracy claim.

`8a60c8b6`, `fe8dfe0c`, and `2a9844b6` retain provisional numbers only in the
diagnostic sidecar and are not drawn as sizing candidates. They lack repeated
normal-aligned support for one or more opposing length/width faces.
`33acbf1b` never enters sizing because Stage 3 rejects its weak partial-overlap
bridge. The five-case page is generated under
`.data/real-clip-review/site/size-v4/`; private assets are not committed.

## Gold-target preparation

Existing annotation is a candidate target, not automatically gold. Selected
tracks need a dedicated X-Points review in which one canonical instance size
and all evaluated frame poses are corrected together. The reviewer also marks:

- frames with sufficient visual/point evidence for pose evaluation;
- tracks whose physical extent cannot be established from available data;
- ambiguous category, articulation, truncation, or association problems;
- whether a result at the final tolerance would require any manual correction.

Gold targets are versioned and never passed to the refiner.

## Future learned-backend corpus

The deferred coverage-expansion direction is documented in the
[object-centric foundation plan](object-centric-foundation-exploration-v1.md).
The current MVP uses the deterministic
[observable-core plan](observable-core-refinement-v1.md) and does not require a
training corpus.
Its training loader is separate from `InferenceDataset`: training may open
explicitly paired examples and labels, while production inference retains no
target API or target-path convention.

The generic representation is trained from broad normalized 3D geometry,
multi-domain public metric trajectories, and recorded observation
randomization. No production detector, LiDAR, vehicle, or Clip family is the
primary distribution. Every derivative of one asset or physical instance
remains in the same split, including alternative detector profiles and
augmented copies. Training, selection-validation, public calibration,
leave-one-domain-out, and locked-test manifests record asset/Clip,
point-materialization, detector, tracker, adapter, label-revision, and content
identities as applicable.

The current four-Clip inventory supports adapter work, qualitative review, and
operational-domain smoke only. It has already informed development and cannot
become a locked qualification set. It must not be used to choose the generic
architecture, tune losses, train the checkpoint, or calibrate its release rule.
Stage 0 therefore begins with public/asset research manifests and factorization
tests, not production-data extraction. Independently selected, previously
unseen production Clips are sealed only after the generic checkpoint and
release policy are frozen.

## Tier 3: reproducible public benchmarks

Provide adapters and split manifests for several user-downloaded public sensor
datasets, with nuScenes and Argoverse 2 Sensor as initial candidates rather than
the complete domain set. Do not redistribute third-party sensor assets. Public
labels still require filtering for tracks whose geometry is consistent enough
to act as a refinement target. Reports include leave-one-dataset,
leave-one-observation, and leave-one-detector holdouts instead of only random
within-dataset splits.

## Split discipline

- `development`: algorithm debugging; results may guide implementation;
- `calibration`: diagnostic/insufficient-evidence thresholds only;
- `test`: locked comparison after a version is frozen;
- `qualitative`: visual failures without metric claims.

Split by physical sequence, never by track or frame. A test failure may be
added to a future development split only in a new benchmark version; the
current locked result remains recorded.
