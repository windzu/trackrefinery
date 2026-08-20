# Learned Track Refinement Plan V1

Status: superseded research plan; retained for historical comparison; no
learned backend is release-qualified

> This direct trajectory box-regression direction was superseded by the
> [object-centric foundation exploration](object-centric-foundation-exploration-v1.md).
> That exploration is now deferred behind the deterministic observable-core
> MVP.
> In particular, production detector tracks are no longer the primary training
> distribution, and `delta_log_lwh` is retained only as a negative-control
> baseline. Do not implement the milestones in this document as the current
> architecture.

## Decision

TrackRefinery's next production research direction is a learned,
trajectory-level refiner. Its historical proposal assumed:

```text
full per-frame clouds + one associated coarse track + exact frame poses
    -> one canonical length/width/height
    -> one refined x/y/z/yaw pose for every input observation
    -> success with diagnostics, or explicit insufficient evidence
```

The first experiment is deliberately narrower: learn only the one canonical
size while holding the pose source fixed. It must prove that learned temporal
reasoning improves physical dimensions before pose refinement is added. A
size-only experiment is not allowed to return public `RefinementSuccess`.

This direction follows the problem decomposition demonstrated by Auto4D,
LabelFormer, and DetZero. LabelFormer is the preferred target architecture
because one model jointly reasons over a complete trajectory. DetZero's
open-source GRM and PRM are implementation references for full 3D size and
per-frame pose prediction. TrackRefinery will not embed either project's data
pipeline or expose a Waymo/OpenPCDet-shaped public API.

The deterministic V2/V3/V4 implementations remain frozen comparison and
diagnostic backends. They are not silently composed with learned output and do
not become a fallback that can turn an unsupported learned result into a
successful refinement. A caller may compare independently produced candidates;
release policy remains outside the library.

## Why size comes first

The reviewed failure is primarily canonical geometry: sparse opposing faces
and a few clutter points can dominate an explicit envelope fit. A learned
model can use the coarse trajectory and repeated object evidence without
treating every extreme point as a physical boundary.

Size-first development isolates that claim. Jointly changing alignment,
dimensions, confidence, and integration in the first experiment would make a
positive or negative result impossible to attribute. If learned size does not
beat the frozen baselines on reviewed targets, work stops before a learned pose
head is built.

## Invariants

- The logical request remains exactly one already-associated rigid-object
  track.
- The caller supplies full clouds. Evidence-region extraction and its margin
  policy are owned by the backend and recorded in the checkpoint profile.
- Existing annotations and evaluation targets are never inference features.
- Exact integer timestamps and `T_world_from_annotation` are retained. A model
  cannot assume fixed frame rate or a vehicle-specific annotation frame.
- One successful rigid-instance result contains one byte-identical canonical
  size for all frames.
- The first supported domain is upright rigid road vehicles. Articulated,
  non-upright, or unsupported categories return insufficient evidence until an
  explicit model profile supports them.
- Core contracts remain NumPy-only. PyTorch and learned-model tooling are
  optional package extras.
- Checkpoints and private data are external artifacts and are never committed
  to this repository.

## Proposed model

### Shared preprocessing

The learned backend receives a normal `RefinementCase` and performs all of the
following internally:

1. choose a deterministic trajectory reference frame from exact frame poses
   and the coarse track;
2. transform every coarse box and candidate point into that trajectory frame;
3. select an enlarged proposal-relative evidence region from each full frame;
4. transform selected points into the corresponding coarse object frame;
5. sample or voxelize points with an explicit validity mask; and
6. encode exact relative timestamps, detector scores, point counts, and coarse
   box parameters as evidence, never as truth.

Crop expansion, voxel size, point budget, frame budget, and normalization are
versioned checkpoint-profile fields. They may have defaults, but they must not
be unexplained constants hidden in inference code. Training and inference use
the same serialized profile.

Non-finite points are rejected by the existing input contract. Empty and weak
frames remain represented through masks and evidence features instead of being
silently duplicated to look dense.

### Milestone A: canonical-size network

The first network contains:

- a shared per-frame point encoder, initially a compact PointPillars/PointNet
  style encoder over proposal-relative points;
- an MLP encoding the coarse box and frame evidence;
- masked cross-frame self-attention with relative time encoding;
- track pooling over valid frame tokens; and
- a head predicting `delta_log_lwh` and per-axis uncertainty relative to a
  robust coarse-track size reference.

Using a residual makes the optimization well conditioned but does not grant
authority to detector dimensions. The model is trained to correct that
reference from point and temporal evidence. It produces a development result
sidecar containing the predicted size, uncertainty, evidence summary, and
checkpoint identity. For the primary experiment, the predicted size is
materialized on the exact frozen coarse per-frame poses used by the baseline;
those poses are not changed. A separately named ablation may use frozen V3
poses, but the two pose sources are never mixed in one comparison. The sidecar
is never serialized as a successful public refinement.

### Milestone B: joint size and pose network

Only after Milestone A passes, add a per-frame head that predicts residual
`x/y/z` and yaw as a normalized sine/cosine representation. Track pooling still
produces exactly one `length/width/height`; per-frame tokens produce pose
residuals for all input observations.

The first joint model handles static and dynamic vehicles with one shared
architecture. It does not use a static/dynamic switch to select different
networks. Exact time and the full coarse trajectory are features. Supervised
losses cover log dimensions, center residuals, yaw, box corners/IoU, and
temporal size invariance. Any temporal pose regularizer must use exact time and
must not force a moving object toward a static or constant-velocity path.

The MVP refines upright boxes. Roll and pitch support requires a separate data
and contract review rather than adding unconstrained quaternion regression.

### Milestone C: calibrated abstention

The model must be able to decline. The initial quality signal combines:

- predicted per-axis and per-frame uncertainty;
- valid-frame and point-evidence counts;
- disagreement across deterministic point/frame subsamples; and
- the magnitude of the requested correction.

A held-out calibration split maps these signals to the versioned strict
acceptance predicate. The mapping and thresholds are stored with the
checkpoint profile. No threshold is tuned against the locked test set.

If any required frame or the shared size is unsupported, the public backend
returns `InsufficientEvidence`; it does not copy coarse boxes or call a
geometric fallback and report success.

## Data plan

### Sources

Training examples combine:

1. real frozen CenterPoint/tracker trajectories matched to annotation tracks;
2. independently reviewed canonical sizes and per-frame poses for calibration
   and locked testing;
3. controlled perturbations of training-only annotations for curriculum and
   corner cases; and
4. explicit hard examples containing neighboring vehicles, ground,
   occlusion, partial visibility, sparse returns, false track tails, and size
   outliers.

Actual model trajectories are the primary training distribution. Training
only on annotation boxes with synthetic noise is not an acceptable result
because it does not reproduce detector/tracker error, crop contamination, or
association artifacts.

Argoverse 2 or another user-downloaded public corpus may bootstrap the point
encoder. Public-domain evaluation and the G150 operational evaluation remain
separate strata; cross-domain gains are not presented as G150 qualification.

### Physical separation and splits

Training code gets an explicit training bundle that can open both examples and
labels. The production `InferenceDataset` API remains unchanged and has no
target access. Calibration and test evaluation continue to join predictions
with targets in a separate evaluator process.

All derivatives of one physical Clip remain in one split, including alternative
tracks, perturbations, crops, and detector profiles. Splitting by instance or
frame is forbidden. Every manifest records source Clip revision, point
materialization identity, detector checkpoint, tracker profile, adapter
version, label revision, and content hashes.

The staged corpus targets are:

| Corpus stage | Minimum purpose |
|---|---|
| Smoke | 50 vehicle tracks from at least 3 Clips; pipeline and overfit checks only |
| Feasibility train | 1,000 usable vehicle tracks from at least 30 physical Clips |
| Selection validation | 200 reviewed tracks from at least 10 held-out Clips |
| Calibration | 100 reviewed tracks from at least 5 held-out Clips |
| Locked test | 200 reviewed tracks from at least 10 additional held-out Clips |

These are minimum planning targets, not a claim that 1,000 tracks guarantee
accuracy. They may be revised in a new dataset manifest before the selection
and test splits are locked, never after inspecting their results. Model and
architecture choices use the selection-validation split. The locked test is
opened once only after the joint model, checkpoint, preprocessing profile, and
calibration rule are frozen. A smaller corpus may run engineering smoke tests
but cannot promote a backend.

The present four-Clip local inventory does not meet this gate. The immediate
data task is to audit available annotated Clips, freeze model trajectories for
them, and select independent Clips for review. `20260817_G150-002_000` remains
qualitative until a reviewed target exists.

### Augmentation

Training-only augmentation should cover realistic initialization failures:

- correlated and per-frame center/yaw drift;
- biased and jittering detector dimensions;
- missed head/tail observations and weak frames;
- point dropout, range-dependent sparsity, and multi-LiDAR density changes;
- neighboring-object and background contamination; and
- score corruption and non-uniform timestamp spacing.

Augmentation is applied to inputs, not targets, and its random seed/profile is
recorded. Test data is never synthetically cleaned or densified.

## Evaluation and promotion gates

Every gate compares the same frozen cases and publishes all cases in the
existing tabbed review site. Aggregate metrics without per-instance review are
not sufficient.

### Gate 0: data and tolerance readiness

- train, selection-validation, calibration, and test manifests pass
  sequence-level leakage checks;
- target review records one canonical size and evaluable frame poses;
- annotation-owner tolerances are frozen in a versioned
  `AcceptanceThresholds` file; and
- coarse-median, Stage 4 geometric, and any learned predictions use identical
  case IDs and target revisions.

No model result can be described as accurate before Gate 0.

### Gate 1: size feasibility

On the frozen selection-validation split, the learned size-only model must:

- improve full-suite strict size pass rate over coarse-track median dimensions;
- report candidate coverage separately and, on cases shared with the frozen
  Stage 4 geometric candidate, improve or retain strict size accuracy;
- reduce length, width, and height MAE and P95 without hiding a regressing axis
  in an average;
- report all outliers and paired per-track deltas;
- retain zero temporal size variance by construction; and
- show a visible reduction in size-correction work in blinded review.

Paired bootstrap confidence intervals accompany metric deltas. If this gate
fails, development returns to data/architecture analysis and the pose head is
not started.

### Gate 2: joint refinement

The joint model must retain Gate 1 geometry performance and improve the
existing center/yaw/IoU metrics over the coarse track. P95 and worst-frame
limits are mandatory; a good median cannot mask one badly moved frame. The
review site shows coarse, size-only, joint, and gold aggregates on shared axes.
Gate 2 uses the same selection-validation split, not the locked test.

### Gate 3: success calibration

After the checkpoint, preprocessing profile, and calibration rule are frozen,
the evaluator opens the untouched locked test once. On that test:

- catastrophic successful tracks are zero;
- the one-sided 95% confidence lower bound for strict successful-track
  precision is at least 95% for the pilot profile;
- every accepted result satisfies canonical-size and per-frame hard limits;
- insufficient-evidence coverage is reported, never optimized away; and
- blinded X-Points review confirms the accepted tracks do not require size or
  pose correction at the agreed pilot tolerance.

A production profile may set a stricter precision target and therefore needs a
larger locked test set. The target and sample-size rationale must be frozen
before that test is run.

### Gate 4: integration

Only a Gate 3 checkpoint can be exposed as a successful TrackRefinery backend.
MMDetection3D installs the published package and supplies the already loaded
full frame clouds and one associated track per call. It owns checkpoint/profile
selection but not TrackRefinery preprocessing internals.

X-4D and the Devkit protocol do not change. The algorithm service still
receives one complete Clip and returns one complete candidate. Predictions are
written only to the service result/candidate path; native Clip caches,
annotations, and training caches remain input-only.

## Implementation slices

### Slice 0: data readiness

- add a learned-training manifest and loader separate from production loaders;
- add sequence leakage and provenance validation;
- export frozen model tracks for annotated Clips;
- create the first reviewed smoke/calibration targets; and
- freeze acceptance thresholds with X-Points examples.

Deliverable: a versioned dataset report and tabbed baseline review. No neural
network is required for this slice.

### Slice 1: reproducible size baseline

- add optional `learned` and `training` dependencies;
- implement serialized preprocessing/checkpoint profiles;
- implement the per-frame encoder, trajectory attention, size head, and
  uncertainty output;
- add deterministic training/inference CLIs and run manifests; and
- produce coarse/geometric/learned size reports and review bundles.

Deliverable: a development sidecar only. Public success remains impossible.

### Slice 2: size iteration

- run ablations for point-only, box-only, and fused input;
- test full-context attention against short-window and pooled baselines;
- test crop/point-budget sensitivity through versioned profiles; and
- inspect every regressed target before changing the architecture.

Deliverable: Gate 1 decision with a frozen checkpoint and full report.

### Slice 3: joint pose

- add per-frame XYZ/yaw residual decoding and losses;
- preserve variable-length masks and complete input-frame coverage;
- add controlled pose perturbation and natural-track non-regression reports;
  and
- extend review bundles with size-only versus joint A/B views.

Deliverable: Gate 2 decision.

### Slice 4: abstention and package API

- calibrate uncertainty without touching the locked test;
- implement stable learned-backend settings and reason codes;
- verify CPU import without PyTorch and GPU inference with the optional extra;
- define checkpoint compatibility and provenance validation; and
- run the locked test and blinded review once.

Deliverable: Gate 3 release candidate or an explicit rejected experiment.

### Slice 5: service integration

- publish an exact TrackRefinery package version;
- install it in the maintained MMDetection3D service;
- register an immutable pipeline profile and checkpoint identity;
- run one-Clip functional determinism and cache non-mutation checks; and
- run the X-Points acceptance flow before enabling broader use.

Deliverable: a new algorithm-service profile; no X-4D protocol fork.

## Package layout direction

The exact module names may change during implementation, but ownership is:

```text
src/trackrefinery/learned/
  profile.py          serialized preprocessing/model contract
  preprocessing.py    full-frame to masked trajectory tensors
  model.py            per-frame encoder + temporal model + heads
  checkpoint.py       compatibility and provenance validation
  refiner.py          TrackRefiner adapter and insufficient-evidence policy

src/trackrefinery/training/
  dataset.py          training-only example/label loader
  augment.py          recorded input perturbations
  losses.py
  runner.py
```

The base package must still import without PyTorch. Training helpers never
become reachable from `InferenceDataset`. Model weights live outside wheels;
the wheel contains code and schema versions only.

## Explicit stop conditions

- Do not start model training on random instance-level splits.
- Do not claim success from synthetic data or qualitative Clip aggregates.
- Do not start joint pose work before learned size passes Gate 1.
- Do not increase crop margins or point budgets after looking at locked-test
  failures; create a new benchmark/model version instead.
- Do not release a model that improves mean IoU but still emits accepted tracks
  requiring manual geometry correction.
- Do not hide unsupported tracks with detector-size fallback.

## References

- [LabelFormer: Object Trajectory Refinement for Offboard Perception from
  LiDAR Point Clouds](https://proceedings.mlr.press/v229/yang23e.html)
- [DetZero official implementation](https://github.com/PJLab-ADG/DetZero)
- [DetZero paper](https://arxiv.org/abs/2306.06023)
- [Auto4D](https://arxiv.org/abs/2101.06586)
- [Waymo 3D Auto Labeling](https://waymo.com/research/offboard-3d-object-detection-from-point-cloud-sequences/)
