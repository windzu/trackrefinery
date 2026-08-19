# Object-Centric Foundation Refinement Exploration V1

Status: proposed exploration plan; supersedes the direct box-regression
implementation direction in `learned-refinement-plan-v1.md`; no learned model
is release-qualified

## Decision

TrackRefinery will explore an object-centric, amodal 3D representation rather
than a detector-specific box correction network. The public unit and result do
not change:

```text
full per-frame clouds + one associated coarse track + exact frame poses
    -> one canonical metric length/width/height
    -> one refined pose for every input observation
    -> success with diagnostics, or explicit insufficient evidence
```

The research model must explain the complete track through four separated
factors:

```text
P_t = Observe(sensor, visibility, noise,
              T_t(Scale_d(Shape(z_shape))))
```

- `z_shape` is a normalized, complete canonical object shape and its structural
  parts;
- `d` is the metric scale, including the canonical length, width, and height;
- `T_t` is the object pose at exact input time `t`; and
- sensor sampling, visibility, density, and noise are nuisance variables that
  explain the partial observation but must not define the object.

The model is useful only if it can infer the first three factors while
marginalizing the fourth. Parameter count is not the definition of a foundation
model here. The defining property is a reusable 3D object representation learned
from broad geometry and observation variation before task-specific refinement.

## Product meaning of success

“No manual correction” is literal. A successful result is one for which a
reviewer accepts the complete instance track in X-Points without moving,
rotating, resizing, or otherwise editing any released box. If any released
frame requires one geometry edit, the track is `needs_edit`.

Numeric size, pose, and IoU metrics remain mandatory diagnostics and regression
guards. They do not redefine `accept_as_is`. The product optimization order is:

1. maximize precision among released tracks;
2. keep catastrophic successful tracks at zero on the locked qualification
   set; and
3. increase coverage only while preserving the first two properties.

Ambiguous evidence must produce `InsufficientEvidence`. A plausible generated
vehicle is not necessarily the observed vehicle.

## What the model must learn

### Observation-level geometry

- distinguish repeatable object surface from background, neighboring objects,
  ground, and isolated returns;
- integrate complementary partial observations without treating point density
  as physical thickness; and
- preserve metric coordinates while remaining invariant to point ordering,
  frame count, scan density, and missing regions.

### Object-level structure

- one rigid identity persists through the whole track;
- its canonical shape and dimensions do not change with pose or observation;
- front/back, left/right, top/bottom, symmetry, and part relationships form a
  consistent canonical coordinate system; and
- observed points constrain, but do not completely specify, occluded surfaces.

### Shape-family prior

The model should learn a continuous distribution over road-vehicle geometry,
not a table of average dimensions by class. Sedans, SUVs, vans, buses, trucks,
and unusual rigid vehicles provide structural diversity. Semantics can help
select a region of that shape distribution, but a class name cannot determine
the instance's dimensions.

### Metric grounding and motion

Large 3D shape corpora commonly provide useful normalized geometry but not
reliable real-world scale. Shape and metric scale are therefore explicit,
separate variables. Raw meter-valued observations and real metric targets
supervise `d`; exact timestamps and annotation-frame-to-world poses supervise
`T_t`.

### Uncertainty

Partial observation is intrinsically ambiguous. The representation must support
multiple plausible completions or an equivalent calibrated posterior. The
release signal must reflect disagreement in the resulting metric size and
per-frame pose, not merely a neural-network confidence score.

## What it must not learn

- a fixed CenterPoint, tracker, vehicle, LiDAR, or customer-domain correction;
- category-average dimensions as a substitute for evidence;
- intensity, ring ID, channel name, or LiDAR identity shortcuts in the generic
  checkpoint;
- a crop policy supplied by the caller;
- annotation targets as inference features; or
- a generative shape that looks plausible while violating observed points.

Coarse boxes initially provide localization and association evidence only.
Their dimensions, score, and detector identity are hidden, randomly dropped, or
independently perturbed in the decisive experiments. A separate ablation may
expose them to measure shortcut learning, but that result cannot define the
generic model.

## Research architecture, not yet an implementation commitment

```text
per-frame metric XYZ + exact time + frame pose + coarse localization
    -> object evidence tokenizer with validity/background masks
    -> temporal object encoder
    -> factorized latent state
         |- normalized canonical shape latent
         |- metric scale
         |- per-frame SE(3)/upright pose
         `- completion/pose uncertainty
    -> observation-consistent shape decoder
    -> canonical box and per-frame boxes
```

The first profile remains upright rigid road vehicles and outputs XYZ/yaw poses.
The latent design must not make later full SE(3) support impossible.

Candidate shape representations are evaluated rather than selected on paper:

1. a point/vector-set autoencoder with an implicit occupancy or SDF decoder;
2. a discrete latent shape representation with conditional completion; and
3. a generative latent model only if deterministic completion cannot represent
   the ambiguity needed for abstention.

A pretrained 3D foundation encoder may initialize or teach the observation
encoder. A large geometry model may act as an offline teacher while a smaller
student serves TrackRefinery inference. Neither is accepted unless it improves
held-out-sensor completion, metric refinement, and final review; semantic
classification or visually attractive meshes are not sufficient evidence.

## Training objectives

The exact loss weights are experiment profiles. The required behaviors are:

- **complete-shape reconstruction:** encode a complete normalized shape and
  reconstruct its surface/occupancy faithfully;
- **masked amodal completion:** reconstruct a complete shape from partial,
  noisy, and complementary observations;
- **observation consistency:** decoded surfaces must explain retained observed
  object points without moving them to a merely plausible shape;
- **sampling invariance:** different structured sparsity views of the same
  object produce the same shape and metric scale posterior;
- **pose equivariance:** applying a known rigid transform changes predicted
  pose accordingly but does not change canonical shape or dimensions;
- **shape/scale separation:** normalized shape stays stable across controlled
  scale changes while metric dimensions follow the scale exactly;
- **temporal identity consistency:** arbitrary valid frame subsets describe one
  common shape and scale;
- **metric supervision:** real meter-valued canonical dimensions and per-frame
  poses supervise the final task variables; and
- **posterior calibration:** completion or ensemble disagreement predicts the
  chance that a released track would require an edit.

Direct `points -> delta_log_lwh` regression is retained only as a negative
control. If the richer representation cannot beat it outside the training
domains, the foundation hypothesis has failed for this task.

## Data roles

### Broad normalized 3D geometry

User-downloaded, license-compatible CAD/mesh corpora teach shape, parts,
symmetry, and completion. They are grouped by original asset identity before
splitting. Their normalized geometry must not be treated as metric ground
truth. Procedural variants may broaden proportions, but they cannot qualify
real accuracy.

### Multi-domain real metric data

Multiple public autonomous-driving datasets supply real LiDAR observations,
metric boxes, trajectories, occlusion, clutter, and detector-independent
initializations. No one source, detector, LiDAR type, or geography is the
primary distribution. Dataset-specific adapters produce the portable
TrackRefinery research contract; source-native fields do not enter the core
model.

Training instances are grouped by physical sequence and object identity.
Alternative crops, detector runs, point resamplings, and augmentations of one
instance remain in the same split.

### Observation randomization

Offline augmentation creates multiple observation processes for the same
underlying object:

- structured beam/row removal and angular-bin resampling;
- density and range variation;
- frame and local-region dropout;
- partial visibility and neighboring/background contamination;
- non-uniform timestamps and track tails; and
- independent pose and localization perturbations.

This is training-time observation randomization, not a runtime sensor-ray
algorithm. The first exploration should use simple, recorded transformations
of real or asset-derived point sets; a detailed physical simulator is added
only if a named experiment proves it necessary.

### Production data

Production data is an out-of-domain qualification source, not a training
source. The G150 Clips already inspected during geometric development remain
operational smoke and qualitative examples; they cannot become a locked test.
After the generic checkpoint and policy are frozen, independently selected,
previously unseen production Clips are sealed and opened once for blind
X-Points qualification.

Neither the known smoke Clips nor the locked qualification Clips participate in
representation pretraining, model selection, loss tuning, confidence
calibration, or release-threshold selection. A later domain-adapted model, if
explicitly requested, receives a separate checkpoint/profile identity and
cannot replace or be reported as the generic checkpoint.

## Generalization matrix

Random train/test splits within one dataset are insufficient. Every model
report must include:

| Axis | Required holdout |
|---|---|
| 3D asset identity | unseen vehicle assets and shape families |
| real dataset | leave-one-dataset-out rotation |
| LiDAR observation | unseen density/beam pattern or LiDAR-CS sensor group |
| detector initialization | unseen detector/error profile |
| range and visibility | separately reported sparse, one-sided, and occluded strata |
| operational domain | sealed X-4D production Clips after freeze |

The generic claim is rejected if performance depends on providing sensor ID,
if a held-out observation process causes a material unexplained collapse, or if
production qualification requires fine-tuning on the qualification set.

## Exploration stages and gates

Each stage creates a reproducible report and fixed visual artifacts. Passing an
earlier stage permits, but does not guarantee, work on the next.

### Stage 0: benchmark and factorization harness

Build research-only manifests and evaluators for complete shapes, partial
observations, exact transforms, metric scale, source identity, and split
provenance. Produce deterministic paired views of the same object under
different poses and structured sampling processes.

Deliverables:

- leakage checks by asset, instance, sequence, dataset, and detector profile;
- a common canonical-shape/partial-observation representation;
- direct-regression and no-completion baselines; and
- a review page showing input partials, decoded complete shape, observed-point
  residuals, canonical box, and target.

Gate: known transforms, scales, masks, and held-out partitions round-trip
exactly. No neural accuracy claim is made.

### Stage 1: shape representation probe

Train small versions of the candidate autoencoders on complete 3D assets. Then
mask them into partial views not seen during training.

Compare:

- point/vector-set latent versus discrete/voxel latent;
- deterministic versus probabilistic completion; and
- random initialization versus a frozen or adapted foundation encoder.

Gate: the selected representation reconstructs held-out complete geometry,
preserves structural proportions, and improves held-out partial completion over
a simple mirrored/envelope baseline. Failure stops model scaling.

### Stage 2: shape, scale, and pose disentanglement

Train on controlled pairs where shape, metric scale, pose, and sampling process
are varied independently. Apply transforms never shown in a paired training
example.

Gate:

- canonical shape is invariant to rigid pose and sampling variation;
- predicted pose is equivariant to the known transform;
- metric dimensions follow controlled scale without changing normalized shape;
  and
- same-instance cross-view latent disagreement decreases without collapsing
  distinct instances.

### Stage 3: multi-domain real trajectory transfer

Fine-tune the task interface on public meter-valued trajectories while
retaining reconstruction, consistency, and equivariance objectives. Coarse
boxes localize the object; their size and detector identity remain unavailable
to the primary model.

Run leave-one-dataset, leave-one-observation, and leave-one-detector experiments.
Compare against:

- coarse-track dimensions and poses;
- frozen geometric V2/V3/V4 diagnostics;
- direct trajectory box regression;
- shape latent without generative completion; and
- full factorized model.

Gate: the full model must improve strict track geometry and worst-frame pose on
held-out domains, not only on pooled in-domain metrics. A gain confined to the
training detector or sensor rejects the generic design.

### Stage 4: uncertainty and release policy

Measure disagreement over completion samples, deterministic frame/point
subsets, and independently trained model members. Calibrate the mapping from
those signals to expected `accept_as_is` outcome using public-domain calibration
data only.

Gate: risk-coverage curves must show a stable high-precision region across
held-out public domains. The release policy is frozen before production data is
opened.

### Stage 5: sealed operational qualification

Run the frozen generic checkpoint on untouched production Clips and expose all
cases in the existing Clip-tabbed review site. A reviewer records only
`accept_as_is`, `needs_edit`, or `not_judgeable`, plus optional diagnostic edit
type.

Gate for a pilot release:

- zero catastrophic successful tracks in the locked set;
- the predeclared one-sided confidence lower bound for successful-track
  `accept_as_is` precision is met;
- every released frame is accepted without geometry edits; and
- coverage and every rejected/unsupported stratum are reported.

Failure does not authorize training on the locked set. It creates a new model
version, new hypotheses, and a new independently locked qualification set.

## Required visual review

Every model comparison uses the same case IDs and camera axes. Each instance
page shows:

- all per-frame partial observations, colored by time;
- the canonical aggregate and decoded complete surface;
- observed points overlaid on the decoded surface;
- coarse, predicted, and gold canonical boxes;
- coarse, predicted, and gold boxes for every evaluated frame;
- at least two completions or uncertainty overlays when the posterior is
  ambiguous; and
- a concise explanation of why the backend released or declined the track.

The Clip-level page keeps one top-level tab per source Clip and tiles all its
instances. Aggregate plots never replace per-instance review.

## Initial experiments

The next work item is Stage 0, not production-data extraction and not a large
training run. Its first experiment matrix is deliberately small:

| Experiment | Question |
|---|---|
| `R0` direct box regression | How strong is the shortcut baseline? |
| `R1` masked point autoencoder | Does reconstructive pretraining improve partial-object representation? |
| `R2` implicit shape decoder | Does predicting complete geometry improve metric size transfer? |
| `R3` factorized shape/scale/pose | Does explicit disentanglement survive unseen transforms and sampling? |
| `R4` pretrained 3D encoder | Does broad semantic/shape pretraining add held-out-domain value? |
| `R5` probabilistic completion | Does multiple-hypothesis uncertainty improve safe release? |

`R5` is not started unless deterministic completion shows material ambiguity.
Model size is scaled only after representation and generalization behavior is
demonstrated on small models.

## Reference map

The references are inputs to experiments, not architecture authority.
Before reusing code, weights, or processed data, record the upstream revision,
license, weight terms, and dataset terms in the experiment manifest. A paper may
inform an independently implemented experiment even when its released artifacts
cannot be redistributed in this package.

| Reference | What to borrow | What not to assume |
|---|---|---|
| [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) | masked local point modeling as a representation pretext | classification transfer proves exact metric completion |
| [T-MAE](https://github.com/codename1995/t-mae) | temporal masked point-cloud pretraining | its two-frame detector-oriented pipeline is the TrackRefinery model |
| [OpenShape](https://github.com/kangtengjia/OpenShape), [ULIP-2](https://github.com/salesforce/ULIP), [Uni3D](https://github.com/baaivision/Uni3D) | large-scale shape semantics and possible encoder initialization/teacher features | normalized open-world recognition embeddings contain trustworthy physical scale |
| [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet), [LaGeM](https://1zb.github.io/LaGeM/) | scalable vector-set geometry latents and implicit decoding | an attractive generated mesh is observation-faithful |
| [ShapeFormer](https://github.com/QhelDIV/ShapeFormer), [AutoSDF](https://arxiv.org/abs/2203.09516) | conditional distributions over completions from arbitrary partial evidence | one sampled completion is the true hidden geometry |
| [ConDor](https://github.com/brown-ivl/ConDor), [CASS](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Learning_Canonical_Shape_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2020_paper.pdf) | canonicalization and explicit shape/pose factorization | single-view RGB-D assumptions or normalized scale transfer directly to LiDAR tracks |
| [Motion2VecSets](https://github.com/VVeiCao/Motion2VecSets) | a shape latent plus temporal motion latent for sparse/noisy sequences | its non-rigid reconstruction target matches rigid box refinement |
| [Auto4D](https://arxiv.org/abs/2101.06586), [LabelFormer](https://proceedings.mlr.press/v229/yang23e.html), [DetZero](https://github.com/PJLab-ADG/DetZero) | one canonical rigid size, whole-track reasoning, and separated geometry/pose diagnostics | detector-specific residual heads or Waymo-shaped inputs provide sensor generality |
| [DTS](https://github.com/WoodwindHu/DTS), [LiDAR-CS](https://github.com/LiDAR-Perception/LiDAR-CS), [MS3D](https://github.com/darrenjkt/MS3D) | structured density variation, cross-sensor tests, and multi-source evidence | domain adaptation on one target is a generic foundation checkpoint |

Optional camera/language conditioning is outside the first exploration. It may
later help semantic or part understanding, but the generic LiDAR trajectory
model must first prove that geometry alone supports its claimed release scope.

## Stop conditions

- Do not scale a model that fails Stage 1 reconstruction or Stage 2
  factorization.
- Do not call a checkpoint generic based on random splits inside one dataset.
- Do not use production qualification data for model or threshold selection.
- Do not accept visually plausible completion without observed-point
  consistency and metric accuracy.
- Do not hide a failing axis, frame, class, range, sensor, or detector stratum
  inside an aggregate score.
- Do not start probabilistic generation merely because it is fashionable; it
  must improve ambiguity detection or final safe-release precision.
- Do not expose a development completion or size-only result as public
  `RefinementSuccess`.
