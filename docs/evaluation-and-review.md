# Evaluation and Review Design

Status: framework, legacy V1 previews, and V2 component/frame-role plus anchored
alignment previews implemented; canonical-size and fixed-size pose diagnostics
remain to be implemented

## What “good” means

A result is good only when its canonical size and every evaluated frame pose
are within the agreed correction tolerance. Improvement in average IoU is not
enough if the output still needs manual geometry edits.

Evaluation compares the same frozen input track before and after refinement:

```text
coarse detector/track boxes (baseline)
    versus
TrackRefinery canonical size + refined poses
    versus
physically separate reviewed gold target
```

## Quantitative measures

Canonical geometry:

- absolute and relative error for length, width, and height separately;
- maximum dimension error and volume error;
- exact temporal size variance, required to be zero for successful rigid
  results.

Per-frame pose:

- horizontal center error and vertical center error;
- yaw circular error and full rotation geodesic error where relevant;
- BEV IoU and 3D IoU;
- median, P95, and worst evaluated-frame error per track.

Evidence/robustness diagnostics:

- selected and rejected component point counts per frame;
- geometry, pose-only, and trajectory-only frame roles;
- selected-component alignment and temporal trajectory residuals;
- axis-wise aggregate spread and voxel concentration before/after;
- ground-contact consistency;
- geometry-frame leave-one-out size stability;
- change from the coarse initialization;
- runtime and peak memory.

Point residuals help diagnose a fit but cannot replace reviewed geometry: a box
can fit visible points while having an incorrect occluded extent.

## Track-level acceptance

Numeric tolerances must be chosen with the annotation owner using examples in
X-Points. The evaluator then defines one versioned predicate such as:

```text
canonical dimensions pass their per-axis tolerances
AND P95 pose errors pass their normal limits
AND every evaluated frame passes hard worst-frame limits
AND all invariant and finite-output checks pass
```

Report:

- strict track pass rate;
- count of improved, unchanged, and regressed tracks;
- success precision among tracks for which the backend returned success;
- insufficient-evidence rate;
- catastrophic-success count, where a reported success violates a hard limit;
- results stratified by class, distance, motion, point count, and visibility.

The primary product metric is strict successful-track precision. Coverage is
secondary: returning insufficient evidence is preferable to returning geometry
that a reviewer must correct.

## Human acceptance

Before a release is considered useful, perform a blinded A/B review of coarse
and refined outputs in X-Points. Record per track:

- no correction required;
- size correction required;
- pose correction required and affected frame count;
- refinement made the result worse;
- insufficient evidence / should not have reported success;
- review and correction time.

The final operational measure is correction time and correction count per Clip,
not only an offline metric.

## Development preview bundle

Every algorithm run should produce an immutable review bundle:

```text
review/<run_id>/<case_id>/
  result.json
  metrics.json                 # when a gold target is available
  aggregate.npz                # selected points in candidate/refined frame
  input_track_aggregate.npz    # same points aligned by frozen input-track poses
  gold_aggregate.npz           # same points aligned by gold poses; review only
  canonical_shape.npz          # registration-stage persistent evidence
  preview.html                 # self-contained or locally served viewer
  thumbnails/
    aggregate_top.png
    aggregate_side.png
    alignment_comparison_top.png
    alignment_comparison_side.png
    canonical_registration_top.png
    worst_frame_<id>.png
```

The aggregate point cloud is required because it directly exposes smearing,
incorrect alignment, missing surfaces, ground contamination, and neighboring
objects. Points should be colorized by frame/time, with coarse, refined, and
gold boxes independently toggleable.

When a separate reviewed target is available, the viewer must show both the
algorithm-aligned aggregate and an annotation-pose-aligned aggregate. They use
the same displayed point indices and differ only in the alignment pose, so the
comparison does not hide errors by changing the crop. Target poses remain in
the evaluation/review path and are never passed to the backend.

An algorithm bundle applies the same rule to its frozen model-track baseline:
`input_track_aggregate.npz` and `aggregate.npz` contain exactly the same point
indices and frame-index colors, transformed only by the input-track poses or
the algorithm poses. Fixed top and side A/B figures use shared axes. This is
the primary visual test for whether registration reduces rather than increases
smearing; comparisons across different instances or sampling budgets are not
valid evidence of improvement.

A fixed aggregate alone is insufficient: pose errors can cancel or smear in
aggregation and hide which frame is wrong. The web viewer therefore provides:

1. per-frame mode with a timeline and separately named
   coarse/registration-candidate/refined/gold overlays;
2. aggregate object-frame mode, colored by source frame;
3. selected-versus-rejected evidence display when a future algorithm supplies
   an evidence-trace sidecar; the framework viewer otherwise shows the local
   context selected only for visualization;
4. top/side/front orthographic views and free 3D navigation;
5. a metrics panel highlighting the worst frame and dimension deltas;
6. a small feedback form with `good`, `size wrong`, `pose wrong`, `point
   selection wrong`, and `insufficient evidence` outcomes.

The review viewer is a standalone development tool in TrackRefinery, not the
annotation product. It should open a generated bundle without X-4D. X-Points
remains the final Clip-level acceptance surface because it measures the actual
human correction workflow.

## Recommended delivery order

1. deterministic aggregate NPZ plus fixed orthographic thumbnails;
2. local static web viewer over the same review-bundle contract;
3. metrics and run-to-run comparison inside the viewer;
4. selected milestone candidates exported to X-Points for blind A/B review.

The fixed artifacts keep experiments reproducible and easy to diff. The web
viewer supplies the interaction needed for human diagnosis. They are two views
of one result bundle rather than competing solutions. The framework currently
implements both views, result/metric JSON, downloadable reviewer feedback, and
the legacy V1 target/ambiguous/background/ground traces. It also exports V1
provisional poses, a canonical point shape, and its trace-only envelope
candidate so the rejected behavior remains reproducible. V2 will replace those
algorithm diagnostics with selected components, frame roles, anchored
aggregation decisions, fixed-size poses, and explicit baseline-regression
checks as specified by the [V2 design](geometric-refinement-v2.md). Every bundle
names its data source so a generated fixture cannot be mistaken for a real
Clip. Candidate geometry is visibly marked as not released until the success
gate exists.

Complete synthetic regression runs additionally contain `suite.json` and
`index.html`. Their index exposes every case as a top-level tab, while each
case has separate tabs for algorithm aggregate, annotation aggregate,
canonical shape, current evidence, per-frame results, metrics, and
diagnostics. A failed or insufficient-evidence case remains visible; suite
generation must not select only favorable examples.

Real development review uses a different Clip-level catalog contract,
`trackrefinery-clip-review-suite-v1`. Each top-level tab is exactly one source
Clip, and that tab tiles every instance bundle from the Clip. Cards are ordered
with algorithm candidates first and then by observation count; selecting a card
opens its instance detail bundle. The catalog must not create one outer tab per
instance or hide short, failed, unsupported, or insufficient-evidence tracks.

Every catalog thumbnail is explicitly labeled as a multi-frame aggregate. Mode
badges, colored card borders, counts, and filters distinguish TrackRefinery
algorithm candidates, frozen inference-and-tracking baselines, and source
annotation references. Top and side thumbnails on one card must use the same
alignment source. An algorithm card spans the catalog width and leads with
same-instance top/side A/B figures before showing evidence classification and
the canonical shape. An unsuccessful algorithm outcome must be marked not
released and must not visually resemble an accepted annotation result. A
successful outcome remains a candidate for caller review; the catalog never
claims that it has been released.

During the dense-first MVP, algorithm cards also state whether the instance
passes the current dense-track gate. Dense-supported cards sort before
out-of-scope sparse cards inside each Clip, but the catalog continues to show
every instance; scope filtering must not hide failures or sparse evidence.

Catalog cards name their alignment source. A frozen model track may be shown
as a coarse-track baseline before refinement. Current source annotations may
be shown as an annotation-aligned reference, but are explicitly marked as not
reviewed gold and must not be passed to the refiner. Only a separately reviewed
target may be called gold or used for quantitative evaluation. A lightweight
`catalog` detail level writes fixed aggregate artifacts and a static detail
page for exhaustive inventory; selected algorithm cases retain the full
interactive bundle.
