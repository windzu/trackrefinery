# Evaluation and Review Design

Status: framework and initial algorithm evidence traces implemented; correction
tolerances and later optimization diagnostics remain to be calibrated

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

- target, ambiguous, background, and ground point counts per frame;
- observed-side/view-direction coverage;
- registration residual and supported outer-envelope residual;
- free-space and ground-contact violations when the required provenance is
  available;
- leave-one-frame-out size and pose stability;
- supported sensor-dropout stability and numerical conditioning;
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
  aggregate.npz                # selected points in refined instance frame
  preview.html                 # self-contained or locally served viewer
  thumbnails/
    aggregate_top.png
    aggregate_side.png
    worst_frame_<id>.png
```

The aggregate point cloud is required because it directly exposes smearing,
incorrect alignment, missing surfaces, ground contamination, and neighboring
objects. Points should be colorized by frame/time, with coarse, refined, and
gold boxes independently toggleable.

A fixed aggregate alone is insufficient: pose errors can cancel or smear in
aggregation and hide which frame is wrong. The web viewer therefore provides:

1. per-frame mode with a timeline and coarse/refined/gold overlays;
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
the initial target/ambiguous/background/ground evidence masks defined by the
[deterministic geometric refinement design](geometric-refinement-v1.md). Later
optimization rounds will reuse the same trace contract for reassigned evidence.
