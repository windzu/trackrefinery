# Portable Data Contract v1

Status: implemented framework contract; geometric backend not yet release-complete

## Logical API

One refinement request identifies exactly one already-associated rigid object:

```text
refine(frame_clouds, instance_track)
  -> success | partial_success | insufficient_evidence
```

`frame_clouds` may be reused without copying across multiple calls. The library
receives full per-frame point clouds and owns target evidence selection; a
caller-supplied crop or crop margin is not part of the required contract.

## Coordinates and time

- All coordinates are right-handed Cartesian metres with `+Z` up.
- In a frame, `points_xyz` and `coarse_box` use that frame's declared
  `annotation_frame_id`.
- Every frame provides the complete rigid transform
  `T_world_from_annotation`; the world frame is stable across the sequence.
- Box centers are geometric centers and size order is
  `[length, width, height]`.
- Quaternion order is `[x, y, z, w]` and represents box-local to the containing
  frame.
- Frame and point timestamps retain exact integer nanoseconds. An algorithm
  must not assume a fixed frame rate.

For X-4D Dataset 0.17, `world` maps to the existing `clip_world` and the
per-frame annotation frame maps to `meta.annotation_frame_id`. The adapter uses
the Devkit's existing transforms; TrackRefinery does not introduce another pose
source.

## Frame cloud

Each immutable frame cloud contains:

- stable `frame_id` and exact `timestamp_ns`;
- `annotation_frame_id`;
- finite `float32 [N, 3]` points in that annotation frame;
- exact `T_world_from_annotation` as translation plus unit quaternion, or an
  equivalent validated SE(3) value;
- optional aligned point features with explicit names and units;
- optional per-point acquisition timestamps and sensor provenance;
- optional sensor origins in the annotation frame for visibility diagnostics.

Multi-LiDAR discovery, calibration, and fusion are upstream adapter work. The
X-4D adapter supplies the same all-current-LiDAR fused evidence used by the
detector. Historical sweeps are not required by v1.

## Instance track

An instance track contains:

- stable `track_id`;
- optional semantic `category` used only by algorithms that explicitly declare
  category priors;
- two or more ordered observations;
- for each observation, its `frame_id`, finite coarse 3D box, optional detector
  score, and observed/interpolated provenance.

The input observations are already associated to the same instance. Detection,
association, fragment reconciliation, and gap completion are outside the
library. V1 refines observed input rows and does not create new head/tail rows.

The coarse boxes are initialization and localization evidence, not targets.
TrackRefinery decides how large an evidence region to inspect and which points
support the object.

## Full and partial successful results

A full or partial successful rigid-object result contains:

- exactly one finite, positive `canonical_size_lwh`;
- one or more authoritative refined poses, expressed in the same annotation
  frame as each observation;
- a `geometry` or `pose_only` role for every authoritative pose;
- byte-equivalent canonical dimensions for every materialized output box;
- numerical diagnostics and evidence summaries sufficient to inspect why the
  fit succeeded.

`success` means every input frame has an authoritative refined pose in input
order. `partial_success` means the result has authority over only a supported
subset. It also contains every remaining input frame as `unsupported`, in
input order, with stable reason codes. The refined and unsupported lists are
disjoint and exactly partition the input track.

An unsupported frame has no TrackRefinery output pose. A downstream adapter
may retain the caller's coarse pose for continuity or review, but it must not
serialize that pose as a refined result. See the accepted
[observable-core refinement](observable-core-refinement-v1.md) scope.

The core result does not contain an X-4D annotation, release decision, instance
token, or model-service protocol object. Adapters construct those downstream.

## Insufficient evidence

No point-only method can guarantee physical dimensions from arbitrary sparse or
single-sided observations. When the backend cannot support its accuracy
contract, it returns a typed `insufficient_evidence` result containing stable
reason codes and diagnostics. It must not copy detector boxes, inject Label
Schema default dimensions, or mark a guess as successful.

## Development dataset layout

The on-disk benchmark representation avoids repeating full point clouds for
every instance:

```text
<bundle>/
  inference/                  # safe to mount in an algorithm process
    dataset.json
    sources/
      <sequence_id>/
        sequence.json
        frames/<frame_id>.npz
    inputs/
      <case_id>/track.json
  targets/                    # different root, evaluator process only
    targetset.json
    cases/<case_id>/target.json
  predictions/                # generated, normally ignored by git
    <run_id>/<case_id>.json
```

`track.json` references frame IDs from one source sequence. Dataset split roles
are assigned by physical sequence, never by track, so the same scene cannot
leak across train, calibration, and test.

Targets contain one reviewed canonical size and reviewed per-frame poses. A
target is not valid merely because it came from an annotation file: benchmark
tracks require explicit geometry-quality review.

## Input/target isolation

`InferenceDataset.open()` accepts only the `inference/` root and has no target
field, method, sibling-directory lookup, or target-path convention.
`TargetDataset.open()` is a separate evaluator-only type that accepts an
explicit `targets/` root. Production runs mount only `inference/`. The evaluator
joins predictions with targets after inference completes. Native Clip
annotations, GT-derived crops, and training databases are forbidden input
evidence.
