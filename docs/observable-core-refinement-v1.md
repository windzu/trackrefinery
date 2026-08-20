# Observable-Core Refinement V1

Status: accepted deterministic MVP scope and public authority contract

## Decision

The first usable TrackRefinery backend will solve only the part of a track that
is directly and repeatedly supported by geometry. Given an associated track of
roughly thirty frames, it may use a dense central subset to estimate one
canonical size and refine poses for that subset. Sparse head/tail frames are
not required to pass and are not silently copied into the refined result.

This is a deliberate precision-first contract:

```text
full frame clouds + associated coarse track + exact frame poses
  -> deterministic evidence qualification
  -> connected observable core
  -> canonical aggregation and size stability gates
  -> fixed-size pose refinement on supported frames
  -> full success | partial success | insufficient evidence
```

The MVP is successful when the released core needs no human size or pose edit.
It does not claim amodal reconstruction of an object whose boundaries were
never observed, and it does not promise corrections for sparse track tails.

## Frame authority

Every input frame has exactly one result disposition:

- `geometry`: its evidence contributes to canonical geometry and its output
  pose is authoritative;
- `pose_only`: it does not determine size, but its pose is authoritatively
  refined against the accepted fixed canonical geometry; or
- `unsupported`: TrackRefinery publishes no pose for the frame and records one
  or more stable reason codes.

`success` retains the strict all-frame meaning. `partial_success` contains at
least one authoritative pose and at least one unsupported frame. Its refined
and unsupported lists are disjoint, preserve input order, and exactly partition
the input track. A caller may display or retain its original coarse box for an
unsupported frame, but must not attribute that box to TrackRefinery.

The canonical size belongs to the accepted object core, not to an individual
frame. Every materialized refined box uses exactly that size.

## Deterministic pipeline

### 1. Per-frame qualification

Use the full frame cloud and coarse box only as a localization seed. Select an
object component and compute auditable evidence measurements such as component
point count, spatial resolution, ground separation, coarse-envelope escape,
local support continuity, and selection stability under deterministic
resampling. Thresholds are versioned backend configuration, not caller input.

### 2. Observable-core selection

Find the strongest temporally connected run of qualified frames. Isolated good
frames do not bridge a weak interval. The core must meet minimum frame count,
total support, view diversity, and motion/pose consistency. If no component
passes, return `insufficient_evidence`.

### 3. Joint canonical aggregation

Transform selected components into a shared object coordinate system and
alternate robust frame alignment with canonical surface aggregation. Optimize
only observable degrees of freedom. Reject point clusters or frames whose
robust residual, overlap, or effect on aggregate sharpness is inconsistent
with the dominant track.

### 4. Canonical-size acceptance

Fit one upright cuboid from repeatable boundary support, not from the outermost
point. A size is publishable only when all required dimensions are stable
under:

- leave-one-frame-out and coherent temporal-subset recomputation;
- deterministic density/resolution perturbations;
- small localization/crop perturbations;
- robust estimator and support-threshold perturbations; and
- removal of any single candidate boundary cluster.

The accepted numerical tolerances are calibrated on physically separate
review targets. No learned confidence score or detector score substitutes for
these tests.

### 5. Fixed-size pose refinement

Freeze the canonical size. Refine each remaining candidate frame against the
canonical surface using bounded pose updates. Promote it to `pose_only` only
when its residual, overlap, correction magnitude, and local temporal
consistency pass. A failed frame becomes `unsupported`; it does not invalidate
an otherwise stable core unless its evidence contradicts object identity or
canonical size.

## Acceptance and evaluation

Size error is evaluated once per instance. Pose metrics and before/after
alignment are evaluated only on authoritative frames, using the same frame
subset and point indices for the coarse baseline and refined result. Reports
must also expose input, authoritative, and unsupported frame counts and IDs.

The primary qualification order is:

1. zero catastrophic released cores on the locked set;
2. `accept_as_is` precision for authoritative frames;
3. canonical-size accuracy and stability;
4. authoritative-frame coverage; and
5. whole-track coverage.

Increasing coverage may not weaken the first three gates.

The review aggregate for `partial_success` contains authoritative frames only.
Unsupported frames remain available in the per-frame timeline with their
coarse input and reason codes, but are not mixed into the refined aggregate.

## Relationship to learned and multimodal work

The object-centric foundation and vision-assisted direction remains valid as
a future coverage-expansion path for missing or unobservable geometry. It is
not required for the deterministic observable-core MVP and must not be used to
turn a plausible completion into a geometric fact. A future learned backend
uses the same frame-authority contract and must separately qualify any frame or
dimension it releases.

## Next implementation milestone

The next backend milestone is not another whole-track heuristic. It is a
traceable `ObservableCoreRefiner` that:

1. emits deterministic per-frame qualification diagnostics;
2. selects one connected core;
3. produces a stable canonical-size candidate with perturbation reports;
4. refines and classifies core/pose-only frames; and
5. returns `PartialRefinementSuccess` only after configured acceptance gates
   pass.

Until those gates are calibrated, experimental traces remain visible but the
backend returns `insufficient_evidence`.
