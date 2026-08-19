"""Deterministic visual review bundles for algorithm-development feedback."""

from __future__ import annotations

import html as html_module
import json
import threading
import webbrowser
from collections.abc import Mapping, Sequence
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from trackrefinery.contracts import (
    Box3D,
    Pose3D,
    RefinementCase,
    RefinementOutcome,
    RefinementSuccess,
)
from trackrefinery.evaluation import EvaluationReport
from trackrefinery.geometric.trace import (
    CanonicalShapeTrace,
    CuboidFitTrace,
    EvidenceState,
    GeometricRefinementTrace,
    validate_geometric_trace,
    write_geometric_trace,
)
from trackrefinery.geometry import (
    BOX_EDGE_INDICES,
    box_corners,
    compose_pose,
    inverse_transform_points,
)
from trackrefinery.refiner import validate_outcome
from trackrefinery.serde import outcome_to_dict
from trackrefinery.targets import GoldTarget

if TYPE_CHECKING:
    from matplotlib.axes import Axes

EVIDENCE_COLORS = {
    EvidenceState.TARGET: "#00c853",
    EvidenceState.AMBIGUOUS: "#ffab00",
    EvidenceState.BACKGROUND: "#78909c",
    EvidenceState.GROUND: "#8d6e63",
}

REVIEW_MODES = frozenset(
    {
        "algorithm_candidate",
        "model_candidate_baseline",
        "source_annotation_reference",
    }
)
REVIEW_DETAIL_LEVELS = frozenset({"catalog", "full"})


def build_review_bundle(
    case: RefinementCase,
    outcome: RefinementOutcome,
    output_dir: str | Path,
    *,
    target: GoldTarget | None = None,
    evaluation: EvaluationReport | None = None,
    trace: GeometricRefinementTrace | None = None,
    data_source: str = "not declared",
    review_mode: str = "algorithm_candidate",
    detail_level: str = "full",
    crop_scale: float = 1.8,
    max_points_per_frame: int = 8_000,
) -> Path:
    """Write one immutable, browser-viewable review bundle."""

    validate_outcome(case, outcome)
    if crop_scale <= 1.0:
        raise ValueError("crop_scale must be greater than 1")
    if max_points_per_frame <= 0:
        raise ValueError("max_points_per_frame must be positive")
    if target is not None and target.case_id != case.case_id:
        raise ValueError("target does not belong to the review case")
    if evaluation is not None and evaluation.case_id != case.case_id:
        raise ValueError("evaluation does not belong to the review case")
    if trace is not None:
        validate_geometric_trace(case, trace)
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("data_source must be a non-empty string")
    data_source = data_source.strip()
    if review_mode not in REVIEW_MODES:
        raise ValueError(f"review_mode must be one of {sorted(REVIEW_MODES)}")
    if detail_level not in REVIEW_DETAIL_LEVELS:
        raise ValueError(f"detail_level must be one of {sorted(REVIEW_DETAIL_LEVELS)}")

    output = Path(output_dir).resolve()
    thumbnails = output / "thumbnails"
    thumbnails.mkdir(parents=True, exist_ok=True)
    observations = {item.frame_id: item for item in case.track.observations}
    refined_poses = (
        {item.frame_id: item.pose for item in outcome.frame_poses}
        if isinstance(outcome, RefinementSuccess)
        else {}
    )
    target_poses = (
        {item.frame_id: item.pose for item in target.frame_poses}
        if target is not None
        else {}
    )

    aggregate_groups: list[NDArray[np.float32]] = []
    frame_indices: list[NDArray[np.int16]] = []
    gold_aggregate_groups: list[NDArray[np.float32]] = []
    gold_frame_indices: list[NDArray[np.int16]] = []
    preview_frames: list[dict[str, object]] = []
    evidence_states: list[NDArray[np.uint8]] = []
    trace_by_frame = (
        {frame.frame_id: frame for frame in trace.frames} if trace is not None else {}
    )
    cuboid_fit = None if trace is None else trace.cuboid_fit
    for frame_index, frame in enumerate(case.frames):
        observation = observations[frame.frame_id]
        coarse_box = observation.coarse_box
        refined_box = (
            Box3D(
                refined_poses[frame.frame_id].translation_xyz,
                outcome.canonical_size_lwh,
                refined_poses[frame.frame_id].orientation_xyzw,
            )
            if isinstance(outcome, RefinementSuccess)
            else None
        )
        gold_box = (
            Box3D(
                target_poses[frame.frame_id].translation_xyz,
                target.canonical_size_lwh,
                target_poses[frame.frame_id].orientation_xyzw,
            )
            if target is not None and frame.frame_id in target_poses
            else None
        )
        frame_trace = trace_by_frame.get(frame.frame_id)
        registration_box = None
        cuboid_candidate_box = None
        if (
            frame_trace is not None
            and frame_trace.registration is not None
            and frame_trace.registration.candidate_pose_annotation is not None
        ):
            candidate_pose = frame_trace.registration.candidate_pose_annotation
            registration_box = Box3D(
                candidate_pose.translation_xyz,
                coarse_box.size_lwh,
                candidate_pose.orientation_xyzw,
            )
            if (
                cuboid_fit is not None
                and cuboid_fit.status == "candidate"
                and cuboid_fit.canonical_size_lwh is not None
                and cuboid_fit.center_in_registration_xyz is not None
            ):
                cuboid_pose = compose_pose(
                    candidate_pose,
                    Pose3D(
                        cuboid_fit.center_in_registration_xyz,
                        (0.0, 0.0, 0.0, 1.0),
                    ),
                )
                cuboid_candidate_box = Box3D(
                    cuboid_pose.translation_xyz,
                    cuboid_fit.canonical_size_lwh,
                    cuboid_pose.orientation_xyzw,
                )
        if frame_trace is None:
            points = _preview_points(
                frame.points_xyz,
                tuple(
                    box
                    for box in (coarse_box, refined_box, gold_box)
                    if box is not None
                ),
                crop_scale,
                max_points_per_frame,
            )
            point_states = None
        else:
            positions = _trace_preview_positions(
                frame_trace.point_states, max_points_per_frame
            )
            indices = frame_trace.roi_point_indices[positions]
            points = frame.points_xyz[indices]
            point_states = frame_trace.point_states[positions]
        alignment_pose = (
            refined_box.pose
            if refined_box is not None
            else (
                registration_box.pose
                if cuboid_candidate_box is None and registration_box is not None
                else (
                    cuboid_candidate_box.pose
                    if cuboid_candidate_box is not None
                    else coarse_box.pose
                )
            )
        )
        local = inverse_transform_points(points, alignment_pose).astype(np.float32)
        aggregate_groups.append(local)
        frame_indices.append(np.full(len(local), frame_index, dtype=np.int16))
        if gold_box is not None:
            gold_local = inverse_transform_points(points, gold_box.pose).astype(
                np.float32
            )
            gold_aggregate_groups.append(gold_local)
            gold_frame_indices.append(
                np.full(len(gold_local), frame_index, dtype=np.int16)
            )
        if point_states is not None:
            evidence_states.append(point_states)
        preview_frames.append(
            {
                "frame_id": frame.frame_id,
                "points_xyz": points,
                "coarse_box": coarse_box,
                "refined_box": refined_box,
                "registration_box": registration_box,
                "cuboid_candidate_box": cuboid_candidate_box,
                "gold_box": gold_box,
                "point_states": point_states,
            }
        )

    aggregate = np.concatenate(aggregate_groups)
    aggregate_frame_index = np.concatenate(frame_indices)
    gold_aggregate = (
        np.concatenate(gold_aggregate_groups) if gold_aggregate_groups else None
    )
    gold_aggregate_frame_index = (
        np.concatenate(gold_frame_indices) if gold_frame_indices else None
    )
    aggregate_evidence_state = (
        np.concatenate(evidence_states) if trace is not None else None
    )
    aggregate_payload: dict[str, NDArray[np.generic]] = {
        "points_xyz": aggregate,
        "frame_index": aggregate_frame_index,
        "frame_ids": np.asarray([frame.frame_id for frame in case.frames]),
    }
    if aggregate_evidence_state is not None:
        aggregate_payload["evidence_state"] = aggregate_evidence_state
    np.savez_compressed(output / "aggregate.npz", **aggregate_payload)
    if gold_aggregate is not None and gold_aggregate_frame_index is not None:
        gold_payload = {
            "points_xyz": gold_aggregate,
            "frame_index": gold_aggregate_frame_index,
            "frame_ids": np.asarray([frame.frame_id for frame in case.frames]),
        }
        if aggregate_evidence_state is not None:
            gold_payload["evidence_state"] = aggregate_evidence_state
        np.savez_compressed(output / "gold_aggregate.npz", **gold_payload)
    canonical_shape = None if trace is None else trace.canonical_shape
    if canonical_shape is not None:
        np.savez_compressed(
            output / "canonical_shape.npz",
            points_xyz=canonical_shape.points_xyz,
            frame_support_count=canonical_shape.frame_support_count,
        )
    (output / "result.json").write_text(
        json.dumps(outcome_to_dict(case.case_id, outcome), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if evaluation is not None:
        evaluation.write_json(output / "metrics.json")
    if trace is not None:
        write_geometric_trace(output, trace)
    bundle_manifest = {
        "contract_version": "trackrefinery-review-bundle-v1",
        "case_id": case.case_id,
        "track_id": case.track.track_id,
        "sequence_id": case.track.sequence_id,
        "category": case.track.category,
        "frame_count": len(case.frames),
        "aggregate_point_count": len(aggregate),
        "outcome_status": outcome.status,
        "data_source": data_source,
        "review_mode": review_mode,
        "detail_level": detail_level,
        "frame_ids": [frame.frame_id for frame in case.frames],
        "crop_scale": crop_scale,
        "max_points_per_frame": max_points_per_frame,
        "has_gold_target": target is not None,
        "has_gold_aligned_aggregate": gold_aggregate is not None,
        "has_metrics": evaluation is not None,
        "has_evidence_trace": trace is not None,
        "has_registration_trace": canonical_shape is not None,
        "has_cuboid_candidate": (
            cuboid_fit is not None and cuboid_fit.status == "candidate"
        ),
        "cuboid_candidate_size_lwh": (
            None if cuboid_fit is None else cuboid_fit.canonical_size_lwh
        ),
        "evidence_trace_path": "evidence_trace.json" if trace is not None else None,
        "evidence_masks_path": "evidence_masks.npz" if trace is not None else None,
        "canonical_shape_path": (
            "canonical_shape.npz" if canonical_shape is not None else None
        ),
        "gold_aggregate_path": (
            "gold_aggregate.npz" if gold_aggregate is not None else None
        ),
    }
    (output / "bundle.json").write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if detail_level == "catalog":
        _write_catalog_thumbnails(
            thumbnails,
            aggregate,
            aggregate_frame_index,
            preview_frames,
            outcome,
            cuboid_fit,
        )
        _write_catalog_html(
            output / "preview.html",
            case,
            outcome,
            data_source,
            review_mode,
        )
    else:
        _write_thumbnails(
            thumbnails,
            aggregate,
            aggregate_frame_index,
            gold_aggregate,
            gold_aggregate_frame_index,
            aggregate_evidence_state,
            preview_frames,
            outcome,
            target,
            evaluation,
            canonical_shape,
            cuboid_fit,
        )
        _write_html(
            output / "preview.html",
            case,
            outcome,
            aggregate,
            aggregate_frame_index,
            gold_aggregate,
            gold_aggregate_frame_index,
            aggregate_evidence_state,
            preview_frames,
            target,
            evaluation,
            trace,
            data_source,
            review_mode,
        )
    return output


def build_clip_review_suite(
    output_dir: str | Path,
    clips: Mapping[str, Sequence[str | Path]],
    *,
    title: str = "TrackRefinery real Clip review",
) -> Path:
    """Tile single-instance bundles under one top-level tab per source Clip."""

    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    if not clips:
        raise ValueError("clip review suite requires at least one Clip")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    clip_rows: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    for clip_id, bundle_dirs in clips.items():
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError("clip_id must be a non-empty string")
        instances = [
            _review_bundle_index_row(output, bundle_dir, seen_case_ids)
            for bundle_dir in bundle_dirs
        ]
        if not instances:
            raise ValueError(f"Clip {clip_id!r} requires at least one instance")
        instances.sort(
            key=lambda row: (
                -int(row.get("frame_count") or 0),
                str(row.get("category") or ""),
                str(row["case_id"]),
            )
        )
        clip_rows.append(
            {
                "clip_id": clip_id.strip(),
                "instance_count": len(instances),
                "instances": instances,
            }
        )

    suite_manifest = {
        "contract_version": "trackrefinery-clip-review-suite-v1",
        "title": title.strip(),
        "clips": clip_rows,
    }
    (output / "clip-suite.json").write_text(
        json.dumps(suite_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(
        _clip_suite_html(title.strip(), clip_rows), encoding="utf-8"
    )
    return output


def _review_bundle_index_row(
    output: Path,
    bundle_dir: str | Path,
    seen_case_ids: set[str],
) -> dict[str, object]:
    bundle = Path(bundle_dir).resolve()
    try:
        relative = bundle.relative_to(output)
    except ValueError as error:
        raise ValueError("review suite bundles must be inside output_dir") from error
    preview = bundle / "preview.html"
    manifest_path = bundle / "bundle.json"
    top = bundle / "thumbnails" / "aggregate_top.png"
    side = bundle / "thumbnails" / "aggregate_side.png"
    if not all(path.is_file() for path in (preview, manifest_path, top, side)):
        raise ValueError(f"{bundle} is not a complete review bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain an object")
    case_id = manifest.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"{manifest_path} has no case_id")
    if case_id in seen_case_ids:
        raise ValueError(f"duplicate review case_id: {case_id}")
    seen_case_ids.add(case_id)
    canonical_top = bundle / "thumbnails" / "canonical_registration_top.png"
    review_mode = manifest.get("review_mode", "algorithm_candidate")
    if canonical_top.is_file():
        top_path = canonical_top
        top_label = "Canonical registered aggregate"
    elif review_mode == "source_annotation_reference":
        top_path = top
        top_label = "Annotation-aligned aggregate"
    else:
        top_path = top
        top_label = "Coarse-track-aligned aggregate"
    return {
        "case_id": case_id,
        "track_id": manifest.get("track_id"),
        "category": manifest.get("category"),
        "frame_count": manifest.get("frame_count"),
        "aggregate_point_count": manifest.get("aggregate_point_count"),
        "outcome_status": manifest.get("outcome_status"),
        "review_mode": review_mode,
        "detail_level": manifest.get("detail_level", "full"),
        "has_gold_aligned_aggregate": manifest.get("has_gold_aligned_aggregate", False),
        "cuboid_candidate_size_lwh": manifest.get("cuboid_candidate_size_lwh"),
        "preview_path": (relative / "preview.html").as_posix(),
        "aggregate_top_path": top_path.relative_to(output).as_posix(),
        "aggregate_top_label": top_label,
        "aggregate_side_path": (
            relative / "thumbnails" / "aggregate_side.png"
        ).as_posix(),
    }


def _clip_suite_html(title: str, clips: list[dict[str, object]]) -> str:
    buttons: list[str] = []
    panels: list[str] = []
    for clip_index, clip in enumerate(clips):
        clip_id = str(clip["clip_id"])
        instances = clip["instances"]
        if not isinstance(instances, list):
            raise ValueError("Clip instances must be a list")
        active = " active" if clip_index == 0 else ""
        buttons.append(
            f'<button class="clip-tab{active}" '
            f'onclick="showClip({clip_index}, this)">'
            f"{html_module.escape(clip_id)}"
            f"<small>{len(instances)} instances</small></button>"
        )
        cards = [_clip_instance_card(instance) for instance in instances]
        panels.append(
            f'<section id="clip-{clip_index}" class="clip-panel{active}">'
            f'<div class="clip-summary"><strong>{html_module.escape(clip_id)}</strong>'
            f"<span>{len(instances)} / {len(instances)} instances shown</span></div>"
            f'<div class="instance-grid">{"".join(cards)}</div></section>'
        )
    title_display = html_module.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title_display}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui; margin: 0; background: #0b1120; color: #e5e7eb; }}
    header {{ position: sticky; top: 0; z-index: 5; padding: 14px 18px 8px;
      background: rgba(11,17,32,.97); border-bottom: 1px solid #263349; }}
    h1 {{ margin: 0 0 12px; font-size: 22px; }}
    nav {{ display: flex; gap: 8px; overflow-x: auto; padding-bottom: 8px; }}
    .clip-tab {{ flex: 0 0 auto; min-width: 220px; padding: 9px 12px;
      border: 1px solid #475569; border-radius: 8px; background: #172033;
      color: #e5e7eb; cursor: pointer; text-align: left; }}
    .clip-tab small, .instance-head small {{ display: block; color: #94a3b8;
      margin-top: 3px; overflow: hidden; text-overflow: ellipsis; }}
    .clip-tab.active {{ background: #1d4ed8; border-color: #60a5fa; }}
    main {{ padding: 16px 18px 32px; }}
    .clip-panel {{ display: none; }} .clip-panel.active {{ display: block; }}
    .clip-summary {{ display: flex; justify-content: space-between; gap: 12px;
      margin-bottom: 14px; color: #cbd5e1; }}
    .instance-grid {{ display: grid; grid-template-columns:
      repeat(auto-fill, minmax(420px, 1fr)); gap: 14px; }}
    .instance-card {{ overflow: hidden; border: 1px solid #334155;
      border-radius: 12px; background: #111827; cursor: pointer; padding: 12px; }}
    .instance-card:hover {{ border-color: #60a5fa; transform: translateY(-1px); }}
    .instance-head {{ display: flex; justify-content: space-between; gap: 12px;
      margin-bottom: 8px; }}
    .status {{ color: #bfdbfe; font-size: 12px; }}
    .views {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
    figure {{ margin: 0; background: white; border-radius: 7px; overflow: hidden; }}
    img {{ display: block; width: 100%; aspect-ratio: 8/5; object-fit: contain; }}
    figcaption {{ color: #334155; font-size: 11px; padding: 3px 7px 5px; }}
    .instance-card p {{ margin: 7px 0 0; color: #94a3b8; font-size: 12px; }}
    .instance-card .mode {{ color: #fbbf24; }}
    dialog {{ width: min(1500px, 96vw); height: 94vh; padding: 0;
      border: 1px solid #475569; border-radius: 10px; background: #111827; }}
    dialog::backdrop {{ background: rgba(0,0,0,.75); }}
    .dialog-head {{ display: flex; justify-content: space-between; align-items: center;
      height: 48px; padding: 0 14px; color: #e5e7eb; }}
    .dialog-head button {{ padding: 6px 12px; }}
    iframe {{ width: 100%; height: calc(94vh - 48px); border: 0; background: #111827; }}
  </style>
</head>
<body>
  <header><h1>{title_display}</h1><nav>{"".join(buttons)}</nav></header>
  <main>{"".join(panels)}</main>
  <dialog id="instance-dialog"><div class="dialog-head">
    <strong id="dialog-title">Instance</strong>
    <button onclick="closeInstance()">Close</button></div>
    <iframe id="instance-frame" title="Instance review"></iframe></dialog>
  <script>
  function showClip(index, button) {{
    document.querySelectorAll('.clip-panel').forEach(
      element => element.classList.remove('active'));
    document.querySelectorAll('.clip-tab').forEach(
      element => element.classList.remove('active'));
    document.getElementById('clip-' + index).classList.add('active');
    button.classList.add('active');
  }}
  function openInstance(path, caseId) {{
    document.getElementById('dialog-title').textContent = caseId;
    document.getElementById('instance-frame').src = path;
    document.getElementById('instance-dialog').showModal();
  }}
  function closeInstance() {{
    document.getElementById('instance-dialog').close();
    document.getElementById('instance-frame').src = 'about:blank';
  }}
  </script>
</body>
</html>"""


def _clip_instance_card(instance: object) -> str:
    if not isinstance(instance, dict):
        raise ValueError("instance index row must be an object")
    case_id = str(instance["case_id"])
    track_id = str(instance.get("track_id") or "unknown")
    category = str(instance.get("category") or "unknown")
    frame_count = html_module.escape(str(instance.get("frame_count") or "?"))
    point_count = html_module.escape(str(instance.get("aggregate_point_count") or "?"))
    status = str(instance.get("outcome_status") or "unknown")
    mode = str(instance.get("review_mode") or "algorithm_candidate")
    if mode == "source_annotation_reference":
        mode_label = "source annotation reference · not gold · not refined"
    elif mode == "model_candidate_baseline":
        mode_label = "model track baseline · refinement not run"
    else:
        mode_label = "model track · TrackRefinery development result"
    candidate = instance.get("cuboid_candidate_size_lwh")
    size_label = "no candidate size"
    if isinstance(candidate, (list, tuple)) and len(candidate) == 3:
        size_label = " x ".join(f"{float(value):.2f}" for value in candidate)
        size_label += " m"
    preview = html_module.escape(str(instance["preview_path"]), quote=True)
    top = html_module.escape(str(instance["aggregate_top_path"]), quote=True)
    side = html_module.escape(str(instance["aggregate_side_path"]), quote=True)
    top_label = html_module.escape(
        str(instance.get("aggregate_top_label") or "Top aggregate")
    )
    return (
        '<article class="instance-card" '
        f'onclick="openInstance({html_module.escape(json.dumps(preview), quote=True)}, '
        f'{html_module.escape(json.dumps(case_id), quote=True)})">'
        '<div class="instance-head">'
        f"<div><strong>{html_module.escape(category)}</strong>"
        f"<small>{html_module.escape(track_id)}</small></div>"
        f'<span class="status">{html_module.escape(status)}</span></div>'
        '<div class="views">'
        f'<figure><img loading="lazy" src="{top}" alt="{top_label}">'
        f"<figcaption>{top_label}</figcaption></figure>"
        f'<figure><img loading="lazy" src="{side}" alt="Side aggregate">'
        "<figcaption>Side aggregate</figcaption></figure></div>"
        f'<p class="mode">{html_module.escape(mode_label)}</p>'
        f"<p>{frame_count} frames · {point_count} displayed points</p>"
        f"<p>{html_module.escape(size_label)}</p></article>"
    )


def build_review_suite(
    output_dir: str | Path,
    bundle_dirs: list[str | Path] | tuple[str | Path, ...],
    *,
    title: str = "TrackRefinery review suite",
) -> Path:
    """Write a tabbed index for case bundles contained under one suite root."""

    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    for bundle_dir in bundle_dirs:
        bundle = Path(bundle_dir).resolve()
        try:
            bundle.relative_to(output)
        except ValueError as error:
            raise ValueError(
                "review suite bundles must be inside output_dir"
            ) from error
        preview = bundle / "preview.html"
        manifest_path = bundle / "bundle.json"
        if not preview.is_file() or not manifest_path.is_file():
            raise ValueError(f"{bundle} is not a review bundle")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"{manifest_path} must contain an object")
        case_id = manifest.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{manifest_path} has no case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate review case_id: {case_id}")
        seen_case_ids.add(case_id)
        rows.append(
            {
                "case_id": case_id,
                "track_id": manifest.get("track_id"),
                "outcome_status": manifest.get("outcome_status"),
                "data_source": manifest.get("data_source"),
                "has_gold_aligned_aggregate": manifest.get(
                    "has_gold_aligned_aggregate", False
                ),
                "cuboid_candidate_size_lwh": manifest.get("cuboid_candidate_size_lwh"),
                "preview_path": preview.relative_to(output).as_posix(),
            }
        )
    if not rows:
        raise ValueError("review suite requires at least one case bundle")

    suite_manifest = {
        "contract_version": "trackrefinery-review-suite-v1",
        "title": title.strip(),
        "cases": rows,
    }
    (output / "suite.json").write_text(
        json.dumps(suite_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    buttons = []
    panels = []
    for index, row in enumerate(rows):
        case_id = str(row["case_id"])
        case_label = html_module.escape(case_id)
        status = html_module.escape(str(row["outcome_status"]))
        preview_path = html_module.escape(str(row["preview_path"]), quote=True)
        active = " active" if index == 0 else ""
        buttons.append(
            f'<button class="case-tab{active}" onclick="showCase({index}, this)">'
            f"{case_label}<small>{status}</small></button>"
        )
        src = f' src="{preview_path}"' if index == 0 else ""
        panels.append(
            f'<div id="case-{index}" class="case-panel{active}">'
            f'<iframe data-src="{preview_path}"{src} title="{case_label}"></iframe>'
            "</div>"
        )
    title_display = html_module.escape(title.strip())
    suite_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title_display}</title>
  <style>
    body {{ font-family: system-ui; margin: 0; background: #0f172a; color: #e5e7eb; }}
    header {{ padding: 14px 18px 8px; }}
    h1 {{ margin: 0 0 12px; font-size: 22px; }}
    nav {{ display: flex; gap: 8px; overflow-x: auto; padding-bottom: 8px; }}
    .case-tab {{ flex: 0 0 auto; min-width: 155px; padding: 9px 12px;
      border: 1px solid #475569; border-radius: 8px; background: #1e293b;
      color: #e5e7eb; cursor: pointer; text-align: left; }}
    .case-tab small {{ display: block; color: #94a3b8; margin-top: 3px; }}
    .case-tab.active {{ background: #1d4ed8; border-color: #60a5fa; }}
    .case-tab.active small {{ color: #dbeafe; }}
    .case-panel {{ display: none; height: calc(100vh - 116px); }}
    .case-panel.active {{ display: block; }}
    iframe {{ width: 100%; height: 100%; border: 0; background: #111827; }}
  </style>
</head>
<body>
  <header><h1>{title_display}</h1><nav>{"".join(buttons)}</nav></header>
  {"".join(panels)}
  <script>
  function showCase(index, button) {{
    document.querySelectorAll('.case-panel').forEach(
      element => element.classList.remove('active')
    );
    document.querySelectorAll('.case-tab').forEach(
      element => element.classList.remove('active')
    );
    const panel = document.getElementById('case-' + index);
    const frame = panel.querySelector('iframe');
    if (!frame.getAttribute('src')) frame.setAttribute('src', frame.dataset.src);
    panel.classList.add('active');
    button.classList.add('active');
  }}
  </script>
</body>
</html>"""
    (output / "index.html").write_text(suite_html, encoding="utf-8")
    return output


def serve_review_bundle(
    bundle_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = False,
) -> None:
    """Serve a generated case bundle or suite without X-4D or X-Points."""

    root = Path(bundle_dir).resolve()
    entrypoint = "preview.html" if (root / "preview.html").is_file() else "index.html"
    if not (root / entrypoint).is_file():
        raise ValueError(f"{root} is not a review bundle or suite")

    class BundleHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(root), **kwargs)

    server = ThreadingHTTPServer((host, port), BundleHandler)
    address, bound_port = server.server_address[:2]
    url = f"http://{address}:{bound_port}/{entrypoint}"
    print(f"TrackRefinery review: {url}")
    if open_browser:
        threading.Timer(0.2, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _preview_points(
    points: NDArray[np.float32],
    boxes: tuple[Box3D, ...],
    scale: float,
    maximum: int,
) -> NDArray[np.float32]:
    mask = np.zeros(len(points), dtype=bool)
    for box in boxes:
        expanded = Box3D(
            center=box.center,
            size_lwh=tuple(scale * value for value in box.size_lwh),
            orientation_xyzw=box.orientation_xyzw,
        )
        local = inverse_transform_points(points, expanded.pose)
        mask |= np.all(np.abs(local) <= np.asarray(expanded.size_lwh) / 2, axis=1)
    selected = points[mask]
    if len(selected) > maximum:
        indices = np.linspace(0, len(selected) - 1, maximum, dtype=np.int64)
        selected = selected[indices]
    return selected


def _trace_preview_positions(
    states: NDArray[np.uint8], maximum: int
) -> NDArray[np.int64]:
    """Deterministically retain every evidence class in a bounded preview."""

    if len(states) <= maximum:
        return np.arange(len(states), dtype=np.int64)
    chosen = np.zeros(len(states), dtype=bool)
    nonempty = [state for state in EvidenceState if np.any(states == state.value)]
    quota = max(1, maximum // len(nonempty))
    for state in nonempty:
        positions = np.flatnonzero(states == state.value)
        take = min(len(positions), quota)
        selected = np.linspace(0, len(positions) - 1, take, dtype=np.int64)
        chosen[positions[selected]] = True
    current = int(np.count_nonzero(chosen))
    if current < maximum:
        remaining = np.flatnonzero(~chosen)
        take = min(len(remaining), maximum - current)
        selected = np.linspace(0, len(remaining) - 1, take, dtype=np.int64)
        chosen[remaining[selected]] = True
    positions = np.flatnonzero(chosen)
    if len(positions) > maximum:
        selected = np.linspace(0, len(positions) - 1, maximum, dtype=np.int64)
        positions = positions[selected]
    return positions.astype(np.int64, copy=False)


def _write_catalog_thumbnails(
    output: Path,
    aggregate: NDArray[np.float32],
    frame_index: NDArray[np.int16],
    preview_frames: list[dict[str, object]],
    outcome: RefinementOutcome,
    cuboid_fit: CuboidFitTrace | None,
) -> None:
    """Render only the two fixed views needed by the all-instance catalog."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "review bundle rendering requires 'pip install trackrefinery[review]'"
        ) from error
    size, result_label, result_color = _review_display_geometry(
        preview_frames, outcome, cuboid_fit
    )
    for name, first, second, x_label, y_label in (
        ("aggregate_top.png", 0, 1, "X (m)", "Y (m)"),
        ("aggregate_side.png", 0, 2, "X (m)", "Z (m)"),
    ):
        figure, axis = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
        axis.scatter(
            aggregate[:, first],
            aggregate[:, second],
            c=frame_index,
            s=0.8,
            cmap="turbo",
            alpha=0.58,
        )
        _plot_box_projection(
            axis,
            Box3D((0.0, 0.0, 0.0), size, (0.0, 0.0, 0.0, 1.0)),
            first,
            second,
            result_color,
            result_label,
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="best", fontsize=7)
        figure.savefig(output / name, dpi=110)
        plt.close(figure)


def _review_display_geometry(
    preview_frames: list[dict[str, object]],
    outcome: RefinementOutcome,
    cuboid_fit: CuboidFitTrace | None,
) -> tuple[tuple[float, float, float], str, str]:
    if isinstance(outcome, RefinementSuccess):
        return outcome.canonical_size_lwh, "result", "#00c853"
    if (
        cuboid_fit is not None
        and cuboid_fit.status == "candidate"
        and cuboid_fit.canonical_size_lwh is not None
    ):
        return cuboid_fit.canonical_size_lwh, "cuboid candidate", "#d500f9"
    coarse_size = tuple(
        np.median(
            [
                item["coarse_box"].size_lwh
                for item in preview_frames
                if isinstance(item["coarse_box"], Box3D)
            ],
            axis=0,
        )
    )
    return coarse_size, "input box", "#7c4dff"


def _write_catalog_html(
    path: Path,
    case: RefinementCase,
    outcome: RefinementOutcome,
    data_source: str,
    review_mode: str,
) -> None:
    mode_label = {
        "algorithm_candidate": "TrackRefinery development result",
        "model_candidate_baseline": "Model candidate baseline; refinement not run",
        "source_annotation_reference": (
            "Source annotation reference; not reviewed gold; refinement not run"
        ),
    }[review_mode]
    path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html_module.escape(case.case_id)}</title>
<style>
body {{ font-family: system-ui; margin: 0; background: #111827; color: #e5e7eb; }}
main {{ max-width: 1400px; margin: auto; padding: 18px; }}
.views {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
figure {{ margin: 0; padding: 8px; border-radius: 10px;
  background: white; color: #111827; }}
img {{ width: 100%; display: block; }}
.notice {{ color: #fbbf24; }}
</style></head><body><main>
<h1>{html_module.escape(case.case_id)}</h1>
<p>{html_module.escape(case.track.category or "unknown")} · {len(case.frames)} frames ·
{html_module.escape(outcome.status)}</p>
<p class="notice">{html_module.escape(mode_label)}</p>
<p>Data source: {html_module.escape(data_source)}</p>
<div class="views">
<figure><img src="thumbnails/aggregate_top.png">
<figcaption>Top aggregate</figcaption></figure>
<figure><img src="thumbnails/aggregate_side.png">
<figcaption>Side aggregate</figcaption></figure>
</div></main></body></html>""",
        encoding="utf-8",
    )


def _write_thumbnails(
    output: Path,
    aggregate: NDArray[np.float32],
    frame_index: NDArray[np.int16],
    gold_aggregate: NDArray[np.float32] | None,
    gold_frame_index: NDArray[np.int16] | None,
    evidence_state: NDArray[np.uint8] | None,
    preview_frames: list[dict[str, object]],
    outcome: RefinementOutcome,
    target: GoldTarget | None,
    evaluation: EvaluationReport | None,
    canonical_shape: CanonicalShapeTrace | None,
    cuboid_fit: CuboidFitTrace | None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "review bundle rendering requires 'pip install trackrefinery[review]'"
        ) from error

    for name, axes in {
        "aggregate_top.png": (0, 1, "X (m)", "Y (m)"),
        "aggregate_side.png": (0, 2, "X (m)", "Z (m)"),
        "aggregate_front.png": (1, 2, "Y (m)", "Z (m)"),
    }.items():
        first, second, x_label, y_label = axes
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.scatter(
            aggregate[:, first],
            aggregate[:, second],
            c=frame_index,
            s=1,
            cmap="turbo",
            alpha=0.55,
        )
        size = (
            outcome.canonical_size_lwh
            if isinstance(outcome, RefinementSuccess)
            else (
                cuboid_fit.canonical_size_lwh
                if cuboid_fit is not None
                and cuboid_fit.status == "candidate"
                and cuboid_fit.canonical_size_lwh is not None
                else tuple(
                    np.median(
                        [
                            item["coarse_box"].size_lwh
                            for item in preview_frames
                            if isinstance(item["coarse_box"], Box3D)
                        ],
                        axis=0,
                    )
                )
            )
        )
        if isinstance(outcome, RefinementSuccess):
            result_label = "result"
            result_color = "#00c853"
        elif cuboid_fit is not None and cuboid_fit.status == "candidate":
            result_label = "cuboid candidate (not released)"
            result_color = "#d500f9"
        else:
            result_label = "coarse median"
            result_color = "#7c4dff"
        _plot_box_projection(
            axis,
            Box3D((0.0, 0.0, 0.0), size, (0.0, 0.0, 0.0, 1.0)),
            first,
            second,
            result_color,
            result_label,
        )
        if target is not None:
            _plot_box_projection(
                axis,
                Box3D((0.0, 0.0, 0.0), target.canonical_size_lwh, (0.0, 0.0, 0.0, 1.0)),
                first,
                second,
                "#00b8d4",
                "gold size",
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="best")
        figure.savefig(output / name, dpi=140)
        plt.close(figure)

    if (
        gold_aggregate is not None
        and gold_frame_index is not None
        and target is not None
    ):
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.scatter(
            gold_aggregate[:, 0],
            gold_aggregate[:, 1],
            c=gold_frame_index,
            s=1,
            cmap="turbo",
            alpha=0.55,
        )
        _plot_box_projection(
            axis,
            Box3D(
                (0.0, 0.0, 0.0),
                target.canonical_size_lwh,
                (0.0, 0.0, 0.0, 1.0),
            ),
            0,
            1,
            "#00b8d4",
            "gold size",
        )
        if cuboid_fit is not None and cuboid_fit.canonical_size_lwh is not None:
            _plot_box_projection(
                axis,
                Box3D(
                    (0.0, 0.0, 0.0),
                    cuboid_fit.canonical_size_lwh,
                    (0.0, 0.0, 0.0, 1.0),
                ),
                0,
                1,
                "#d500f9",
                "cuboid candidate size",
            )
        axis.set_title("Annotation-pose-aligned aggregate")
        axis.set_xlabel("gold object X (m)")
        axis.set_ylabel("gold object Y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="best")
        figure.savefig(output / "gold_aggregate_top.png", dpi=140)
        plt.close(figure)

    if canonical_shape is not None:
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        scatter = axis.scatter(
            canonical_shape.points_xyz[:, 0],
            canonical_shape.points_xyz[:, 1],
            c=canonical_shape.frame_support_count,
            s=2,
            cmap="viridis",
            alpha=0.75,
        )
        figure.colorbar(scatter, ax=axis, label="supporting frames")
        axis.set_title("Canonical shape after alternating registration")
        axis.set_xlabel("canonical X (m)")
        axis.set_ylabel("canonical Y (m)")
        axis.set_aspect("equal", adjustable="box")
        if (
            cuboid_fit is not None
            and cuboid_fit.status == "candidate"
            and cuboid_fit.canonical_size_lwh is not None
            and cuboid_fit.center_in_registration_xyz is not None
        ):
            _plot_box_projection(
                axis,
                Box3D(
                    cuboid_fit.center_in_registration_xyz,
                    cuboid_fit.canonical_size_lwh,
                    (0.0, 0.0, 0.0, 1.0),
                ),
                0,
                1,
                "#d500f9",
                "visible-envelope candidate",
            )
            axis.legend(loc="best")
        figure.savefig(output / "canonical_registration_top.png", dpi=140)
        plt.close(figure)

    if evidence_state is not None:
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        _scatter_evidence_2d(axis, aggregate, evidence_state, 0, 1)
        _plot_box_projection(
            axis,
            Box3D((0.0, 0.0, 0.0), size, (0.0, 0.0, 0.0, 1.0)),
            0,
            1,
            result_color,
            result_label,
        )
        axis.set_title("Current evidence classification")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="best")
        figure.savefig(output / "aggregate_evidence_top.png", dpi=140)
        plt.close(figure)

    worst_frame_id = _worst_frame_id(evaluation, preview_frames)
    frame = next(item for item in preview_frames if item["frame_id"] == worst_frame_id)
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    points = frame["points_xyz"]
    point_states = frame["point_states"]
    if isinstance(point_states, np.ndarray):
        _scatter_evidence_2d(axis, points, point_states, 0, 1)
    else:
        axis.scatter(points[:, 0], points[:, 1], s=1, c="#9e9e9e", alpha=0.65)
    for key, color, label in (
        ("coarse_box", "#ffab00", "coarse"),
        ("registration_box", "#7c4dff", "registration candidate"),
        ("cuboid_candidate_box", "#d500f9", "cuboid candidate"),
        ("refined_box", "#00c853", "refined"),
        ("gold_box", "#00b8d4", "gold"),
    ):
        box = frame[key]
        if isinstance(box, Box3D):
            _plot_box_projection(axis, box, 0, 1, color, label)
    axis.set_title(f"Frame {worst_frame_id}")
    axis.set_xlabel("annotation X (m)")
    axis.set_ylabel("annotation Y (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best")
    figure.savefig(output / f"worst_frame_{worst_frame_id}.png", dpi=140)
    plt.close(figure)


def _scatter_evidence_2d(
    axis: Axes,
    points: NDArray[np.floating],
    states: NDArray[np.uint8],
    first: int,
    second: int,
) -> None:
    for state in EvidenceState:
        selected = points[states == state.value]
        axis.scatter(
            selected[:, first],
            selected[:, second],
            s=2,
            c=EVIDENCE_COLORS[state],
            alpha=0.7,
            label=state.name.lower(),
        )


def _plot_box_projection(
    axis: Axes,
    box: Box3D,
    first: int,
    second: int,
    color: str,
    label: str,
) -> None:
    corners = box_corners(box)
    used_label = False
    for start, end in BOX_EDGE_INDICES:
        axis.plot(
            corners[[start, end], first],
            corners[[start, end], second],
            color=color,
            linewidth=1.4,
            label=label if not used_label else None,
        )
        used_label = True


def _worst_frame_id(
    evaluation: EvaluationReport | None, preview_frames: list[dict[str, object]]
) -> str:
    if evaluation is None or evaluation.refined is None:
        return str(preview_frames[0]["frame_id"])
    worst = max(
        evaluation.refined.frames,
        key=lambda item: (
            item.center_xy_error_m + item.center_z_error_m + item.yaw_error_deg / 10
        ),
    )
    return worst.frame_id


def _write_html(
    path: Path,
    case: RefinementCase,
    outcome: RefinementOutcome,
    aggregate: NDArray[np.float32],
    aggregate_frame_index: NDArray[np.int16],
    gold_aggregate: NDArray[np.float32] | None,
    gold_aggregate_frame_index: NDArray[np.int16] | None,
    aggregate_evidence_state: NDArray[np.uint8] | None,
    preview_frames: list[dict[str, object]],
    target: GoldTarget | None,
    evaluation: EvaluationReport | None,
    trace: GeometricRefinementTrace | None,
    data_source: str,
    review_mode: str,
) -> None:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as error:
        raise RuntimeError(
            "review bundle rendering requires 'pip install trackrefinery[review]'"
        ) from error

    aggregate_figure = go.Figure()
    for index, frame in enumerate(case.frames):
        points = aggregate[aggregate_frame_index == index]
        aggregate_figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={"size": 1.5, "opacity": 0.55},
                name=frame.frame_id,
            )
        )
    result_size = (
        outcome.canonical_size_lwh
        if isinstance(outcome, RefinementSuccess)
        else (
            trace.cuboid_fit.canonical_size_lwh
            if trace is not None
            and trace.cuboid_fit is not None
            and trace.cuboid_fit.status == "candidate"
            and trace.cuboid_fit.canonical_size_lwh is not None
            else tuple(
                np.median(
                    [item.coarse_box.size_lwh for item in case.track.observations],
                    axis=0,
                )
            )
        )
    )
    if isinstance(outcome, RefinementSuccess):
        result_label = "result size"
        result_color = "#00c853"
    elif (
        trace is not None
        and trace.cuboid_fit is not None
        and trace.cuboid_fit.status == "candidate"
    ):
        result_label = "cuboid candidate (not released)"
        result_color = "#d500f9"
    else:
        result_label = "coarse median size (not refined)"
        result_color = "#7c4dff"
    _add_plotly_box(
        aggregate_figure,
        Box3D((0.0, 0.0, 0.0), result_size, (0.0, 0.0, 0.0, 1.0)),
        result_label,
        result_color,
    )
    if target is not None:
        _add_plotly_box(
            aggregate_figure,
            Box3D((0.0, 0.0, 0.0), target.canonical_size_lwh, (0.0, 0.0, 0.0, 1.0)),
            "gold size",
            "#00b8d4",
        )
    aggregate_title = (
        "Source-annotation-aligned aggregate (reference only; not gold)"
        if review_mode == "source_annotation_reference"
        else "Algorithm-candidate-aligned aggregate (points colored by frame)"
    )
    aggregate_figure.update_layout(
        title=aggregate_title,
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )

    gold_aggregate_figure = None
    if (
        gold_aggregate is not None
        and gold_aggregate_frame_index is not None
        and target is not None
    ):
        gold_aggregate_figure = go.Figure()
        for index, frame in enumerate(case.frames):
            points = gold_aggregate[gold_aggregate_frame_index == index]
            gold_aggregate_figure.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker={"size": 1.5, "opacity": 0.55},
                    name=frame.frame_id,
                )
            )
        _add_plotly_box(
            gold_aggregate_figure,
            Box3D(
                (0.0, 0.0, 0.0),
                target.canonical_size_lwh,
                (0.0, 0.0, 0.0, 1.0),
            ),
            "gold size",
            "#00b8d4",
        )
        if (
            trace is not None
            and trace.cuboid_fit is not None
            and trace.cuboid_fit.canonical_size_lwh is not None
        ):
            _add_plotly_box(
                gold_aggregate_figure,
                Box3D(
                    (0.0, 0.0, 0.0),
                    trace.cuboid_fit.canonical_size_lwh,
                    (0.0, 0.0, 0.0, 1.0),
                ),
                "cuboid candidate size",
                "#d500f9",
            )
        gold_aggregate_figure.update_layout(
            title="Annotation-pose-aligned aggregate (review only)",
            scene={"aspectmode": "data"},
            margin={"l": 0, "r": 0, "t": 45, "b": 0},
        )

    canonical_figure = None
    if trace is not None and trace.canonical_shape is not None:
        shape = trace.canonical_shape
        canonical_figure = go.Figure(
            data=[
                go.Scatter3d(
                    x=shape.points_xyz[:, 0],
                    y=shape.points_xyz[:, 1],
                    z=shape.points_xyz[:, 2],
                    mode="markers",
                    marker={
                        "size": 2,
                        "color": shape.frame_support_count,
                        "colorscale": "Viridis",
                        "colorbar": {"title": "frames"},
                        "opacity": 0.75,
                    },
                    name="persistent evidence",
                )
            ]
        )
        canonical_figure.update_layout(
            title="Canonical shape after alternating registration",
            scene={"aspectmode": "data"},
            margin={"l": 0, "r": 0, "t": 45, "b": 0},
        )
        if (
            trace.cuboid_fit is not None
            and trace.cuboid_fit.status == "candidate"
            and trace.cuboid_fit.canonical_size_lwh is not None
            and trace.cuboid_fit.center_in_registration_xyz is not None
        ):
            _add_plotly_box(
                canonical_figure,
                Box3D(
                    trace.cuboid_fit.center_in_registration_xyz,
                    trace.cuboid_fit.canonical_size_lwh,
                    (0.0, 0.0, 0.0, 1.0),
                ),
                "visible-envelope candidate",
                "#d500f9",
            )

    evidence_figure = None
    if aggregate_evidence_state is not None:
        evidence_figure = go.Figure()
        for state_trace in _plotly_evidence_traces(aggregate, aggregate_evidence_state):
            evidence_figure.add_trace(state_trace)
        _add_plotly_box(
            evidence_figure,
            Box3D((0.0, 0.0, 0.0), result_size, (0.0, 0.0, 0.0, 1.0)),
            result_label,
            result_color,
        )
        evidence_figure.update_layout(
            title="Current evidence classification",
            scene={"aspectmode": "data"},
            margin={"l": 0, "r": 0, "t": 45, "b": 0},
        )

    animation_frames = []
    for item in preview_frames:
        points = item["points_xyz"]
        point_states = item["point_states"]
        traces: list[object]
        if isinstance(point_states, np.ndarray):
            traces = _plotly_evidence_traces(points, point_states)
        else:
            traces = [
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker={"size": 1.5, "color": "#a0a0a0", "opacity": 0.65},
                    name="context points",
                )
            ]
        for key, label, color in (
            ("coarse_box", "coarse", "#ffab00"),
            ("registration_box", "registration candidate", "#7c4dff"),
            ("cuboid_candidate_box", "cuboid candidate", "#d500f9"),
            ("refined_box", "refined", "#00c853"),
            ("gold_box", "gold", "#00b8d4"),
        ):
            box = item[key]
            if isinstance(box, Box3D):
                traces.append(_plotly_box_trace(box, label, color))
        animation_frames.append(go.Frame(name=str(item["frame_id"]), data=traces))
    frame_figure = go.Figure(
        data=animation_frames[0].data,
        frames=animation_frames,
        layout=go.Layout(
            title="Per-frame annotation coordinates",
            scene={"aspectmode": "data"},
            margin={"l": 0, "r": 0, "t": 45, "b": 0},
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 350}}],
                        }
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "label": frame.name,
                            "method": "animate",
                            "args": [[frame.name], {"mode": "immediate"}],
                        }
                        for frame in animation_frames
                    ]
                }
            ],
        ),
    )
    aggregate_html = pio.to_html(
        aggregate_figure, include_plotlyjs=True, full_html=False
    )
    gold_aggregate_html = (
        pio.to_html(
            gold_aggregate_figure,
            include_plotlyjs=False,
            full_html=False,
        )
        if gold_aggregate_figure is not None
        else ""
    )
    canonical_html = (
        pio.to_html(canonical_figure, include_plotlyjs=False, full_html=False)
        if canonical_figure is not None
        else ""
    )
    evidence_html = (
        pio.to_html(evidence_figure, include_plotlyjs=False, full_html=False)
        if evidence_figure is not None
        else ""
    )
    frame_html = pio.to_html(frame_figure, include_plotlyjs=False, full_html=False)
    metrics_json = (
        json.dumps(evaluation.to_dict(), indent=2, sort_keys=True)
        if evaluation is not None
        else "No gold metrics for this case."
    )
    trace_json = (
        json.dumps(trace.to_summary_dict(), indent=2, sort_keys=True)
        if trace is not None
        else "No algorithm evidence trace for this case."
    )
    case_display = html_module.escape(case.case_id)
    track_display = html_module.escape(case.track.track_id)
    source_display = html_module.escape(data_source)
    review_mode_display = html_module.escape(review_mode)
    candidate_display = (
        "none"
        if trace is None
        or trace.cuboid_fit is None
        or trace.cuboid_fit.status != "candidate"
        or trace.cuboid_fit.canonical_size_lwh is None
        else " x ".join(
            f"{value:.3f} m" for value in trace.cuboid_fit.canonical_size_lwh
        )
    )
    view_tabs = [
        ("algorithm", "Algorithm aggregate", aggregate_html),
    ]
    if gold_aggregate_html:
        view_tabs.append(("annotation", "Annotation aggregate", gold_aggregate_html))
    if canonical_html:
        view_tabs.append(("canonical", "Canonical shape", canonical_html))
    if evidence_html:
        view_tabs.append(("evidence", "Current evidence", evidence_html))
    view_tabs.extend(
        (
            ("frames", "Per-frame result", frame_html),
            (
                "metrics",
                "Metrics",
                f"<h2>Metrics</h2><pre>{html_module.escape(metrics_json)}</pre>",
            ),
            (
                "trace",
                "Diagnostics",
                f"<h2>Evidence trace</h2><pre>{html_module.escape(trace_json)}</pre>",
            ),
        )
    )
    view_buttons = "".join(
        (
            f'<button class="view-tab{" active" if index == 0 else ""}" '
            f"onclick=\"showView('{tab_id}', this)\">{label}</button>"
        )
        for index, (tab_id, label, _) in enumerate(view_tabs)
    )
    view_panels = "".join(
        (
            f'<section id="view-{tab_id}" '
            f'class="view-panel{" active" if index == 0 else ""}">{content}</section>'
        )
        for index, (tab_id, _, content) in enumerate(view_tabs)
    )
    feedback_filename = json.dumps(f"feedback-{case.case_id}.json")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>TrackRefinery review: {case_display}</title>
  <style>
    body {{ font-family: system-ui; margin: 0; background: #111827; color: #e5e7eb; }}
    main {{ max-width: 1500px; margin: auto; padding: 20px; }}
    section {{ background: #fff; color: #111827; border-radius: 10px;
      margin: 16px 0; padding: 14px; }}
    pre {{ max-height: 380px; overflow: auto; background: #0f172a;
      color: #d1fae5; padding: 12px; }}
    button {{ padding: 8px 14px; }}
    .view-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 0; }}
    .view-tab {{ border: 1px solid #64748b; border-radius: 8px;
      background: #1e293b; color: #e5e7eb; cursor: pointer; }}
    .view-tab.active {{ background: #2563eb; border-color: #60a5fa; }}
    .view-panel {{ display: none; }}
    .view-panel.active {{ display: block; }}
  </style>
</head>
<body>
<main>
  <h1>{case_display}</h1>
  <p>Track {track_display} · {outcome.status}</p>
  <p><strong>Data source:</strong> {source_display}</p>
  <p><strong>Review mode:</strong> {review_mode_display}</p>
  <p><strong>Trace-only cuboid candidate:</strong> {candidate_display}</p>
  <nav class="view-tabs">{view_buttons}</nav>
  {view_panels}
  <section>
    <h2>Reviewer feedback</h2>
    <select id="verdict">
      <option>good</option>
      <option>size_wrong</option>
      <option>pose_wrong</option>
      <option>point_selection_wrong</option>
      <option>insufficient_evidence</option>
    </select>
    <input id="notes" placeholder="notes">
    <button onclick="downloadFeedback()">Download feedback JSON</button>
  </section>
</main>
<script>
function showView(id, button) {{
  document.querySelectorAll('.view-panel').forEach(
    element => element.classList.remove('active')
  );
  document.querySelectorAll('.view-tab').forEach(
    element => element.classList.remove('active')
  );
  const panel = document.getElementById('view-' + id);
  panel.classList.add('active');
  button.classList.add('active');
  panel.querySelectorAll('.plotly-graph-div').forEach(
    element => Plotly.Plots.resize(element)
  );
}}
function downloadFeedback() {{
  const value = {{
    contract_version: 'trackrefinery-review-feedback-v1',
    case_id: {json.dumps(case.case_id)},
    track_id: {json.dumps(case.track.track_id)},
    verdict: document.getElementById('verdict').value,
    notes: document.getElementById('notes').value
  }};
  const blob = new Blob(
    [JSON.stringify(value, null, 2) + '\\n'],
    {{type: 'application/json'}}
  );
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = {feedback_filename};
  link.click();
  URL.revokeObjectURL(link.href);
}}
</script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def _add_plotly_box(figure: object, box: Box3D, name: str, color: str) -> None:
    figure.add_trace(_plotly_box_trace(box, name, color))


def _plotly_evidence_traces(
    points: NDArray[np.floating], states: NDArray[np.uint8]
) -> list[object]:
    import plotly.graph_objects as go

    traces = []
    for state in EvidenceState:
        selected = points[states == state.value]
        traces.append(
            go.Scatter3d(
                x=selected[:, 0],
                y=selected[:, 1],
                z=selected[:, 2],
                mode="markers",
                marker={
                    "size": 1.8,
                    "color": EVIDENCE_COLORS[state],
                    "opacity": 0.7,
                },
                name=state.name.lower(),
            )
        )
    return traces


def _plotly_box_trace(box: Box3D, name: str, color: str) -> object:
    import plotly.graph_objects as go

    corners = box_corners(box)
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for start, end in BOX_EDGE_INDICES:
        x.extend((float(corners[start, 0]), float(corners[end, 0]), None))
        y.extend((float(corners[start, 1]), float(corners[end, 1]), None))
        z.extend((float(corners[start, 2]), float(corners[end, 2]), None))
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": color, "width": 5},
        name=name,
    )
