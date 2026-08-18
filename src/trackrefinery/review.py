"""Deterministic visual review bundles for algorithm-development feedback."""

from __future__ import annotations

import html as html_module
import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from trackrefinery.contracts import (
    Box3D,
    RefinementCase,
    RefinementOutcome,
    RefinementSuccess,
)
from trackrefinery.evaluation import EvaluationReport
from trackrefinery.geometric.trace import (
    EvidenceState,
    GeometricRefinementTrace,
    validate_geometric_trace,
    write_geometric_trace,
)
from trackrefinery.geometry import (
    BOX_EDGE_INDICES,
    box_corners,
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


def build_review_bundle(
    case: RefinementCase,
    outcome: RefinementOutcome,
    output_dir: str | Path,
    *,
    target: GoldTarget | None = None,
    evaluation: EvaluationReport | None = None,
    trace: GeometricRefinementTrace | None = None,
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
    preview_frames: list[dict[str, object]] = []
    evidence_states: list[NDArray[np.uint8]] = []
    trace_by_frame = (
        {frame.frame_id: frame for frame in trace.frames} if trace is not None else {}
    )
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
            refined_box.pose if refined_box is not None else coarse_box.pose
        )
        local = inverse_transform_points(points, alignment_pose).astype(np.float32)
        aggregate_groups.append(local)
        frame_indices.append(np.full(len(local), frame_index, dtype=np.int16))
        if point_states is not None:
            evidence_states.append(point_states)
        preview_frames.append(
            {
                "frame_id": frame.frame_id,
                "points_xyz": points,
                "coarse_box": coarse_box,
                "refined_box": refined_box,
                "gold_box": gold_box,
                "point_states": point_states,
            }
        )

    aggregate = np.concatenate(aggregate_groups)
    aggregate_frame_index = np.concatenate(frame_indices)
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
        "outcome_status": outcome.status,
        "frame_ids": [frame.frame_id for frame in case.frames],
        "crop_scale": crop_scale,
        "max_points_per_frame": max_points_per_frame,
        "has_gold_target": target is not None,
        "has_metrics": evaluation is not None,
        "has_evidence_trace": trace is not None,
        "evidence_trace_path": "evidence_trace.json" if trace is not None else None,
        "evidence_masks_path": "evidence_masks.npz" if trace is not None else None,
    }
    (output / "bundle.json").write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _write_thumbnails(
        thumbnails,
        aggregate,
        aggregate_frame_index,
        aggregate_evidence_state,
        preview_frames,
        outcome,
        target,
        evaluation,
    )
    _write_html(
        output / "preview.html",
        case,
        outcome,
        aggregate,
        aggregate_frame_index,
        aggregate_evidence_state,
        preview_frames,
        target,
        evaluation,
        trace,
    )
    return output


def serve_review_bundle(
    bundle_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = False,
) -> None:
    """Serve a generated bundle without requiring X-4D or X-Points."""

    root = Path(bundle_dir).resolve()
    if not (root / "preview.html").is_file():
        raise ValueError(f"{root} is not a review bundle")

    class BundleHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(root), **kwargs)

    server = ThreadingHTTPServer((host, port), BundleHandler)
    address, bound_port = server.server_address[:2]
    url = f"http://{address}:{bound_port}/preview.html"
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


def _write_thumbnails(
    output: Path,
    aggregate: NDArray[np.float32],
    frame_index: NDArray[np.int16],
    evidence_state: NDArray[np.uint8] | None,
    preview_frames: list[dict[str, object]],
    outcome: RefinementOutcome,
    target: GoldTarget | None,
    evaluation: EvaluationReport | None,
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
        _plot_box_projection(
            axis,
            Box3D((0.0, 0.0, 0.0), size, (0.0, 0.0, 0.0, 1.0)),
            first,
            second,
            "#00c853",
            "result",
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

    if evidence_state is not None:
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        _scatter_evidence_2d(axis, aggregate, evidence_state, 0, 1)
        _plot_box_projection(
            axis,
            Box3D((0.0, 0.0, 0.0), size, (0.0, 0.0, 0.0, 1.0)),
            0,
            1,
            "#d500f9",
            "coarse median size",
        )
        axis.set_title("Initial evidence classification")
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
    aggregate_evidence_state: NDArray[np.uint8] | None,
    preview_frames: list[dict[str, object]],
    target: GoldTarget | None,
    evaluation: EvaluationReport | None,
    trace: GeometricRefinementTrace | None,
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
        else tuple(
            np.median(
                [item.coarse_box.size_lwh for item in case.track.observations], axis=0
            )
        )
    )
    _add_plotly_box(
        aggregate_figure,
        Box3D((0.0, 0.0, 0.0), result_size, (0.0, 0.0, 0.0, 1.0)),
        "result size",
        "#00c853",
    )
    if target is not None:
        _add_plotly_box(
            aggregate_figure,
            Box3D((0.0, 0.0, 0.0), target.canonical_size_lwh, (0.0, 0.0, 0.0, 1.0)),
            "gold size",
            "#00b8d4",
        )
    aggregate_figure.update_layout(
        title="Object-frame aggregate (points colored by frame)",
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )

    evidence_figure = None
    if aggregate_evidence_state is not None:
        evidence_figure = go.Figure()
        for state_trace in _plotly_evidence_traces(aggregate, aggregate_evidence_state):
            evidence_figure.add_trace(state_trace)
        _add_plotly_box(
            evidence_figure,
            Box3D((0.0, 0.0, 0.0), result_size, (0.0, 0.0, 0.0, 1.0)),
            "coarse median size",
            "#d500f9",
        )
        evidence_figure.update_layout(
            title="Initial evidence classification",
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
    evidence_html = (
        pio.to_html(evidence_figure, include_plotlyjs=False, full_html=False)
        if evidence_figure is not None
        else ""
    )
    evidence_section = f"<section>{evidence_html}</section>" if evidence_html else ""
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
  </style>
</head>
<body>
<main>
  <h1>{case_display}</h1>
  <p>Track {track_display} · {outcome.status}</p>
  <section>{aggregate_html}</section>
  {evidence_section}
  <section>{frame_html}</section>
  <section><h2>Metrics</h2><pre>{metrics_json}</pre></section>
  <section><h2>Evidence trace</h2><pre>{trace_json}</pre></section>
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
