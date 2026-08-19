"""Visual bundles for controlled Stage 3 recovery diagnostics."""

# ruff: noqa: E501 -- self-contained HTML/CSS/JavaScript is kept readable.

from __future__ import annotations

import html as html_module
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trackrefinery.controlled_recovery import (
    CONTROLLED_RECOVERY_CONTRACT,
    ControlledRecoveryRun,
)
from trackrefinery.geometric.trace import EvidenceState, FrameRole
from trackrefinery.geometry import inverse_transform_points

CONTROLLED_RECOVERY_REVIEW_CONTRACT = "trackrefinery-controlled-recovery-review-v2"
CONTROLLED_RECOVERY_SUITE_CONTRACT = "trackrefinery-controlled-recovery-suite-v2"


def build_controlled_recovery_bundle(
    run: ControlledRecoveryRun,
    output_dir: str | Path,
    *,
    data_source: str,
    max_points_per_frame: int = 8_000,
) -> Path:
    """Write one reference/input/output same-point recovery review."""

    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("data_source must be a non-empty string")
    if max_points_per_frame <= 0:
        raise ValueError("max_points_per_frame must be positive")
    if run.reference_case.case_id != run.perturbed_case.case_id:
        raise ValueError("reference and perturbed cases do not match")

    output = Path(output_dir).resolve()
    thumbnails = output / "thumbnails"
    thumbnails.mkdir(parents=True, exist_ok=True)
    perturbations = {item.frame_id: item for item in run.perturbations}
    proxy_groups: list[NDArray[np.float32]] = []
    reference_groups: list[NDArray[np.float32]] = []
    input_groups: list[NDArray[np.float32]] = []
    output_groups: list[NDArray[np.float32]] = []
    frame_groups: list[NDArray[np.int16]] = []
    displayed_frame_ids: list[str] = []
    selected_point_count = 0
    for index, (
        frame,
        reference_observation,
        input_observation,
        frame_trace,
    ) in enumerate(
        zip(
            run.reference_case.frames,
            run.reference_case.track.observations,
            run.perturbed_case.track.observations,
            run.component_trace.frames,
            strict=True,
        )
    ):
        component = frame_trace.component
        if component is None or component.frame_role is not FrameRole.GEOMETRY:
            continue
        positions = np.flatnonzero(
            frame_trace.point_states == EvidenceState.TARGET.value
        )
        selected_point_count += len(positions)
        positions = _sample_positions(positions, max_points_per_frame)
        indices = frame_trace.roi_point_indices[positions]
        points = frame.points_xyz[indices]
        reference_registration = run.reference_output_trace.frames[index].registration
        reference_pose = (
            reference_registration.candidate_pose_annotation
            if reference_registration is not None
            and reference_registration.candidate_pose_annotation is not None
            else reference_observation.coarse_box.pose
        )
        registration = run.output_trace.frames[index].registration
        output_pose = (
            registration.candidate_pose_annotation
            if registration is not None
            and registration.candidate_pose_annotation is not None
            else input_observation.coarse_box.pose
        )
        proxy_groups.append(
            inverse_transform_points(
                points, reference_observation.coarse_box.pose
            ).astype(np.float32)
        )
        reference_groups.append(
            inverse_transform_points(points, reference_pose).astype(np.float32)
        )
        input_groups.append(
            inverse_transform_points(points, input_observation.coarse_box.pose).astype(
                np.float32
            )
        )
        output_groups.append(
            inverse_transform_points(points, output_pose).astype(np.float32)
        )
        frame_groups.append(np.full(len(points), len(displayed_frame_ids), np.int16))
        displayed_frame_ids.append(frame.frame_id)
    if not reference_groups:
        raise ValueError("recovery review requires selected geometry components")

    proxy = np.concatenate(proxy_groups)
    reference = np.concatenate(reference_groups)
    perturbed = np.concatenate(input_groups)
    repaired = np.concatenate(output_groups)
    frame_index = np.concatenate(frame_groups)
    np.savez_compressed(
        output / "aggregates.npz",
        proxy_points_xyz=proxy,
        reference_points_xyz=reference,
        input_points_xyz=perturbed,
        output_points_xyz=repaired,
        frame_index=frame_index,
        frame_ids=np.asarray(displayed_frame_ids),
    )
    run.report.write_json(output / "recovery.json")
    manifest = {
        "contract_version": CONTROLLED_RECOVERY_REVIEW_CONTRACT,
        "recovery_contract_version": CONTROLLED_RECOVERY_CONTRACT,
        "case_id": run.reference_case.case_id,
        "track_id": run.reference_case.track.track_id,
        "category": run.reference_case.track.category,
        "algorithm_variant": run.report.algorithm_variant,
        "profile": run.report.profile.to_dict(),
        "anchor_frame_id": run.report.anchor_frame_id,
        "geometry_frame_count": run.report.geometry_frame_count,
        "perturbed_frame_count": run.report.perturbed_frame_count,
        "displayed_point_count": len(reference),
        "selected_component_point_count": selected_point_count,
        "data_source": data_source.strip(),
        "reference_semantics": "frozen_model_track_proxy_not_gold",
        "equivariant_reference_semantics": (
            "same_algorithm_unperturbed_output_not_gold"
        ),
        "metrics": {
            key: value
            for key, value in run.report.to_dict().items()
            if key
            in {
                "input_translation_rms_m",
                "output_translation_rms_m",
                "input_yaw_rms_deg",
                "output_yaw_rms_deg",
                "translation_rms_reduction_fraction",
                "yaw_rms_reduction_fraction",
                "improved_translation_frame_fraction",
                "improved_yaw_frame_fraction",
                "registered_frame_count",
                "retained_coarse_frame_count",
                "unavailable_frame_count",
                "equivariant_output_translation_rms_m",
                "equivariant_output_yaw_rms_deg",
                "equivariant_translation_rms_reduction_fraction",
                "equivariant_yaw_rms_reduction_fraction",
                "equivariant_improved_translation_frame_fraction",
                "equivariant_improved_yaw_frame_fraction",
            }
        },
        "aggregate_path": "aggregates.npz",
        "report_path": "recovery.json",
        "preview_path": "preview.html",
        "comparison_top_path": "thumbnails/comparison_top.png",
        "comparison_side_path": "thumbnails/comparison_side.png",
        "perturbations": [item.to_dict() for item in perturbations.values()],
    }
    (output / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_recovery_thumbnails(
        thumbnails,
        proxy,
        reference,
        perturbed,
        repaired,
        frame_index,
        run,
    )
    _write_recovery_html(
        output / "preview.html",
        proxy,
        reference,
        perturbed,
        repaired,
        frame_index,
        displayed_frame_ids,
        run,
        data_source.strip(),
    )
    return output


def build_controlled_recovery_suite(
    output_dir: str | Path,
    bundle_dirs: Sequence[str | Path],
    *,
    title: str = "TrackRefinery controlled pose recovery",
) -> Path:
    """Build case tabs containing every controlled perturbation profile."""

    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    if not bundle_dirs:
        raise ValueError("controlled recovery suite requires at least one bundle")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for bundle_dir in bundle_dirs:
        row = _recovery_bundle_row(output, bundle_dir)
        profile = row["profile"]
        if not isinstance(profile, dict):
            raise ValueError("recovery profile must be an object")
        identity = (
            str(row["case_id"]),
            str(row["algorithm_variant"]),
            str(profile["name"]),
        )
        if identity in seen:
            raise ValueError(f"duplicate recovery bundle: {identity}")
        seen.add(identity)
        grouped.setdefault(identity[0], []).append(row)
    cases = []
    for case_id, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                str(row["algorithm_variant"]),
                float(dict(row["profile"])["maximum_translation_m"]),
            )
        )
        cases.append({"case_id": case_id, "profiles": rows})
    manifest = {
        "contract_version": CONTROLLED_RECOVERY_SUITE_CONTRACT,
        "title": title.strip(),
        "cases": cases,
    }
    (output / "recovery-suite.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "index.html").write_text(
        _recovery_suite_html(title.strip(), cases), encoding="utf-8"
    )
    return output


def _sample_positions(positions: NDArray[np.int64], maximum: int) -> NDArray[np.int64]:
    if len(positions) <= maximum:
        return positions
    selected = np.linspace(0, len(positions) - 1, maximum, dtype=np.int64)
    return positions[selected]


def _write_recovery_thumbnails(
    output: Path,
    proxy: NDArray[np.float32],
    reference: NDArray[np.float32],
    perturbed: NDArray[np.float32],
    repaired: NDArray[np.float32],
    frame_index: NDArray[np.int16],
    run: ControlledRecoveryRun,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "recovery review rendering requires 'pip install trackrefinery[review]'"
        ) from error
    frame_maximum = max(1, int(np.max(frame_index)))
    report = run.report
    for filename, first, second, labels in (
        ("comparison_top.png", 0, 1, ("X (m)", "Y (m)")),
        ("comparison_side.png", 0, 2, ("X (m)", "Z (m)")),
    ):
        figure, axes = plt.subplots(
            1, 4, figsize=(20, 5), sharex=True, sharey=True, constrained_layout=True
        )
        projected = np.concatenate(
            [
                proxy[:, [first, second]],
                reference[:, [first, second]],
                perturbed[:, [first, second]],
                repaired[:, [first, second]],
            ]
        )
        lower = np.min(projected, axis=0)
        upper = np.max(projected, axis=0)
        padding = np.maximum((upper - lower) * 0.05, 0.05)
        titles = (
            "PROXY · frozen model track · not gold",
            "REFERENCE · same algorithm on unperturbed input",
            (
                f"INPUT · injected up to {report.profile.maximum_translation_m:.2f} m"
                f" / {report.profile.maximum_yaw_deg:.1f}°"
            ),
            (
                "OUTPUT · equivariant RMS "
                f"{report.input_translation_rms_m:.3f}→"
                f"{report.equivariant_output_translation_rms_m:.3f} m, "
                f"{report.input_yaw_rms_deg:.2f}→"
                f"{report.equivariant_output_yaw_rms_deg:.2f}°"
            ),
        )
        for axis, points, title in zip(
            axes, (proxy, reference, perturbed, repaired), titles, strict=True
        ):
            axis.scatter(
                points[:, first],
                points[:, second],
                c=frame_index,
                s=0.8,
                cmap="turbo",
                vmin=0,
                vmax=frame_maximum,
                alpha=0.58,
            )
            axis.set_title(title, fontsize=9)
            axis.set_xlabel(labels[0])
            axis.set_ylabel(labels[1])
            axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
            axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
            axis.set_aspect("equal", adjustable="box")
        figure.suptitle(
            "Controlled recovery · identical selected components, colors, and axes"
        )
        figure.savefig(output / filename, dpi=130)
        plt.close(figure)


def _write_recovery_html(
    path: Path,
    proxy: NDArray[np.float32],
    reference: NDArray[np.float32],
    perturbed: NDArray[np.float32],
    repaired: NDArray[np.float32],
    frame_index: NDArray[np.int16],
    frame_ids: list[str],
    run: ControlledRecoveryRun,
    data_source: str,
) -> None:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as error:
        raise RuntimeError(
            "recovery review rendering requires 'pip install trackrefinery[review]'"
        ) from error
    plots: list[str] = []
    for mode_index, (title, aggregate) in enumerate(
        (
            ("PROXY · frozen model-track input · not gold", proxy),
            ("REFERENCE · same algorithm on unperturbed input", reference),
            ("INPUT · controlled perturbation seen by Stage 3", perturbed),
            ("OUTPUT · anchored aggregation candidate", repaired),
        )
    ):
        figure = go.Figure()
        for index, frame_id in enumerate(frame_ids):
            points = aggregate[frame_index == index]
            figure.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker={"size": 1.5, "opacity": 0.55},
                    name=frame_id,
                )
            )
        figure.update_layout(
            title=title,
            scene={"aspectmode": "data"},
            margin={"l": 0, "r": 0, "t": 45, "b": 0},
        )
        plots.append(
            pio.to_html(
                figure,
                include_plotlyjs=mode_index == 0,
                full_html=False,
            )
        )
    report = run.report
    metrics = report.to_dict()
    rows = "".join(
        "<tr>"
        f"<td>{html_module.escape(frame.frame_id)}</td>"
        f"<td>{frame.phase:+.3f}</td>"
        f"<td>{frame.injected_translation_m:.3f} m</td>"
        f"<td>{frame.output_translation_error_m:.3f} m</td>"
        f"<td>{frame.equivariant_output_translation_error_m:.3f} m</td>"
        f"<td>{frame.injected_yaw_deg:.2f}°</td>"
        f"<td>{frame.output_yaw_error_deg:.2f}°</td>"
        f"<td>{frame.equivariant_output_yaw_error_deg:.2f}°</td>"
        f"<td>{html_module.escape(frame.output_status)}</td>"
        "</tr>"
        for frame in report.frames
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html_module.escape(report.case_id)} · controlled recovery</title>
<style>
* {{ box-sizing:border-box }} body {{ font-family:system-ui; margin:0;
background:#0b1120; color:#e5e7eb }} header,main {{ padding:16px 20px }}
header {{ border-bottom:1px solid #334155 }} h1 {{ margin:0 0 8px }}
.notice {{ color:#fcd34d }} .summary {{ display:grid;
grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:10px }}
.metric {{ background:#172033; border:1px solid #334155; border-radius:8px;
padding:10px }} nav {{ display:flex; gap:8px; margin:16px 0 }}
button {{ padding:8px 12px; border:1px solid #64748b; border-radius:7px;
background:#172033; color:#e5e7eb; cursor:pointer }} button.active {{ background:#1d4ed8 }}
.view {{ display:none }} .view.active {{ display:block }} .plotly-graph-div {{ height:72vh }}
img {{ width:100%; background:white; border-radius:8px }} table {{ width:100%;
border-collapse:collapse; font-size:12px }} th,td {{ border-bottom:1px solid #334155;
padding:7px; text-align:left }} pre {{ white-space:pre-wrap; background:#111827;
padding:12px; border-radius:8px }}
</style></head><body><header>
<h1>{html_module.escape(report.case_id)} · {html_module.escape(report.profile.name)}</h1>
<p>{html_module.escape(data_source)} · {html_module.escape(report.algorithm_variant)} ·
anchor {html_module.escape(report.anchor_frame_id)}</p>
<p class="notice">PROXY is the frozen model track and is not reviewed gold.
REFERENCE is the same algorithm's unperturbed output. The perturbed run receives
INPUT only; it never receives PROXY or REFERENCE poses.</p>
<div class="summary">
<div class="metric"><b>Equivariant translation RMS</b><br>
{report.input_translation_rms_m:.3f} →
{report.equivariant_output_translation_rms_m:.3f} m<br>
recovery {report.equivariant_translation_rms_reduction_fraction:.1%}</div>
<div class="metric"><b>Equivariant yaw RMS</b><br>{report.input_yaw_rms_deg:.2f} →
{report.equivariant_output_yaw_rms_deg:.2f}°<br>
recovery {report.equivariant_yaw_rms_reduction_fraction:.1%}</div>
<div class="metric"><b>Absolute proxy RMS</b><br>translation
{report.output_translation_rms_m:.3f} m · yaw {report.output_yaw_rms_deg:.2f}°</div>
<div class="metric"><b>Disposition</b><br>{report.registered_frame_count} registered ·
{report.retained_coarse_frame_count} retained · {report.unavailable_frame_count}
unavailable</div></div></header><main>
<img src="thumbnails/comparison_top.png" alt="Proxy reference input output comparison">
<nav><button class="active" onclick="showView(0,this)">PROXY</button>
<button onclick="showView(1,this)">REFERENCE</button>
<button onclick="showView(2,this)">INPUT</button>
<button onclick="showView(3,this)">OUTPUT</button>
<button onclick="showView(4,this)">Per-frame metrics</button>
<button onclick="showView(5,this)">JSON</button></nav>
<section class="view active">{plots[0]}</section>
<section class="view">{plots[1]}</section><section class="view">{plots[2]}</section>
<section class="view">{plots[3]}</section>
<section class="view"><h2>Per-frame recovery</h2><table><thead><tr>
<th>Frame</th><th>Phase</th><th>Input XY</th><th>Proxy XY</th><th>Equiv XY</th>
<th>Input yaw</th><th>Proxy yaw</th><th>Equiv yaw</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table></section>
<section class="view"><pre>{html_module.escape(json.dumps(metrics, indent=2, sort_keys=True))}</pre></section>
</main><script>function showView(index,button){{document.querySelectorAll('.view').forEach(
(e,i)=>e.classList.toggle('active',i===index));document.querySelectorAll('nav button').forEach(
e=>e.classList.remove('active'));button.classList.add('active');const p=document.querySelectorAll(
'.view')[index];p.querySelectorAll('.plotly-graph-div').forEach(e=>Plotly.Plots.resize(e));}}</script>
</body></html>""",
        encoding="utf-8",
    )


def _recovery_bundle_row(output: Path, bundle_dir: str | Path) -> dict[str, object]:
    bundle = Path(bundle_dir).resolve()
    try:
        relative = bundle.relative_to(output)
    except ValueError as error:
        raise ValueError("recovery bundles must be inside output_dir") from error
    manifest_path = bundle / "bundle.json"
    required = (
        manifest_path,
        bundle / "preview.html",
        bundle / "thumbnails" / "comparison_top.png",
        bundle / "thumbnails" / "comparison_side.png",
    )
    if not all(path.is_file() for path in required):
        raise ValueError(f"{bundle} is not a complete recovery bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_version") != CONTROLLED_RECOVERY_REVIEW_CONTRACT:
        raise ValueError(f"{manifest_path} is not a controlled recovery bundle")
    return {
        **manifest,
        "preview_path": (relative / "preview.html").as_posix(),
        "comparison_top_path": (
            relative / "thumbnails" / "comparison_top.png"
        ).as_posix(),
        "comparison_side_path": (
            relative / "thumbnails" / "comparison_side.png"
        ).as_posix(),
    }


def _recovery_suite_html(title: str, cases: list[dict[str, object]]) -> str:
    buttons: list[str] = []
    panels: list[str] = []
    for index, case in enumerate(cases):
        case_id = str(case["case_id"])
        profiles = case["profiles"]
        if not isinstance(profiles, list):
            raise ValueError("recovery profiles must be a list")
        active = " active" if index == 0 else ""
        buttons.append(
            f'<button class="case-tab{active}" onclick="showCase({index},this)">'
            f"{html_module.escape(case_id)}<small>{len(profiles)} profiles</small></button>"
        )
        cards = "".join(_recovery_suite_card(row) for row in profiles)
        panels.append(
            f'<section id="case-{index}" class="case-panel{active}">{cards}</section>'
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html_module.escape(title)}</title><style>
*{{box-sizing:border-box}} body{{font-family:system-ui;margin:0;background:#0b1120;color:#e5e7eb}}
header{{position:sticky;top:0;z-index:2;padding:14px 18px;background:#0b1120f7;
border-bottom:1px solid #334155}} h1{{margin:0 0 8px}} .warning{{color:#fcd34d}}
nav{{display:flex;gap:8px;overflow-x:auto}} button{{border:1px solid #475569;border-radius:8px;
background:#172033;color:#e5e7eb;padding:9px 12px;cursor:pointer;text-align:left}}
button small{{display:block;color:#94a3b8}} button.active{{background:#1d4ed8}}
main{{padding:16px 18px}} .case-panel{{display:none}} .case-panel.active{{display:grid;gap:16px}}
.card{{border:1px solid #334155;border-left:6px solid #a78bfa;border-radius:10px;
padding:12px;background:#111827;cursor:pointer}} .card:hover{{border-color:#60a5fa}}
.head{{display:flex;justify-content:space-between;gap:12px}} .badge{{color:#c4b5fd}}
.views{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:9px}}
img{{width:100%;background:white;border-radius:7px}} .metrics{{display:flex;gap:18px;
flex-wrap:wrap;color:#cbd5e1;font-size:13px}} dialog{{width:min(1500px,96vw);height:94vh;
padding:0;border:1px solid #475569;background:#111827}} iframe{{width:100%;height:calc(94vh - 46px);
border:0}} .dialog-head{{height:46px;padding:8px 12px;display:flex;justify-content:space-between}}
</style></head><body><header><h1>{html_module.escape(title)}</h1>
<p class="warning">Known-error Stage 3 diagnostic. PROXY is not annotation gold;
REFERENCE is the same algorithm's unperturbed output. Neither is an input to the
perturbed run.</p><nav>{"".join(buttons)}</nav></header>
<main>{"".join(panels)}</main><dialog id="detail"><div class="dialog-head"><b id="detail-title">
Recovery</b><button onclick="closeDetail()">Close</button></div><iframe id="detail-frame"></iframe>
</dialog><script>function showCase(i,b){{document.querySelectorAll('.case-panel').forEach(
e=>e.classList.remove('active'));document.querySelectorAll('.case-tab').forEach(
e=>e.classList.remove('active'));document.getElementById('case-'+i).classList.add('active');
b.classList.add('active')}} function openDetail(path,title){{document.getElementById('detail-title').textContent=title;
document.getElementById('detail-frame').src=path;document.getElementById('detail').showModal()}}
function closeDetail(){{document.getElementById('detail').close();document.getElementById('detail-frame').src='about:blank'}}
</script></body></html>"""


def _recovery_suite_card(value: object) -> str:
    if not isinstance(value, dict):
        raise ValueError("recovery suite row must be an object")
    profile = value.get("profile")
    metrics = value.get("metrics")
    if not isinstance(profile, dict) or not isinstance(metrics, dict):
        raise ValueError("recovery suite row is missing profile or metrics")
    name = str(profile["name"])
    variant = str(value["algorithm_variant"])
    preview = html_module.escape(str(value["preview_path"]), quote=True)
    title = f"{value['case_id']} · {variant} · {name}"
    return (
        '<article class="card" '
        f'onclick="openDetail({html_module.escape(json.dumps(preview), quote=True)},'
        f'{html_module.escape(json.dumps(title), quote=True)})">'
        f'<div class="head"><div><b>{html_module.escape(variant)}</b><br>'
        f"<span>{html_module.escape(name.upper())}</span><br>"
        f'<span class="badge">up to {float(profile["maximum_translation_m"]):.2f} m / '
        f"{float(profile['maximum_yaw_deg']):.1f}°</span></div>"
        f"<span>{int(value['perturbed_frame_count'])} perturbed geometry frames</span></div>"
        '<div class="views">'
        f'<img src="{html_module.escape(str(value["comparison_top_path"]), quote=True)}">'
        f'<img src="{html_module.escape(str(value["comparison_side_path"]), quote=True)}">'
        '</div><div class="metrics">'
        f"<span>Equiv XY {float(metrics['input_translation_rms_m']):.3f} → "
        f"{float(metrics['equivariant_output_translation_rms_m']):.3f} m "
        f"({float(metrics['equivariant_translation_rms_reduction_fraction']):+.1%})</span>"
        f"<span>Equiv yaw {float(metrics['input_yaw_rms_deg']):.2f} → "
        f"{float(metrics['equivariant_output_yaw_rms_deg']):.2f}° "
        f"({float(metrics['equivariant_yaw_rms_reduction_fraction']):+.1%})</span>"
        f"<span>Proxy XY/yaw {float(metrics['output_translation_rms_m']):.3f} m / "
        f"{float(metrics['output_yaw_rms_deg']):.2f}°</span>"
        f"<span>registered {int(metrics['registered_frame_count'])}</span></div></article>"
    )
