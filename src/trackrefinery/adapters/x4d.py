"""X-4D Dataset 0.17 adapter for local, frozen development inputs.

The adapter is intentionally outside the core. It materializes a portable
source-only TrackRefinery inference root first; algorithm processes then open
that root without access to the native Clip or evaluation targets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from trackrefinery.dataset import (
    ALLOWED_ROLES,
    INFERENCE_FORMAT,
    INFERENCE_VERSION,
    SEQUENCE_CONTRACT,
    TRACK_CONTRACT,
)

X4D_SOURCE_KINDS = frozenset({"model_candidate", "source_annotation_reference"})


@dataclass(frozen=True, slots=True)
class X4DInferenceExport:
    inference_root: Path
    clip_id: str
    case_ids: tuple[str, ...]
    lidar_channels: tuple[str, ...]
    annotation_revision: str
    source_kind: str
    dropped_nonfinite_points: int


def export_x4d_clip_inference(
    clip_dir: str | Path,
    annotation_document: str | Path | Mapping[str, Any],
    output_root: str | Path,
    *,
    role: str,
    source_kind: Literal["model_candidate", "source_annotation_reference"],
) -> X4DInferenceExport:
    """Export one native Clip and one frozen track document into v1 inputs.

    ``annotation_document`` may be a native ``annotation.json`` document or a
    protocol-v2 candidate envelope containing an ``annotation`` member. The
    document supplies coarse tracks only. Point frames are materialized with
    public Devkit transforms and never read annotation content.
    """

    if role not in ALLOWED_ROLES:
        raise ValueError(f"role must be one of {sorted(ALLOWED_ROLES)}")
    if source_kind not in X4D_SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {sorted(X4D_SOURCE_KINDS)}")
    try:
        from x4d_devkit import ClipLoader, derive_annotation_views
        from x4d_devkit.annotation_contract import parse_clip_annotations
        from x4d_devkit.core.transform import rotation_matrix_to_quat
    except ImportError as error:
        raise RuntimeError(
            "the X-4D adapter requires a Dataset 0.17 compatible x4d-devkit"
        ) from error

    clip_path = Path(clip_dir).resolve()
    output = Path(output_root).resolve()
    loader = ClipLoader(clip_path)
    loader.require_schema_version("0.17")
    annotation_payload, envelope = _load_annotation_payload(annotation_document)
    annotation = parse_clip_annotations(annotation_payload)
    if annotation.clip_id != loader.meta.clip_id:
        raise ValueError("annotation document clip_id does not match the native Clip")
    if annotation.annotation_frame_id != loader.annotation_frame_id:
        raise ValueError(
            "annotation document frame does not match meta.annotation_frame_id"
        )

    raw_samples = _read_json_array(clip_path / "sample.json")
    views = derive_annotation_views(list(annotation.annotations), raw_samples)
    track_documents = _track_documents(
        clip_id=loader.meta.clip_id,
        rows=views.annotation_rows(),
        samples=raw_samples,
    )
    if not track_documents:
        raise ValueError("annotation document contains no multi-frame instances")

    lidar_channels = tuple(
        sorted(
            channel
            for channel, sensor in loader.meta.sensors.items()
            if sensor.modality == "lidar"
        )
    )
    if not lidar_channels:
        raise ValueError("Clip declares no LiDAR channels")
    sequence_dir = output / "sources" / loader.meta.clip_id
    frames_dir = sequence_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, object]] = []
    dropped_total = 0
    for sample in loader.samples:
        if not sample.is_keyframe:
            continue
        arrays, origins, dropped = _materialize_frame(
            loader,
            sample.token,
            lidar_channels,
        )
        dropped_total += dropped
        points_path = frames_dir / f"{sample.token}.npz"
        np.savez_compressed(points_path, **arrays)
        sample_data = _representative_lidar_sample_data(
            loader, sample.token, lidar_channels
        )
        world_from_annotation = loader.get_transform(
            loader.annotation_frame_id,
            "clip_world",
            sd=sample_data,
        )
        frame_rows.append(
            {
                "frame_id": sample.token,
                "timestamp_ns": sample.timestamp.epoch_ns,
                "annotation_frame_id": loader.annotation_frame_id,
                "world_from_annotation": {
                    "translation_xyz": world_from_annotation.translation.tolist(),
                    "orientation_xyzw": list(
                        rotation_matrix_to_quat(world_from_annotation.rotation)
                    ),
                },
                "points_path": f"frames/{sample.token}.npz",
                "sensor_origins": origins,
                "dropped_nonfinite_points": dropped,
            }
        )

    _write_json(
        sequence_dir / "sequence.json",
        {
            "contract_version": SEQUENCE_CONTRACT,
            "sequence_id": loader.meta.clip_id,
            "sensors": [{"sensor_id": channel} for channel in lidar_channels],
            "feature_names": ["intensity"],
            "frames": frame_rows,
        },
    )
    case_rows: list[dict[str, str]] = []
    for document in track_documents:
        case_id = str(document["case_id"])
        track_path = output / "inputs" / case_id / "track.json"
        _write_json(track_path, document)
        case_rows.append(
            {
                "case_id": case_id,
                "sequence_id": loader.meta.clip_id,
                "track_path": f"inputs/{case_id}/track.json",
            }
        )
    source_identity_path = clip_path / ".cache_identity.json"
    source_identity = (
        json.loads(source_identity_path.read_text(encoding="utf-8"))
        if source_identity_path.is_file()
        else None
    )
    _write_json(
        output / "dataset.json",
        {
            "format": INFERENCE_FORMAT,
            "version": INFERENCE_VERSION,
            "dataset_id": f"x4d-{loader.meta.clip_id}-{annotation.annotation_revision}",
            "generator": {
                "name": "trackrefinery.adapters.x4d",
                "source_kind": source_kind,
                "annotation_revision": annotation.annotation_revision,
                "annotation_document_sha256": _annotation_input_sha256(
                    annotation_document
                ),
                "candidate_provenance": envelope.get("provenance"),
                "source_identity": source_identity,
                "lidar_channels": list(lidar_channels),
                "dropped_nonfinite_points": dropped_total,
            },
            "sequences": [
                {
                    "sequence_id": loader.meta.clip_id,
                    "role": role,
                    "manifest_path": (f"sources/{loader.meta.clip_id}/sequence.json"),
                }
            ],
            "cases": case_rows,
        },
    )
    return X4DInferenceExport(
        inference_root=output,
        clip_id=loader.meta.clip_id,
        case_ids=tuple(row["case_id"] for row in case_rows),
        lidar_channels=lidar_channels,
        annotation_revision=annotation.annotation_revision,
        source_kind=source_kind,
        dropped_nonfinite_points=dropped_total,
    )


def _materialize_frame(
    loader: Any,
    sample_token: str,
    lidar_channels: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, list[float]], int]:
    by_channel = {
        row.channel: row for row in loader.sample_data_for_sample(sample_token)
    }
    point_groups: list[np.ndarray] = []
    intensity_groups: list[np.ndarray] = []
    timestamp_groups: list[np.ndarray] = []
    sensor_index_groups: list[np.ndarray] = []
    origins: dict[str, list[float]] = {}
    dropped = 0
    for sensor_index, channel in enumerate(lidar_channels):
        if channel not in by_channel:
            raise ValueError(
                f"sample {sample_token!r} has no LiDAR row for {channel!r}"
            )
        sample_data = by_channel[channel]
        records = loader.load_point_records(
            sample_data, frame=loader.annotation_frame_id
        )
        points = np.column_stack((records["x"], records["y"], records["z"])).astype(
            np.float32,
            copy=False,
        )
        intensity = np.asarray(records["intensity"], dtype=np.float32)
        finite = np.isfinite(points).all(axis=1) & np.isfinite(intensity)
        dropped += int(len(points) - int(finite.sum()))
        point_groups.append(points[finite])
        intensity_groups.append(intensity[finite, None])
        timestamp_groups.append(
            np.asarray(records["timestamp_ns"][finite], dtype=np.uint64)
        )
        sensor_index_groups.append(
            np.full(int(finite.sum()), sensor_index, dtype=np.int16)
        )
        annotation_from_sensor = loader.get_transform(
            loader.sensor_frame_id(sample_data),
            loader.annotation_frame_id,
            sd=sample_data,
        )
        origins[channel] = annotation_from_sensor.translation.tolist()
    if not point_groups or not any(len(group) for group in point_groups):
        raise ValueError(f"sample {sample_token!r} has no finite LiDAR points")
    return (
        {
            "points_xyz": np.concatenate(point_groups).astype(np.float32, copy=False),
            "point_features": np.concatenate(intensity_groups).astype(
                np.float32, copy=False
            ),
            "point_timestamps_ns": np.concatenate(timestamp_groups).astype(
                np.uint64, copy=False
            ),
            "point_sensor_index": np.concatenate(sensor_index_groups).astype(
                np.int16, copy=False
            ),
        },
        origins,
        dropped,
    )


def _representative_lidar_sample_data(
    loader: Any,
    sample_token: str,
    lidar_channels: tuple[str, ...],
) -> Any:
    rows = {row.channel: row for row in loader.sample_data_for_sample(sample_token)}
    return rows[lidar_channels[0]]


def _track_documents(
    *,
    clip_id: str,
    rows: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    sample_order = {
        str(row["sample_token"]): index for index, row in enumerate(samples)
    }
    by_instance: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_instance.setdefault(str(row["instance_token"]), []).append(row)
    documents: list[dict[str, object]] = []
    for instance_token in sorted(by_instance):
        instance_rows = sorted(
            by_instance[instance_token],
            key=lambda row: sample_order[str(row["sample_token"])],
        )
        if len(instance_rows) < 2:
            continue
        categories = {str(row["category"]) for row in instance_rows}
        if len(categories) != 1:
            raise ValueError("one instance cannot change category")
        observations: list[dict[str, object]] = []
        for row in instance_rows:
            box = row["bbox_3d"]
            if not isinstance(box, Mapping):
                raise ValueError("bbox_3d must be an object")
            translation = _xyz(box["translation"], ("x", "y", "z"))
            size = _xyz(box["size"], ("length", "width", "height"))
            rotation = _xyz(box["rotation"], ("qx", "qy", "qz", "qw"))
            observations.append(
                {
                    "frame_id": str(row["sample_token"]),
                    "coarse_box": {
                        "center": translation,
                        "size_lwh": size,
                        "orientation_xyzw": rotation,
                    },
                    "score": None,
                    "kind": "observed",
                }
            )
        case_id = f"{clip_id}--{instance_token}"
        documents.append(
            {
                "contract_version": TRACK_CONTRACT,
                "case_id": case_id,
                "sequence_id": clip_id,
                "track_id": instance_token,
                "category": next(iter(categories)),
                "observations": observations,
            }
        )
    return documents


def _xyz(value: object, fields: tuple[str, ...]) -> list[float]:
    if not isinstance(value, Mapping):
        raise ValueError("box component must be an object")
    result = [float(value[field]) for field in fields]
    if not np.isfinite(result).all():
        raise ValueError("box values must be finite")
    return result


def _load_annotation_payload(
    value: str | Path | Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("annotation input must contain an object")
    nested = payload.get("annotation")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("candidate annotation must be an object")
        return nested, payload
    return payload, {}


def _annotation_input_sha256(value: str | Path | Mapping[str, Any]) -> str:
    if isinstance(value, Mapping):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    else:
        encoded = Path(value).read_bytes()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain an array of objects")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
