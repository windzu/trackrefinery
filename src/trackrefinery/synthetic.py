"""Generate deterministic full-frame development inputs and separate targets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trackrefinery.contracts import Box3D, Pose3D
from trackrefinery.geometry import compose_pose, inverse_pose, transform_points
from trackrefinery.serde import box_to_dict, pose_to_dict

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class CaseSpec:
    sequence_id: str
    role: str
    frame_count: int
    motion_per_frame: tuple[float, float]
    visibility: str = "complete"
    outlier_count: int = 0
    clutter_count: int = 0
    neighbor: bool = False
    expected_refinable: bool = True


@dataclass(frozen=True, slots=True)
class GeneratedDataset:
    root: Path
    inference_root: Path
    target_root: Path


CASES = (
    CaseSpec("static_complete", "development", 5, (0.0, 0.0)),
    CaseSpec("moving_complete", "development", 5, (0.8, 0.12)),
    CaseSpec("robust_outliers", "calibration", 5, (0.3, 0.0), outlier_count=80),
    CaseSpec(
        "partial_visibility",
        "calibration",
        4,
        (0.25, 0.0),
        visibility="partial",
        expected_refinable=False,
    ),
    CaseSpec(
        "nearby_clutter",
        "test",
        4,
        (0.4, 0.0),
        clutter_count=480,
        expected_refinable=False,
    ),
    CaseSpec("neighboring_tracks", "test", 4, (0.35, 0.0), neighbor=True),
)

TARGET_SIZE = np.asarray([4.45, 1.86, 1.62], dtype=np.float32)
NEIGHBOR_SIZE = np.asarray([4.9, 2.05, 1.8], dtype=np.float32)
SENSOR_IDS = ("lidar_front", "lidar_left")
SENSOR_ORIGINS = {
    "lidar_front": [-1.5, 0.0, 1.9],
    "lidar_left": [0.0, 0.85, 1.75],
}


def generate_dataset(output: str | Path, *, seed: int = 20260819) -> GeneratedDataset:
    root = Path(output).resolve()
    inference_root = root / "inference"
    target_root = root / "targets"
    inference_root.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)

    sequence_rows: list[dict[str, str]] = []
    case_rows: list[dict[str, str]] = []
    target_rows: list[dict[str, str]] = []
    root_rng = np.random.default_rng(seed)
    for spec in CASES:
        case_seed = int(root_rng.integers(0, np.iinfo(np.uint32).max))
        generated_cases = _generate_sequence(
            inference_root,
            target_root,
            spec,
            np.random.default_rng(case_seed),
        )
        sequence_rows.append(
            {
                "sequence_id": spec.sequence_id,
                "role": spec.role,
                "manifest_path": f"sources/{spec.sequence_id}/sequence.json",
            }
        )
        for case_id in generated_cases:
            case_rows.append(
                {
                    "case_id": case_id,
                    "sequence_id": spec.sequence_id,
                    "track_path": f"inputs/{case_id}/track.json",
                }
            )
            target_rows.append(
                {"case_id": case_id, "target_path": f"cases/{case_id}/target.json"}
            )

    _write_json(
        inference_root / "dataset.json",
        {
            "format": "trackrefinery-inference-dataset",
            "version": 1,
            "dataset_id": "synthetic-v1",
            "generator": {"seed": seed, "name": "trackrefinery.synthetic"},
            "sequences": sequence_rows,
            "cases": case_rows,
        },
    )
    _write_json(
        target_root / "targetset.json",
        {
            "format": "trackrefinery-target-dataset",
            "version": 1,
            "dataset_id": "synthetic-v1",
            "cases": target_rows,
        },
    )
    return GeneratedDataset(root, inference_root, target_root)


def _generate_sequence(
    inference_root: Path,
    target_root: Path,
    spec: CaseSpec,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    sequence_dir = inference_root / "sources" / spec.sequence_id
    frame_dir = sequence_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    target_observations: list[dict[str, Any]] = []
    coarse_observations: list[dict[str, Any]] = []
    neighbor_targets: list[dict[str, Any]] = []
    neighbor_coarse: list[dict[str, Any]] = []

    for frame_index in range(spec.frame_count):
        frame_id = f"{frame_index:06d}"
        timestamp_ns = 1_700_000_000_000_000_000 + frame_index * (
            100_000_000 + (frame_index % 2) * 7_000_000
        )
        world_from_annotation = Pose3D(
            translation_xyz=(
                frame_index * 0.48,
                0.08 * float(np.sin(frame_index)),
                0.0,
            ),
            orientation_xyzw=_yaw_quaternion(0.012 * frame_index),
        )
        annotation_from_world = inverse_pose(world_from_annotation)
        target_center_world = np.asarray(
            [
                12.0 + frame_index * spec.motion_per_frame[0],
                -0.4 + frame_index * spec.motion_per_frame[1],
                0.0,
            ],
            dtype=np.float32,
        )
        target_yaw_world = 0.13 + frame_index * 0.018
        target_pose_world = Pose3D(
            tuple(target_center_world), _yaw_quaternion(target_yaw_world)
        )
        target_pose_annotation = compose_pose(annotation_from_world, target_pose_world)
        target_points_world = _cuboid_surface_points(
            rng,
            target_center_world,
            TARGET_SIZE,
            target_yaw_world,
            600 if spec.visibility == "complete" else 65,
            visibility=spec.visibility,
        )
        point_groups_world = [
            target_points_world,
            _ground_points(rng, frame_index, 220),
        ]

        if spec.outlier_count:
            outliers = rng.uniform(
                target_center_world - np.asarray([5.0, 4.0, 1.5], dtype=np.float32),
                target_center_world + np.asarray([5.0, 4.0, 2.5], dtype=np.float32),
                size=(spec.outlier_count, 3),
            ).astype(np.float32)
            point_groups_world.append(outliers)
        if spec.clutter_count:
            clutter = rng.normal(
                loc=target_center_world
                + np.asarray([1.9, 1.25, 0.0], dtype=np.float32),
                scale=np.asarray([1.2, 0.42, 0.7], dtype=np.float32),
                size=(spec.clutter_count, 3),
            ).astype(np.float32)
            point_groups_world.append(clutter)

        if spec.neighbor:
            neighbor_center_world = target_center_world + np.asarray(
                [0.55, 2.35, 0.09], dtype=np.float32
            )
            neighbor_yaw_world = target_yaw_world - 0.06
            point_groups_world.append(
                _cuboid_surface_points(
                    rng,
                    neighbor_center_world,
                    NEIGHBOR_SIZE,
                    neighbor_yaw_world,
                    720,
                )
            )
            neighbor_pose_annotation = compose_pose(
                annotation_from_world,
                Pose3D(
                    tuple(neighbor_center_world),
                    _yaw_quaternion(neighbor_yaw_world),
                ),
            )
            neighbor_targets.append(
                _target_pose_row(frame_id, neighbor_pose_annotation)
            )
            neighbor_coarse.append(
                _coarse_observation(
                    frame_id,
                    neighbor_pose_annotation,
                    NEIGHBOR_SIZE,
                    frame_index,
                    lateral_sign=-1.0,
                )
            )

        points_world = np.concatenate(point_groups_world).astype(np.float32, copy=False)
        points_annotation = transform_points(
            points_world, annotation_from_world
        ).astype(np.float32)
        sensor_index = np.where(points_annotation[:, 1] >= 0, 1, 0).astype(np.int16)
        intensity = np.clip(rng.normal(0.62, 0.18, len(points_world)), 0.0, 1.0)
        features = intensity[:, None].astype(np.float32)
        point_offsets = rng.integers(
            -50_000_000, 50_000_001, len(points_world), dtype=np.int64
        )
        point_timestamps = (timestamp_ns + point_offsets).astype(np.uint64)
        point_path = frame_dir / f"{frame_id}.npz"
        np.savez_compressed(
            point_path,
            points_xyz=points_annotation,
            point_features=features,
            point_timestamps_ns=point_timestamps,
            point_sensor_index=sensor_index,
        )
        frame_rows.append(
            {
                "frame_id": frame_id,
                "timestamp_ns": timestamp_ns,
                "annotation_frame_id": "synthetic_base",
                "world_from_annotation": pose_to_dict(world_from_annotation),
                "points_path": f"frames/{frame_id}.npz",
                "sensor_origins": SENSOR_ORIGINS,
            }
        )
        target_observations.append(_target_pose_row(frame_id, target_pose_annotation))
        coarse_observations.append(
            _coarse_observation(
                frame_id,
                target_pose_annotation,
                TARGET_SIZE,
                frame_index,
            )
        )

    _write_json(
        sequence_dir / "sequence.json",
        {
            "contract_version": "trackrefinery-frame-sequence-v1",
            "sequence_id": spec.sequence_id,
            "sensors": [
                {"sensor_id": sensor_id, "modality": "lidar"}
                for sensor_id in SENSOR_IDS
            ],
            "feature_names": ["intensity"],
            "frames": frame_rows,
        },
    )
    case_ids = [spec.sequence_id]
    _write_track(
        inference_root,
        spec.sequence_id,
        spec.sequence_id,
        "target",
        "car",
        coarse_observations,
    )
    _write_target(
        target_root,
        spec.sequence_id,
        spec.sequence_id,
        "target",
        TARGET_SIZE,
        target_observations,
        spec.expected_refinable,
    )
    if spec.neighbor:
        neighbor_case_id = f"{spec.sequence_id}_neighbor"
        case_ids.append(neighbor_case_id)
        _write_track(
            inference_root,
            neighbor_case_id,
            spec.sequence_id,
            "neighbor",
            "truck",
            neighbor_coarse,
        )
        _write_target(
            target_root,
            neighbor_case_id,
            spec.sequence_id,
            "neighbor",
            NEIGHBOR_SIZE,
            neighbor_targets,
            True,
        )
    return tuple(case_ids)


def _write_track(
    root: Path,
    case_id: str,
    sequence_id: str,
    track_id: str,
    category: str,
    observations: list[dict[str, Any]],
) -> None:
    _write_json(
        root / "inputs" / case_id / "track.json",
        {
            "contract_version": "trackrefinery-instance-track-v1",
            "case_id": case_id,
            "sequence_id": sequence_id,
            "track_id": track_id,
            "category": category,
            "observations": observations,
        },
    )


def _write_target(
    root: Path,
    case_id: str,
    sequence_id: str,
    track_id: str,
    size: FloatArray,
    observations: list[dict[str, Any]],
    expected_refinable: bool,
) -> None:
    _write_json(
        root / "cases" / case_id / "target.json",
        {
            "contract_version": "trackrefinery-gold-target-v1",
            "case_id": case_id,
            "sequence_id": sequence_id,
            "track_id": track_id,
            "canonical_size_lwh": size.tolist(),
            "frame_poses": observations,
            "expected_refinable": expected_refinable,
        },
    )


def _cuboid_surface_points(
    rng: np.random.Generator,
    center: FloatArray,
    size: FloatArray,
    yaw: float,
    count: int,
    *,
    visibility: str = "complete",
) -> FloatArray:
    local = rng.uniform(-0.5, 0.5, size=(count, 3)).astype(np.float32) * size
    if visibility == "partial":
        face = rng.integers(0, 2, count)
        local[face == 0, 0] = -size[0] / 2
        local[face == 1, 1] = size[1] / 2
    else:
        face = rng.integers(0, 6, count)
        axis = face // 2
        sign = np.where(face % 2 == 0, -0.5, 0.5)
        local[np.arange(count), axis] = size[axis] * sign
    rotation = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    noise = rng.normal(0.0, 0.012, size=local.shape).astype(np.float32)
    return ((local + noise) @ rotation.T + center).astype(np.float32)


def _ground_points(
    rng: np.random.Generator, frame_index: int, count: int
) -> FloatArray:
    xy = rng.uniform(
        [-2.0 + frame_index * 0.48, -8.0],
        [28.0 + frame_index * 0.48, 8.0],
        size=(count, 2),
    ).astype(np.float32)
    z = rng.normal(-0.82, 0.012, size=(count, 1)).astype(np.float32)
    return np.concatenate((xy, z), axis=1)


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))


def _target_pose_row(frame_id: str, pose: Pose3D) -> dict[str, Any]:
    return {"frame_id": frame_id, "pose": pose_to_dict(pose), "evaluable": True}


def _coarse_observation(
    frame_id: str,
    target_pose: Pose3D,
    size: FloatArray,
    frame_index: int,
    *,
    lateral_sign: float = 1.0,
) -> dict[str, Any]:
    phase = float(frame_index)
    center_error = np.asarray(
        [0.18 * np.sin(phase + 0.4), lateral_sign * 0.14 * np.cos(phase), 0.05]
    )
    center = np.asarray(target_pose.translation_xyz) + center_error
    size_scale = np.asarray([1.13 + 0.025 * np.sin(phase), 0.89, 1.08])
    target_yaw = _yaw_from_quaternion(target_pose.orientation_xyzw)
    box = Box3D(
        center=tuple(center),
        size_lwh=tuple(size * size_scale),
        orientation_xyzw=_yaw_quaternion(target_yaw + 0.055 * np.cos(phase)),
    )
    return {
        "frame_id": frame_id,
        "coarse_box": box_to_dict(box),
        "score": float(0.86 - 0.025 * (frame_index % 3)),
        "kind": "observed",
    }


def _yaw_from_quaternion(value: tuple[float, float, float, float]) -> float:
    x, y, z, w = value
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="destination bundle directory")
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args(argv)
    generated = generate_dataset(args.output, seed=args.seed)
    print(f"generated inference data at {generated.inference_root}")
    print(f"generated separate targets at {generated.target_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
