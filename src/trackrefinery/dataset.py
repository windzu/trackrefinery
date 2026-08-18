"""Strict source-only reader for portable refinement inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trackrefinery.contracts import (
    FrameCloud,
    InstanceTrack,
    ObservationKind,
    RefinementCase,
    TrackObservation,
)
from trackrefinery.serde import (
    box_from_dict,
    pose_from_dict,
    require_list,
    require_object,
    require_string,
)

INFERENCE_FORMAT = "trackrefinery-inference-dataset"
INFERENCE_VERSION = 1
SEQUENCE_CONTRACT = "trackrefinery-frame-sequence-v1"
TRACK_CONTRACT = "trackrefinery-instance-track-v1"
ALLOWED_ROLES = frozenset({"development", "calibration", "test", "qualitative"})


class DatasetContractError(ValueError):
    """Raised when a portable dataset violates the versioned contract."""


@dataclass(frozen=True, slots=True)
class SequenceEntry:
    sequence_id: str
    role: str
    manifest_path: str


@dataclass(frozen=True, slots=True)
class CaseEntry:
    case_id: str
    sequence_id: str
    track_path: str


@dataclass(frozen=True, slots=True)
class FrameSequence:
    sequence_id: str
    frames: tuple[FrameCloud, ...]


class InferenceDataset:
    """Load source-only cases; this class has no target API or target path."""

    def __init__(
        self,
        root: Path,
        dataset_id: str,
        sequences: tuple[SequenceEntry, ...],
        cases: tuple[CaseEntry, ...],
    ) -> None:
        self.root = root
        self.dataset_id = dataset_id
        self.sequences = sequences
        self.cases = cases
        self._sequence_cache: dict[str, FrameSequence] = {}

    @classmethod
    def open(cls, root: str | Path) -> InferenceDataset:
        root_path = Path(root).resolve()
        payload = _read_json(root_path / "dataset.json")
        if payload.get("format") != INFERENCE_FORMAT:
            raise DatasetContractError(f"format must be {INFERENCE_FORMAT!r}")
        if payload.get("version") != INFERENCE_VERSION:
            raise DatasetContractError(f"version must be {INFERENCE_VERSION}")
        dataset_id = _safe_id(payload.get("dataset_id"), "dataset_id")

        sequences: list[SequenceEntry] = []
        sequence_ids: set[str] = set()
        for index, value in enumerate(
            require_list(payload.get("sequences"), "sequences")
        ):
            row = require_object(value, f"sequences[{index}]")
            sequence_id = _safe_id(row.get("sequence_id"), "sequence_id")
            role = require_string(row.get("role"), "role")
            if role not in ALLOWED_ROLES:
                raise DatasetContractError(
                    f"role must be one of {sorted(ALLOWED_ROLES)}"
                )
            if sequence_id in sequence_ids:
                raise DatasetContractError(f"duplicate sequence_id {sequence_id!r}")
            sequence_ids.add(sequence_id)
            manifest = require_string(row.get("manifest_path"), "manifest_path")
            _resolve_relative(root_path, manifest, "manifest_path")
            sequences.append(SequenceEntry(sequence_id, role, manifest))

        cases: list[CaseEntry] = []
        case_ids: set[str] = set()
        for index, value in enumerate(require_list(payload.get("cases"), "cases")):
            row = require_object(value, f"cases[{index}]")
            case_id = _safe_id(row.get("case_id"), "case_id")
            sequence_id = _safe_id(row.get("sequence_id"), "sequence_id")
            if sequence_id not in sequence_ids:
                raise DatasetContractError("case references an unknown sequence")
            if case_id in case_ids:
                raise DatasetContractError(f"duplicate case_id {case_id!r}")
            case_ids.add(case_id)
            track_path = require_string(row.get("track_path"), "track_path")
            _resolve_relative(root_path, track_path, "track_path")
            cases.append(CaseEntry(case_id, sequence_id, track_path))
        if not sequences or not cases:
            raise DatasetContractError("dataset must contain sequences and cases")
        return cls(root_path, dataset_id, tuple(sequences), tuple(cases))

    def load_sequence(self, sequence_id: str) -> FrameSequence:
        if sequence_id in self._sequence_cache:
            return self._sequence_cache[sequence_id]
        entries = {item.sequence_id: item for item in self.sequences}
        if sequence_id not in entries:
            raise KeyError(f"unknown sequence_id {sequence_id!r}")
        entry = entries[sequence_id]
        manifest_path = _resolve_relative(
            self.root, entry.manifest_path, "manifest_path"
        )
        payload = _read_json(manifest_path)
        if payload.get("contract_version") != SEQUENCE_CONTRACT:
            raise DatasetContractError(
                f"sequence contract_version must be {SEQUENCE_CONTRACT!r}"
            )
        if payload.get("sequence_id") != sequence_id:
            raise DatasetContractError("sequence_id does not match dataset index")
        sensor_ids = tuple(
            _safe_id(
                require_object(row, "sensor").get("sensor_id"),
                "sensor.sensor_id",
            )
            for row in require_list(payload.get("sensors", []), "sensors")
        )
        if len(sensor_ids) != len(set(sensor_ids)):
            raise DatasetContractError("sensor IDs must be unique")
        feature_names = tuple(
            require_string(item, "feature_name")
            for item in require_list(payload.get("feature_names", []), "feature_names")
        )

        frames: list[FrameCloud] = []
        sequence_dir = manifest_path.parent
        for index, value in enumerate(require_list(payload.get("frames"), "frames")):
            row = require_object(value, f"frames[{index}]")
            points_path = _resolve_relative(
                sequence_dir,
                require_string(row.get("points_path"), "points_path"),
                "points_path",
            )
            arrays = _load_points(points_path)
            origins = {
                _safe_id(sensor_id, "sensor origin ID"): tuple(float(v) for v in origin)
                for sensor_id, origin in require_object(
                    row.get("sensor_origins", {}), "sensor_origins"
                ).items()
            }
            try:
                frame = FrameCloud(
                    frame_id=_safe_id(row.get("frame_id"), "frame_id"),
                    timestamp_ns=row.get("timestamp_ns"),
                    annotation_frame_id=require_string(
                        row.get("annotation_frame_id"), "annotation_frame_id"
                    ),
                    world_from_annotation=pose_from_dict(
                        row.get("world_from_annotation"), "world_from_annotation"
                    ),
                    points_xyz=arrays["points_xyz"],
                    point_features=arrays.get("point_features"),
                    feature_names=feature_names,
                    point_timestamps_ns=arrays.get("point_timestamps_ns"),
                    point_sensor_index=arrays.get("point_sensor_index"),
                    sensor_ids=sensor_ids,
                    sensor_origins=origins,
                )
            except (TypeError, ValueError) as error:
                raise DatasetContractError(
                    f"invalid frame at {manifest_path}:{index}: {error}"
                ) from error
            frames.append(frame)
        try:
            sequence = FrameSequence(sequence_id, tuple(frames))
            frame_ids = [frame.frame_id for frame in sequence.frames]
            timestamps = [frame.timestamp_ns for frame in sequence.frames]
            if len(frame_ids) != len(set(frame_ids)):
                raise ValueError("frame IDs must be unique")
            if timestamps != sorted(timestamps) or len(timestamps) != len(
                set(timestamps)
            ):
                raise ValueError("frames must have strictly increasing timestamps")
        except ValueError as error:
            raise DatasetContractError(
                f"invalid sequence {sequence_id!r}: {error}"
            ) from error
        self._sequence_cache[sequence_id] = sequence
        return sequence

    def load_case(self, case_id: str) -> RefinementCase:
        entries = {item.case_id: item for item in self.cases}
        if case_id not in entries:
            raise KeyError(f"unknown case_id {case_id!r}")
        entry = entries[case_id]
        track_path = _resolve_relative(self.root, entry.track_path, "track_path")
        payload = _read_json(track_path)
        if payload.get("contract_version") != TRACK_CONTRACT:
            raise DatasetContractError(
                f"track contract_version must be {TRACK_CONTRACT!r}"
            )
        if payload.get("case_id") != case_id:
            raise DatasetContractError("case_id does not match dataset index")
        if payload.get("sequence_id") != entry.sequence_id:
            raise DatasetContractError("track sequence_id does not match dataset index")
        observations: list[TrackObservation] = []
        for index, value in enumerate(
            require_list(payload.get("observations"), "observations")
        ):
            row = require_object(value, f"observations[{index}]")
            try:
                observations.append(
                    TrackObservation(
                        frame_id=_safe_id(row.get("frame_id"), "frame_id"),
                        coarse_box=box_from_dict(row.get("coarse_box"), "coarse_box"),
                        score=row.get("score"),
                        kind=ObservationKind(row.get("kind", "observed")),
                    )
                )
            except (TypeError, ValueError) as error:
                raise DatasetContractError(
                    f"invalid observation at {track_path}:{index}: {error}"
                ) from error
        try:
            track = InstanceTrack(
                track_id=_safe_id(payload.get("track_id"), "track_id"),
                sequence_id=entry.sequence_id,
                category=payload.get("category"),
                observations=tuple(observations),
            )
            sequence = self.load_sequence(entry.sequence_id)
            frames_by_id = {frame.frame_id: frame for frame in sequence.frames}
            frames = tuple(frames_by_id[item.frame_id] for item in observations)
            return RefinementCase(case_id=case_id, frames=frames, track=track)
        except KeyError as error:
            raise DatasetContractError("track references an unknown frame") from error
        except ValueError as error:
            raise DatasetContractError(f"invalid case {case_id!r}: {error}") from error

    def validate(self) -> tuple[RefinementCase, ...]:
        return tuple(self.load_case(entry.case_id) for entry in self.cases)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return require_object(json.loads(path.read_text(encoding="utf-8")), str(path))
    except FileNotFoundError as error:
        raise DatasetContractError(f"missing file: {path}") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise DatasetContractError(f"invalid JSON in {path}: {error}") from error


def _load_points(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    allowed = {
        "points_xyz",
        "point_features",
        "point_timestamps_ns",
        "point_sensor_index",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            if "points_xyz" not in names:
                raise DatasetContractError(f"{path} has no points_xyz array")
            if not names.issubset(allowed):
                raise DatasetContractError(
                    f"{path} contains unsupported arrays: {sorted(names - allowed)}"
                )
            return {name: archive[name] for name in names}
    except (OSError, ValueError) as error:
        raise DatasetContractError(f"cannot load point file {path}: {error}") from error


def _safe_id(value: object, name: str) -> str:
    result = require_string(value, name)
    if result in {".", ".."} or "/" in result or "\\" in result:
        raise DatasetContractError(f"{name} must be a safe path component")
    return result


def _resolve_relative(root: Path, value: str, name: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise DatasetContractError(f"{name} must be relative")
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise DatasetContractError(f"{name} must remain inside its root")
    return resolved
