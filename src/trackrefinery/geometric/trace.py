"""Immutable point-evidence traces and portable sidecar serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trackrefinery.contracts import RefinementCase, RefinementOutcome

EVIDENCE_TRACE_CONTRACT = "trackrefinery-geometric-evidence-trace-v1"


class EvidenceState(IntEnum):
    """Point ownership state stored in compact trace arrays."""

    TARGET = 1
    AMBIGUOUS = 2
    BACKGROUND = 3
    GROUND = 4


@dataclass(frozen=True, slots=True)
class GroundPlaneEstimate:
    """Robust plane represented as ``z = a*x + b*y + c``."""

    z_from_xyc: tuple[float, float, float]
    normal_xyz: tuple[float, float, float]
    candidate_count: int
    inlier_count: int
    rmse_m: float
    tilt_deg: float

    def __post_init__(self) -> None:
        coefficients = tuple(float(value) for value in self.z_from_xyc)
        normal = tuple(float(value) for value in self.normal_xyz)
        numeric = (*coefficients, *normal, float(self.rmse_m), float(self.tilt_deg))
        if len(coefficients) != 3 or len(normal) != 3 or not np.isfinite(numeric).all():
            raise ValueError("ground plane values must be finite triplets")
        if self.candidate_count < 3:
            raise ValueError("ground plane requires at least three candidates")
        if not 3 <= self.inlier_count <= self.candidate_count:
            raise ValueError("ground plane inlier count is invalid")
        if self.rmse_m < 0 or not 0 <= self.tilt_deg < 90:
            raise ValueError("ground plane residual or tilt is invalid")
        norm = float(np.linalg.norm(normal))
        if not np.isclose(norm, 1.0, atol=1e-6) or normal[2] <= 0:
            raise ValueError("ground plane normal must be unit length with positive Z")
        object.__setattr__(self, "z_from_xyc", coefficients)
        object.__setattr__(self, "normal_xyz", normal)

    def to_dict(self) -> dict[str, object]:
        return {
            "z_from_xyc": list(self.z_from_xyc),
            "normal_xyz": list(self.normal_xyz),
            "candidate_count": self.candidate_count,
            "inlier_count": self.inlier_count,
            "rmse_m": self.rmse_m,
            "tilt_deg": self.tilt_deg,
        }

    @classmethod
    def from_dict(cls, value: object) -> GroundPlaneEstimate:
        if not isinstance(value, dict):
            raise ValueError("ground_plane must be an object")
        return cls(
            z_from_xyc=_float_triplet(value.get("z_from_xyc"), "z_from_xyc"),
            normal_xyz=_float_triplet(value.get("normal_xyz"), "normal_xyz"),
            candidate_count=_integer(value.get("candidate_count"), "candidate_count"),
            inlier_count=_integer(value.get("inlier_count"), "inlier_count"),
            rmse_m=_number(value.get("rmse_m"), "rmse_m"),
            tilt_deg=_number(value.get("tilt_deg"), "tilt_deg"),
        )


@dataclass(frozen=True, slots=True)
class FrameEvidenceTrace:
    """Point states for one ROI, indexed back into the immutable frame cloud."""

    frame_id: str
    roi_point_indices: NDArray[np.int64]
    point_states: NDArray[np.uint8]
    ground_plane: GroundPlaneEstimate | None = None
    represented_sensor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be a non-empty string")
        indices = np.asarray(self.roi_point_indices)
        states = np.asarray(self.point_states)
        if indices.dtype != np.int64 or indices.ndim != 1:
            raise ValueError("roi_point_indices must be int64 with shape [R]")
        if states.dtype != np.uint8 or states.shape != indices.shape:
            raise ValueError("point_states must be uint8 and align with ROI indices")
        if len(indices) and (np.any(indices < 0) or np.any(np.diff(indices) <= 0)):
            raise ValueError("ROI indices must be unique, ordered, and non-negative")
        valid_states = np.asarray(
            [value.value for value in EvidenceState], dtype=np.uint8
        )
        if len(states) and not np.isin(states, valid_states).all():
            raise ValueError("point_states contains an unsupported evidence state")
        sensors = tuple(self.represented_sensor_ids)
        if len(sensors) != len(set(sensors)) or any(not value for value in sensors):
            raise ValueError("represented sensor IDs must be unique and non-empty")
        indices = indices.copy()
        states = states.copy()
        indices.setflags(write=False)
        states.setflags(write=False)
        object.__setattr__(self, "roi_point_indices", indices)
        object.__setattr__(self, "point_states", states)
        object.__setattr__(self, "represented_sensor_ids", sensors)

    def count(self, state: EvidenceState) -> int:
        return int(np.count_nonzero(self.point_states == state.value))

    @property
    def counts(self) -> dict[str, int]:
        return {state.name.lower(): self.count(state) for state in EvidenceState}

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "roi_point_count": len(self.roi_point_indices),
            "point_state_counts": self.counts,
            "represented_sensor_ids": list(self.represented_sensor_ids),
            "ground_plane": (
                None if self.ground_plane is None else self.ground_plane.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class GeometricRefinementTrace:
    """One deterministic algorithm-stage trace for a refinement case."""

    case_id: str
    track_id: str
    algorithm_version: str
    config_schema_version: str
    config_sha256: str
    settings_json: str
    stage: str
    frames: tuple[FrameEvidenceTrace, ...]

    def __post_init__(self) -> None:
        string_values = (
            self.case_id,
            self.track_id,
            self.algorithm_version,
            self.config_schema_version,
            self.config_sha256,
            self.stage,
        )
        if any(not isinstance(value, str) or not value for value in string_values):
            raise ValueError("trace identifiers must be non-empty strings")
        if len(self.config_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.config_sha256
        ):
            raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
        try:
            settings = json.loads(self.settings_json)
        except json.JSONDecodeError as error:
            raise ValueError("settings_json must contain canonical JSON") from error
        if not isinstance(settings, dict):
            raise ValueError("settings_json must contain an object")
        canonical = json.dumps(
            settings, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if canonical != self.settings_json:
            raise ValueError("settings_json must use canonical JSON encoding")
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if digest != self.config_sha256:
            raise ValueError("config_sha256 does not match settings_json")
        frames = tuple(self.frames)
        frame_ids = [frame.frame_id for frame in frames]
        if not frames or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("trace frames must be non-empty and unique")
        object.__setattr__(self, "frames", frames)

    @property
    def settings(self) -> dict[str, object]:
        value = json.loads(self.settings_json)
        if not isinstance(value, dict):
            raise AssertionError("validated trace settings must be an object")
        return value

    @property
    def total_counts(self) -> dict[str, int]:
        return {
            state.name.lower(): sum(frame.count(state) for frame in self.frames)
            for state in EvidenceState
        }

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "contract_version": EVIDENCE_TRACE_CONTRACT,
            "case_id": self.case_id,
            "track_id": self.track_id,
            "algorithm_version": self.algorithm_version,
            "config_schema_version": self.config_schema_version,
            "config_sha256": self.config_sha256,
            "settings": self.settings,
            "stage": self.stage,
            "total_point_state_counts": self.total_counts,
            "frames": [frame.to_summary_dict() for frame in self.frames],
        }


@dataclass(frozen=True, slots=True)
class GeometricRefinementRun:
    """Development result containing the public outcome and its point trace."""

    outcome: RefinementOutcome
    trace: GeometricRefinementTrace

    def __post_init__(self) -> None:
        if self.outcome.track_id != self.trace.track_id:
            raise ValueError("run outcome and trace track IDs do not match")


def validate_geometric_trace(
    case: RefinementCase, trace: GeometricRefinementTrace
) -> None:
    """Validate trace ownership and point-index alignment with one case."""

    if trace.case_id != case.case_id or trace.track_id != case.track.track_id:
        raise ValueError("evidence trace does not belong to the refinement case")
    expected = [frame.frame_id for frame in case.frames]
    actual = [frame.frame_id for frame in trace.frames]
    if actual != expected:
        raise ValueError("evidence trace must preserve every input frame")
    for frame, frame_trace in zip(case.frames, trace.frames, strict=True):
        if len(frame_trace.roi_point_indices) and int(
            frame_trace.roi_point_indices[-1]
        ) >= len(frame.points_xyz):
            raise ValueError("evidence trace references a point outside its frame")
        if frame.point_sensor_index is None and frame_trace.represented_sensor_ids:
            raise ValueError("trace cannot claim sensors without point provenance")
        if frame.point_sensor_index is not None:
            represented = tuple(
                frame.sensor_ids[index]
                for index in sorted(
                    set(
                        int(value)
                        for value in frame.point_sensor_index[
                            frame_trace.roi_point_indices
                        ]
                    )
                )
            )
            if represented != frame_trace.represented_sensor_ids:
                raise ValueError("trace represented sensors do not match ROI points")


def write_geometric_trace(
    output_dir: str | Path, trace: GeometricRefinementTrace
) -> tuple[Path, Path]:
    """Write JSON metadata and compact point masks without embedding frame points."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray[np.generic]] = {}
    manifest = trace.to_summary_dict()
    frame_rows = manifest["frames"]
    if not isinstance(frame_rows, list):
        raise AssertionError("trace frame summaries must be a list")
    for index, (frame, row) in enumerate(zip(trace.frames, frame_rows, strict=True)):
        index_key = f"frame_{index:06d}_roi_indices"
        state_key = f"frame_{index:06d}_point_states"
        arrays[index_key] = frame.roi_point_indices
        arrays[state_key] = frame.point_states
        if not isinstance(row, dict):
            raise AssertionError("trace frame summary must be an object")
        row["roi_indices_key"] = index_key
        row["point_states_key"] = state_key
    manifest["arrays_path"] = "evidence_masks.npz"

    arrays_path = output / "evidence_masks.npz"
    manifest_path = output / "evidence_trace.json"
    np.savez_compressed(arrays_path, **arrays)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path, arrays_path


def read_geometric_trace(path: str | Path) -> GeometricRefinementTrace:
    """Read a trace from its directory or ``evidence_trace.json`` path."""

    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "evidence_trace.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("contract_version") != EVIDENCE_TRACE_CONTRACT
    ):
        raise ValueError("unsupported geometric evidence trace contract")
    arrays_name = payload.get("arrays_path")
    if not isinstance(arrays_name, str) or not arrays_name:
        raise ValueError("trace arrays_path must be a non-empty string")
    arrays_path = (manifest_path.parent / arrays_name).resolve()
    if arrays_path.parent != manifest_path.parent:
        raise ValueError("trace arrays_path must remain beside its manifest")
    frame_rows = payload.get("frames")
    if not isinstance(frame_rows, list) or not frame_rows:
        raise ValueError("trace frames must be a non-empty list")
    frames: list[FrameEvidenceTrace] = []
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for row in frame_rows:
            if not isinstance(row, dict):
                raise ValueError("trace frame row must be an object")
            index_key = row.get("roi_indices_key")
            state_key = row.get("point_states_key")
            if not isinstance(index_key, str) or not isinstance(state_key, str):
                raise ValueError("trace frame array keys must be strings")
            ground = row.get("ground_plane")
            sensors = row.get("represented_sensor_ids")
            if not isinstance(sensors, list) or any(
                not isinstance(value, str) for value in sensors
            ):
                raise ValueError("represented_sensor_ids must be a string list")
            frames.append(
                FrameEvidenceTrace(
                    frame_id=_string(row.get("frame_id"), "frame_id"),
                    roi_point_indices=np.asarray(arrays[index_key], dtype=np.int64),
                    point_states=np.asarray(arrays[state_key], dtype=np.uint8),
                    ground_plane=(
                        None
                        if ground is None
                        else GroundPlaneEstimate.from_dict(ground)
                    ),
                    represented_sensor_ids=tuple(sensors),
                )
            )
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("trace settings must be an object")
    settings_json = json.dumps(
        settings, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return GeometricRefinementTrace(
        case_id=_string(payload.get("case_id"), "case_id"),
        track_id=_string(payload.get("track_id"), "track_id"),
        algorithm_version=_string(
            payload.get("algorithm_version"), "algorithm_version"
        ),
        config_schema_version=_string(
            payload.get("config_schema_version"), "config_schema_version"
        ),
        config_sha256=_string(payload.get("config_sha256"), "config_sha256"),
        settings_json=settings_json,
        stage=_string(payload.get("stage"), "stage"),
        frames=tuple(frames),
    )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float_triplet(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-value list")
    return tuple(_number(item, f"{name}[]") for item in value)  # type: ignore[return-value]
