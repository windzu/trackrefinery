"""Evaluation-only gold targets, kept separate from inference datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trackrefinery.contracts import Pose3D
from trackrefinery.serde import (
    pose_from_dict,
    require_float_tuple,
    require_list,
    require_object,
    require_string,
)

TARGET_FORMAT = "trackrefinery-target-dataset"
TARGET_VERSION = 1
TARGET_CONTRACT = "trackrefinery-gold-target-v1"


@dataclass(frozen=True, slots=True)
class GoldFramePose:
    frame_id: str
    pose: Pose3D
    evaluable: bool = True

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if not isinstance(self.evaluable, bool):
            raise ValueError("evaluable must be a boolean")


@dataclass(frozen=True, slots=True)
class GoldTarget:
    case_id: str
    sequence_id: str
    track_id: str
    canonical_size_lwh: tuple[float, float, float]
    frame_poses: tuple[GoldFramePose, ...]
    expected_refinable: bool = True

    def __post_init__(self) -> None:
        if not self.case_id or not self.sequence_id or not self.track_id:
            raise ValueError("target identifiers must not be empty")
        size = tuple(float(item) for item in self.canonical_size_lwh)
        if (
            len(size) != 3
            or not all(math.isfinite(item) for item in size)
            or any(item <= 0 for item in size)
        ):
            raise ValueError(
                "canonical_size_lwh must contain three finite positive values"
            )
        frame_ids = [item.frame_id for item in self.frame_poses]
        if not frame_ids or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("target frame poses must be unique and non-empty")
        if not any(item.evaluable for item in self.frame_poses):
            raise ValueError("a gold target needs at least one evaluable frame")
        if not isinstance(self.expected_refinable, bool):
            raise ValueError("expected_refinable must be a boolean")
        object.__setattr__(self, "canonical_size_lwh", size)


@dataclass(frozen=True, slots=True)
class TargetEntry:
    case_id: str
    target_path: str


class TargetDataset:
    """A separate evaluator-only dataset with no inference loading behavior."""

    def __init__(
        self, root: Path, dataset_id: str, entries: tuple[TargetEntry, ...]
    ) -> None:
        self.root = root
        self.dataset_id = dataset_id
        self.entries = entries

    @classmethod
    def open(cls, root: str | Path) -> TargetDataset:
        root_path = Path(root).resolve()
        payload = _read_json(root_path / "targetset.json")
        if payload.get("format") != TARGET_FORMAT:
            raise ValueError(f"format must be {TARGET_FORMAT!r}")
        if payload.get("version") != TARGET_VERSION:
            raise ValueError(f"version must be {TARGET_VERSION}")
        dataset_id = require_string(payload.get("dataset_id"), "dataset_id")
        entries: list[TargetEntry] = []
        seen: set[str] = set()
        for value in require_list(payload.get("cases"), "cases"):
            row = require_object(value, "case")
            case_id = require_string(row.get("case_id"), "case_id")
            if case_id in seen:
                raise ValueError(f"duplicate target case_id {case_id!r}")
            seen.add(case_id)
            target_path = require_string(row.get("target_path"), "target_path")
            _resolve_relative(root_path, target_path)
            entries.append(TargetEntry(case_id, target_path))
        if not entries:
            raise ValueError("target dataset must contain cases")
        return cls(root_path, dataset_id, tuple(entries))

    def load_target(self, case_id: str) -> GoldTarget:
        entries = {item.case_id: item for item in self.entries}
        if case_id not in entries:
            raise KeyError(f"unknown target case_id {case_id!r}")
        payload = _read_json(_resolve_relative(self.root, entries[case_id].target_path))
        if payload.get("contract_version") != TARGET_CONTRACT:
            raise ValueError(f"target contract_version must be {TARGET_CONTRACT!r}")
        if payload.get("case_id") != case_id:
            raise ValueError("target case_id does not match target index")
        frame_poses = tuple(
            GoldFramePose(
                frame_id=require_string(
                    require_object(value, "frame_pose").get("frame_id"), "frame_id"
                ),
                pose=pose_from_dict(
                    require_object(value, "frame_pose").get("pose"), "pose"
                ),
                evaluable=require_object(value, "frame_pose").get("evaluable", True),
            )
            for value in require_list(payload.get("frame_poses"), "frame_poses")
        )
        return GoldTarget(
            case_id=case_id,
            sequence_id=require_string(payload.get("sequence_id"), "sequence_id"),
            track_id=require_string(payload.get("track_id"), "track_id"),
            canonical_size_lwh=require_float_tuple(
                payload.get("canonical_size_lwh"), 3, "canonical_size_lwh"
            ),
            frame_poses=frame_poses,
            expected_refinable=payload.get("expected_refinable", True),
        )


def _read_json(path: Path) -> dict[str, Any]:
    return require_object(json.loads(path.read_text(encoding="utf-8")), str(path))


def _resolve_relative(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("target paths must be relative")
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("target paths must remain inside the target root")
    return resolved
