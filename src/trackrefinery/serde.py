"""JSON serialization for public values and refinement outcomes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trackrefinery.contracts import (
    Box3D,
    InsufficientEvidence,
    Pose3D,
    RefinedFramePose,
    RefinementOutcome,
    RefinementSuccess,
)

RESULT_CONTRACT = "trackrefinery-refinement-result-v1"


def pose_to_dict(pose: Pose3D) -> dict[str, object]:
    return {
        "translation_xyz": list(pose.translation_xyz),
        "orientation_xyzw": list(pose.orientation_xyzw),
    }


def pose_from_dict(value: object, name: str = "pose") -> Pose3D:
    payload = require_object(value, name)
    return Pose3D(
        translation_xyz=require_float_tuple(
            payload.get("translation_xyz"), 3, f"{name}.translation_xyz"
        ),
        orientation_xyzw=require_float_tuple(
            payload.get("orientation_xyzw"), 4, f"{name}.orientation_xyzw"
        ),
    )


def box_to_dict(box: Box3D) -> dict[str, object]:
    return {
        "center": list(box.center),
        "size_lwh": list(box.size_lwh),
        "orientation_xyzw": list(box.orientation_xyzw),
    }


def box_from_dict(value: object, name: str = "box") -> Box3D:
    payload = require_object(value, name)
    return Box3D(
        center=require_float_tuple(payload.get("center"), 3, f"{name}.center"),
        size_lwh=require_float_tuple(payload.get("size_lwh"), 3, f"{name}.size_lwh"),
        orientation_xyzw=require_float_tuple(
            payload.get("orientation_xyzw"), 4, f"{name}.orientation_xyzw"
        ),
    )


def outcome_to_dict(case_id: str, outcome: RefinementOutcome) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_version": RESULT_CONTRACT,
        "case_id": case_id,
        "track_id": outcome.track_id,
        "status": outcome.status,
        "diagnostics": thaw_json(outcome.diagnostics),
    }
    if isinstance(outcome, RefinementSuccess):
        payload["canonical_size_lwh"] = list(outcome.canonical_size_lwh)
        payload["frame_poses"] = [
            {"frame_id": item.frame_id, "pose": pose_to_dict(item.pose)}
            for item in outcome.frame_poses
        ]
    else:
        payload["reasons"] = list(outcome.reasons)
    return payload


def outcome_from_dict(value: object) -> tuple[str, RefinementOutcome]:
    payload = require_object(value, "result")
    if payload.get("contract_version") != RESULT_CONTRACT:
        raise ValueError(f"contract_version must be {RESULT_CONTRACT!r}")
    case_id = require_string(payload.get("case_id"), "case_id")
    track_id = require_string(payload.get("track_id"), "track_id")
    diagnostics = require_object(payload.get("diagnostics", {}), "diagnostics")
    status = payload.get("status")
    if status == "success":
        frame_rows = require_list(payload.get("frame_poses"), "frame_poses")
        outcome: RefinementOutcome = RefinementSuccess(
            track_id=track_id,
            canonical_size_lwh=require_float_tuple(
                payload.get("canonical_size_lwh"), 3, "canonical_size_lwh"
            ),
            frame_poses=tuple(
                RefinedFramePose(
                    frame_id=require_string(
                        require_object(row, "frame_pose").get("frame_id"),
                        "frame_pose.frame_id",
                    ),
                    pose=pose_from_dict(
                        require_object(row, "frame_pose").get("pose"),
                        "frame_pose.pose",
                    ),
                )
                for row in frame_rows
            ),
            diagnostics=diagnostics,
        )
    elif status == "insufficient_evidence":
        reasons = require_list(payload.get("reasons"), "reasons")
        outcome = InsufficientEvidence(
            track_id=track_id,
            reasons=tuple(require_string(item, "reason") for item in reasons),
            diagnostics=diagnostics,
        )
    else:
        raise ValueError("status must be 'success' or 'insufficient_evidence'")
    return case_id, outcome


def write_outcome(path: str | Path, case_id: str, outcome: RefinementOutcome) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(outcome_to_dict(case_id, outcome), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_outcome(path: str | Path) -> tuple[str, RefinementOutcome]:
    return outcome_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def require_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value


def require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def require_float_tuple(value: object, length: int, name: str) -> tuple[float, ...]:
    values = require_list(value, name)
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    try:
        return tuple(float(item) for item in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numbers") from error
