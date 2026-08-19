from __future__ import annotations

from trackrefinery.adapters.x4d import _track_documents


def _box(x: float) -> dict[str, object]:
    return {
        "translation": {"x": x, "y": 2.0, "z": 0.8},
        "size": {"length": 4.5, "width": 1.9, "height": 1.6},
        "rotation": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
    }


def test_x4d_track_export_orders_frames_and_preserves_lwh() -> None:
    samples = [
        {"sample_token": "frame-0"},
        {"sample_token": "frame-1"},
        {"sample_token": "frame-2"},
    ]
    rows = [
        {
            "sample_token": "frame-1",
            "instance_token": "vehicle-1",
            "category": "car",
            "bbox_3d": _box(2.0),
        },
        {
            "sample_token": "frame-0",
            "instance_token": "vehicle-1",
            "category": "car",
            "bbox_3d": _box(1.0),
        },
        {
            "sample_token": "frame-2",
            "instance_token": "single-frame",
            "category": "truck",
            "bbox_3d": _box(3.0),
        },
    ]

    [document] = _track_documents(clip_id="clip-1", rows=rows, samples=samples)

    assert document["case_id"] == "clip-1--vehicle-1"
    assert document["category"] == "car"
    observations = document["observations"]
    assert isinstance(observations, list)
    assert [row["frame_id"] for row in observations] == ["frame-0", "frame-1"]
    assert observations[0]["coarse_box"]["size_lwh"] == [4.5, 1.9, 1.6]
