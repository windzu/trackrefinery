from __future__ import annotations

import numpy as np
import pytest

from trackrefinery import (
    Box3D,
    FrameCloud,
    InstanceTrack,
    InsufficientEvidence,
    Pose3D,
    RefinedFramePose,
    RefinementCase,
    RefinementSuccess,
    TrackObservation,
    TrackRefiner,
)


def _box() -> Box3D:
    return Box3D((1.0, 2.0, 0.0), (4.0, 2.0, 1.5), (0.0, 0.0, 0.0, 1.0))


def _case() -> RefinementCase:
    frames = tuple(
        FrameCloud(
            frame_id=str(index),
            timestamp_ns=10 + index,
            annotation_frame_id="base",
            world_from_annotation=Pose3D.identity(),
            points_xyz=np.zeros((2, 3), dtype=np.float32),
        )
        for index in range(2)
    )
    track = InstanceTrack(
        track_id="object-1",
        sequence_id="scene",
        category="car",
        observations=tuple(
            TrackObservation(frame_id=frame.frame_id, coarse_box=_box())
            for frame in frames
        ),
    )
    return RefinementCase("case", frames, track)


class PerfectStub(TrackRefiner):
    def _refine(self, case: RefinementCase) -> RefinementSuccess:
        return RefinementSuccess(
            track_id=case.track.track_id,
            canonical_size_lwh=(4.0, 2.0, 1.5),
            frame_poses=tuple(
                RefinedFramePose(item.frame_id, item.coarse_box.pose)
                for item in case.track.observations
            ),
            diagnostics={"iterations": 2, "stable": True},
        )


def test_subclass_is_importable_and_checked_by_public_base_class() -> None:
    result = PerfectStub().refine(_case())

    assert result.status == "success"
    assert result.canonical_size_lwh == (4.0, 2.0, 1.5)
    assert result.diagnostics["iterations"] == 2


def test_success_must_cover_every_input_frame() -> None:
    class MissingFrame(TrackRefiner):
        def _refine(self, case: RefinementCase) -> RefinementSuccess:
            return RefinementSuccess(
                track_id=case.track.track_id,
                canonical_size_lwh=(4.0, 2.0, 1.5),
                frame_poses=(
                    RefinedFramePose("0", case.track.observations[0].coarse_box.pose),
                ),
            )

    with pytest.raises(ValueError, match="every input frame"):
        MissingFrame().refine(_case())


def test_insufficient_evidence_is_a_first_class_outcome() -> None:
    outcome = InsufficientEvidence(
        track_id="object-1",
        reasons=("one_sided_visibility",),
        diagnostics={"visible_sides": ["front"]},
    )

    assert outcome.status == "insufficient_evidence"
    assert outcome.reasons == ("one_sided_visibility",)


def test_frame_rejects_nan_and_preserves_exact_point_time() -> None:
    with pytest.raises(ValueError, match="finite"):
        FrameCloud(
            frame_id="0",
            timestamp_ns=10,
            annotation_frame_id="base",
            world_from_annotation=Pose3D.identity(),
            points_xyz=np.asarray([[1.0, np.nan, 3.0]], dtype=np.float32),
        )

    point_times = np.asarray([9, 11], dtype=np.uint64)
    frame = FrameCloud(
        frame_id="0",
        timestamp_ns=10,
        annotation_frame_id="base",
        world_from_annotation=Pose3D.identity(),
        points_xyz=np.zeros((2, 3), dtype=np.float32),
        point_timestamps_ns=point_times,
    )
    assert frame.point_timestamps_ns.dtype == np.uint64
    assert not frame.points_xyz.flags.writeable


def test_case_is_exactly_one_track_and_matching_frames() -> None:
    case = _case()
    reversed_track = InstanceTrack(
        track_id=case.track.track_id,
        sequence_id=case.track.sequence_id,
        category=case.track.category,
        observations=tuple(reversed(case.track.observations)),
    )
    with pytest.raises(ValueError, match="exactly match"):
        RefinementCase(case.case_id, case.frames, reversed_track)
