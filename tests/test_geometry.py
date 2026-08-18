from __future__ import annotations

import numpy as np

from trackrefinery import Box3D, Pose3D
from trackrefinery.evaluation import bev_iou, iou_3d
from trackrefinery.geometry import (
    compose_pose,
    inverse_pose,
    inverse_transform_points,
    points_in_box,
    transform_points,
)


def test_pose_inverse_and_composition_round_trip() -> None:
    pose = Pose3D(
        (3.0, -2.0, 0.5),
        (0.0, 0.0, np.sin(0.3), np.cos(0.3)),
    )
    points = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])

    transformed = transform_points(points, pose)

    assert np.allclose(inverse_transform_points(transformed, pose), points)
    identity = compose_pose(pose, inverse_pose(pose))
    assert np.allclose(identity.translation_xyz, 0.0, atol=1e-8)
    assert np.allclose(identity.orientation_xyzw, (0.0, 0.0, 0.0, 1.0))


def test_oriented_box_membership_and_iou() -> None:
    box = Box3D(
        center=(1.0, 2.0, 0.0),
        size_lwh=(4.0, 2.0, 2.0),
        orientation_xyzw=(0.0, 0.0, np.sin(0.25), np.cos(0.25)),
    )
    points = transform_points(np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), box.pose)

    assert points_in_box(points, box).tolist() == [True, False]
    assert np.isclose(bev_iou(box, box), 1.0)
    assert np.isclose(iou_3d(box, box), 1.0)
    disjoint = Box3D((20.0, 20.0, 0.0), box.size_lwh, box.orientation_xyzw)
    assert bev_iou(box, disjoint) == 0.0
    assert iou_3d(box, disjoint) == 0.0
