"""Small dependency-free rigid-geometry helpers used by tools and adapters."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from trackrefinery.contracts import Box3D, Pose3D


def quaternion_matrix(
    orientation_xyzw: tuple[float, float, float, float],
) -> NDArray[np.float64]:
    x, y, z, w = orientation_xyzw
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points(
    points_xyz: NDArray[np.floating], pose: Pose3D
) -> NDArray[np.float64]:
    """Apply a local-to-parent pose to row-vector points."""

    points = np.asarray(points_xyz, dtype=np.float64)
    return points @ quaternion_matrix(pose.orientation_xyzw).T + np.asarray(
        pose.translation_xyz
    )


def inverse_transform_points(
    points_xyz: NDArray[np.floating], pose: Pose3D
) -> NDArray[np.float64]:
    """Transform parent-frame row-vector points into pose-local coordinates."""

    points = np.asarray(points_xyz, dtype=np.float64)
    return (points - np.asarray(pose.translation_xyz)) @ quaternion_matrix(
        pose.orientation_xyzw
    )


def compose_pose(parent_from_middle: Pose3D, middle_from_local: Pose3D) -> Pose3D:
    parent_rotation = quaternion_matrix(parent_from_middle.orientation_xyzw)
    translation = parent_rotation @ np.asarray(middle_from_local.translation_xyz)
    translation += np.asarray(parent_from_middle.translation_xyz)
    orientation = quaternion_multiply(
        parent_from_middle.orientation_xyzw,
        middle_from_local.orientation_xyzw,
    )
    return Pose3D(tuple(translation), orientation)


def inverse_pose(parent_from_local: Pose3D) -> Pose3D:
    x, y, z, w = parent_from_local.orientation_xyzw
    inverse_orientation = (-x, -y, -z, w)
    inverse_rotation = quaternion_matrix(inverse_orientation)
    translation = -(inverse_rotation @ np.asarray(parent_from_local.translation_xyz))
    return Pose3D(tuple(translation), inverse_orientation)


def quaternion_multiply(
    parent_from_middle: tuple[float, float, float, float],
    middle_from_local: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = parent_from_middle
    x2, y2, z2, w2 = middle_from_local
    value = np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    value /= np.linalg.norm(value)
    return tuple(value)


def yaw_from_quaternion(orientation_xyzw: tuple[float, float, float, float]) -> float:
    x, y, z, w = orientation_xyzw
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def angle_difference(first: float, second: float) -> float:
    return float(np.arctan2(np.sin(first - second), np.cos(first - second)))


def rotation_geodesic(
    first_xyzw: tuple[float, float, float, float],
    second_xyzw: tuple[float, float, float, float],
) -> float:
    dot = abs(float(np.dot(first_xyzw, second_xyzw)))
    return float(2 * np.arccos(np.clip(dot, -1.0, 1.0)))


def points_in_box(points_xyz: NDArray[np.floating], box: Box3D) -> NDArray[np.bool_]:
    local = inverse_transform_points(points_xyz, box.pose)
    half_size = np.asarray(box.size_lwh, dtype=np.float64) / 2
    return np.all(np.abs(local) <= half_size + 1e-7, axis=1)


def box_corners(box: Box3D) -> NDArray[np.float64]:
    length, width, height = np.asarray(box.size_lwh, dtype=np.float64) / 2
    local = np.asarray(
        [
            [-length, -width, -height],
            [length, -width, -height],
            [length, width, -height],
            [-length, width, -height],
            [-length, -width, height],
            [length, -width, height],
            [length, width, height],
            [-length, width, height],
        ]
    )
    return transform_points(local, box.pose)


BOX_EDGE_INDICES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
