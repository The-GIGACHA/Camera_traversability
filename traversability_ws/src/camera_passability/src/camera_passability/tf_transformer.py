from __future__ import annotations

from dataclasses import dataclass
from typing import List

import rospy
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 (PointStamped 변환 등록)
from geometry_msgs.msg import PointStamped

from .config import BASE_LINK_FRAME
from .depth_projector import CamPoint


@dataclass(frozen=True)
class RobotPoint:
    x: float
    y: float
    cls: int


class TfPointTransformer:
    """
    Step 4-a: camera_frame -> base_link 로 점 변환
    """

    def __init__(
        self,
        base_frame: str = BASE_LINK_FRAME,
        timeout_sec: float = 0.05,
        cache_time_sec: float = 10.0,
    ):
        self._base_frame = base_frame
        self._timeout = rospy.Duration(timeout_sec)

        self._buffer = tf2_ros.Buffer(cache_time=rospy.Duration(cache_time_sec))
        self._listener = tf2_ros.TransformListener(self._buffer)

    def transform_points(self, cam_points: List[CamPoint], header) -> List[RobotPoint]:
        robot_points: List[RobotPoint] = []

        for p in cam_points:
            pt_cam = PointStamped()
            pt_cam.header = header
            pt_cam.point.x = float(p.X)
            pt_cam.point.y = float(p.Y)
            pt_cam.point.z = float(p.Z)

            try:
                transform = self._buffer.lookup_transform(
                    self._base_frame,
                    pt_cam.header.frame_id,
                    pt_cam.header.stamp,
                    timeout=self._timeout,
                )
                pt_base = tf2_geometry_msgs.do_transform_point(pt_cam, transform)
            except (
                tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException,
            ) as e:
                rospy.logdebug(f"[tf_transformer] transform failed: {e}")
                continue

            robot_points.append(RobotPoint(x=float(pt_base.point.x), y=float(pt_base.point.y), cls=p.cls))

        return robot_points

