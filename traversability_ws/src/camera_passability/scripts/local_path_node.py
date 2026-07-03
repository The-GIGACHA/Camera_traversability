#!/usr/bin/env python3
"""
local_path_node.py  (ROS1)
===========================
traversability costmap → local path (nav_msgs/Path) 발행 노드.

구독:
  /traversability/costmap   (nav_msgs/OccupancyGrid)
  /dynamic_obstacles/passable (std_msgs/Bool)  — 경고 로그용

발행:
  /camera/local_path        (nav_msgs/Path, frame_id = base_link)

동작:
  costmap 수신마다 LocalPathPlanner.plan() 실행 후
  결과 Waypoint 배열을 PoseStamped 시퀀스로 패킹해 발행.
  동적 장애물로 통과 불가 판정 시 경고만 남기고 경로는 그대로 발행
  (회피 전략은 상위 경로 계획기가 결정).
"""

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool

from camera_passability.local_path_planner import LocalPathPlanner, GRID_H, GRID_W
from camera_passability.config import (
    BASE_LINK_FRAME,
    TOPIC_TRAVERSABILITY_COSTMAP,
    TOPIC_LOCAL_PATH,
    TOPIC_PASSABLE,
)


class LocalPathNode:

    def __init__(self) -> None:
        rospy.init_node("local_path_node", anonymous=False)

        self._planner = LocalPathPlanner()

        # 최근 동적 장애물 통과 가능 여부 (경고 로그용)
        self._dyn_passable: bool = True

        # ── 발행자 ────────────────────────────────────────────────── #
        self._pub = rospy.Publisher(
            TOPIC_LOCAL_PATH, Path, queue_size=1
        )

        # ── 구독자 ────────────────────────────────────────────────── #
        rospy.Subscriber(
            TOPIC_TRAVERSABILITY_COSTMAP,
            OccupancyGrid,
            self._costmap_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            TOPIC_PASSABLE,
            Bool,
            self._passable_cb,
            queue_size=5,
        )

        rospy.loginfo("local_path_node ready.")

    # ══════════════════════════════════════════════════════════════════ #
    #  콜백
    # ══════════════════════════════════════════════════════════════════ #

    def _passable_cb(self, msg: Bool) -> None:
        self._dyn_passable = bool(msg.data)

    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        # 그리드 크기 불일치 방어
        if msg.info.width != GRID_W or msg.info.height != GRID_H:
            rospy.logwarn_throttle(
                5.0,
                f"[local_path] costmap size mismatch "
                f"({msg.info.width}×{msg.info.height} expected {GRID_W}×{GRID_H})",
            )
            return

        # OccupancyGrid data(flat) → 2D ndarray
        grid = np.array(msg.data, dtype=np.float32).reshape(GRID_H, GRID_W)

        # ── 동적 장애물 경고 ─────────────────────────────────────── #
        if not self._dyn_passable:
            rospy.logwarn_throttle(
                1.0,
                "[local_path] dynamic obstacle BLOCKED — path may be unsafe",
            )

        # ── 경로 계산 ─────────────────────────────────────────────── #
        waypoints = self._planner.plan(grid)

        # ── nav_msgs/Path 패킹 ────────────────────────────────────── #
        path_msg = Path()
        path_msg.header.stamp    = msg.header.stamp
        path_msg.header.frame_id = BASE_LINK_FRAME

        for wp in waypoints:
            pose = PoseStamped()
            pose.header           = path_msg.header
            pose.pose.position.x  = wp.x
            pose.pose.position.y  = wp.y
            pose.pose.position.z  = 0.0
            pose.pose.orientation.w = 1.0   # 방향은 상위 레이어에서 보간
            path_msg.poses.append(pose)

        self._pub.publish(path_msg)

        rospy.logdebug(
            f"[local_path] published {len(waypoints)} waypoints, "
            f"end=({waypoints[-1].x:.2f}, {waypoints[-1].y:.2f}) m"
            if waypoints else "[local_path] empty waypoints"
        )

    def spin(self) -> None:
        rospy.spin()


if __name__ == "__main__":
    LocalPathNode().spin()
