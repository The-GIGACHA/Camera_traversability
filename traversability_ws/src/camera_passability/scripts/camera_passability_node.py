#!/usr/bin/env python3
"""
camera_passability_node.py  (ROS1)
====================================
동적 장애물 파이프라인 메인 노드.

Step 1: CameraSynchronizer   — Color + Depth 시간 동기화
Step 2: YoloDetector         — 동적 객체 탐지 + depth 마스킹
Step 3: DepthProjector       — bbox → 카메라 3D 좌표 (CamPoint)
Step 4-a: TfPointTransformer — camera_frame → base_link 변환
Step 4-b: filter_fov         — 전방 부채꼴 + 최대 거리 필터
Step 4-c: PassabilityJudger  — 통과성 판단 + 가상 벽 생성
Step 4-d: 발행               — /dynamic_obstacles/passable  (std_msgs/Bool)
                               /dynamic_obstacles/pointcloud (sensor_msgs/PointCloud2)

traversability costmap 생성은 Traversability_traversability_node.py 가 담당.
이 노드는 동적 장애물(person/bicycle) 통과성 판단만 담당.
"""

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
from std_msgs.msg import Bool, Header

from camera_passability.camera_sync import CameraSynchronizer
from camera_passability.yolo_detector import YoloDetector
from camera_passability.depth_projector import DepthProjector
from camera_passability.tf_transformer import TfPointTransformer
from camera_passability.fov_filter import filter_fov
from camera_passability.passability_judger import PassabilityJudger
from camera_passability.config import (
    BASE_LINK_FRAME,
    TOPIC_DYNAMIC_OBSTACLES,
    TOPIC_PASSABLE,
)

_PC2_FIELDS = [
    PointField("x", 0,  PointField.FLOAT32, 1),
    PointField("y", 4,  PointField.FLOAT32, 1),
    PointField("z", 8,  PointField.FLOAT32, 1),
]


class CameraPassabilityNode:

    def __init__(self) -> None:
        rospy.init_node("camera_passability_node", anonymous=False)

        # ── 파이프라인 구성 요소 ──────────────────────────────────────── #
        self._yolo      = YoloDetector()
        self._projector = DepthProjector()
        self._tf        = TfPointTransformer()
        self._judger    = PassabilityJudger()

        # ── 발행자 ────────────────────────────────────────────────────── #
        self._pub_passable = rospy.Publisher(
            TOPIC_PASSABLE, Bool, queue_size=1
        )
        self._pub_cloud = rospy.Publisher(
            TOPIC_DYNAMIC_OBSTACLES, PointCloud2, queue_size=1
        )

        # ── Step 1: 카메라 동기화 (마지막에 초기화 — 콜백 바로 시작) ──── #
        self._sync = CameraSynchronizer(callback=self._pipeline_cb)

        rospy.loginfo("camera_passability_node ready.")

    # ══════════════════════════════════════════════════════════════════ #
    #  메인 파이프라인 콜백
    # ══════════════════════════════════════════════════════════════════ #

    def _pipeline_cb(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        info_msg: CameraInfo,
    ) -> None:
        header = info_msg.header

        # ── Step 2: YOLO 탐지 + depth 마스킹 ────────────────────────── #
        detections, _ = self._yolo.detect_and_mask_depth(color_img, depth_img)

        if not detections:
            self._pub_passable.publish(Bool(data=True))
            self._publish_cloud([], header)
            return

        # ── Step 3: bbox → 카메라 3D ──────────────────────────────── #
        cam_points = self._projector.project(detections, depth_img, info_msg)

        if not cam_points:
            self._pub_passable.publish(Bool(data=True))
            self._publish_cloud([], header)
            return

        # ── Step 4-a: camera_frame → base_link ───────────────────── #
        robot_points = self._tf.transform_points(cam_points, header)

        # ── Step 4-b: FOV 필터 ───────────────────────────────────── #
        robot_points = filter_fov(robot_points)

        if not robot_points:
            self._pub_passable.publish(Bool(data=True))
            self._publish_cloud([], header)
            return

        # ── Step 4-c: 통과성 판단 ─────────────────────────────────── #
        passable, wall_pts = self._judger.judge(robot_points)

        if not passable:
            rospy.logwarn_throttle(
                1.0,
                f"[camera_passability] BLOCKED — {len(robot_points)} obstacles, "
                f"virtual wall {len(wall_pts)} pts",
            )

        # ── Step 4-d: 발행 ────────────────────────────────────────── #
        self._pub_passable.publish(Bool(data=passable))

        obstacle_pts = [(p.x, p.y, 0.0) for p in robot_points]
        self._publish_cloud(obstacle_pts + wall_pts, header)

    # ══════════════════════════════════════════════════════════════════ #
    #  PointCloud2 발행 헬퍼
    # ══════════════════════════════════════════════════════════════════ #

    def _publish_cloud(self, pts, header) -> None:
        hdr = Header()
        hdr.stamp    = header.stamp
        hdr.frame_id = BASE_LINK_FRAME
        cloud = pc2.create_cloud(hdr, _PC2_FIELDS, pts)
        self._pub_cloud.publish(cloud)

    def spin(self) -> None:
        rospy.spin()


if __name__ == "__main__":
    CameraPassabilityNode().spin()
