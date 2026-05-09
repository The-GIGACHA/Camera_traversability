#!/usr/bin/env python3
"""
visualizer_node.py  (ROS1)
===========================
카메라 영상 + traversability costmap + local path를 한 창에 시각화.

구독:
  /camera/color/image_raw/compressed   — 카메라 컬러 영상
  /traversability/costmap              — OccupancyGrid (0~100)
  /camera/local_path                   — nav_msgs/Path (base_link)
  /dynamic_obstacles/pointcloud        — 동적 장애물 위치 (base_link)
  /dynamic_obstacles/passable          — 통과 가능 여부

출력:
  OpenCV 창 "Camera Passability" — 좌: 카메라 / 우: costmap BEV
  (rosbag 재생 중 실시간으로 업데이트)

창 구성:
  ┌─────────────────────┬─────────────────────┐
  │                     │  BEV Costmap        │
  │  Camera Feed        │  (bird's eye view)  │
  │  + passable 표시    │  + local path (파랑)│
  │                     │  + 장애물 (빨강)    │
  └─────────────────────┴─────────────────────┘
"""

import threading

import cv2
import numpy as np
import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import CompressedImage, PointCloud2
from std_msgs.msg import Bool
    
from camera_passability.config import (
    SRE_CELL_SIZE_M,
    SRE_MAX_RANGE_M,
    SRE_HALF_WIDTH_M,
    TOPIC_TRAVERSABILITY_COSTMAP,
    TOPIC_LOCAL_PATH,
    TOPIC_DYNAMIC_OBSTACLES,
    TOPIC_PASSABLE,
    TOPIC_COLOR_COMPRESSED,
)

# ── 그리드 상수 ───────────────────────────────────────────────────────── #
GRID_H    = int(SRE_MAX_RANGE_M  / SRE_CELL_SIZE_M)   # 25
GRID_W    = int(2 * SRE_HALF_WIDTH_M / SRE_CELL_SIZE_M)  # 20
ORIGIN_GX = GRID_W // 2
ORIGIN_GY = GRID_H - 1

# BEV 이미지에서 셀 1개당 픽셀 수 (크게 할수록 선명)
CELL_PX = 20

BEV_H = GRID_H * CELL_PX
BEV_W = GRID_W * CELL_PX

# 카메라 영상 표시 높이 (BEV와 높이 맞춤)
CAM_DISPLAY_H = BEV_H
CAM_DISPLAY_W = int(BEV_H * 4 / 3)  # 4:3 비율 유지


def cost_to_color(cost: float) -> tuple:
    """cost 0~100 → BGR 색상. 초록(safe) → 노랑 → 빨강(danger)."""
    c = float(np.clip(cost, 0, 100)) / 100.0
    if c < 0.5:
        # 초록 → 노랑
        r = int(c * 2 * 255)
        g = 255
    else:
        # 노랑 → 빨강
        r = 255
        g = int((1.0 - (c - 0.5) * 2) * 255)
    return (0, g, r)  # BGR


class VisualizerNode:

    def __init__(self) -> None:
        rospy.init_node("visualizer_node", anonymous=False)

        self._bridge = CvBridge()
        self._lock   = threading.Lock()

        # 최신 데이터 캐시
        self._cam_img:    np.ndarray | None = None
        self._costmap:    np.ndarray | None = None   # (GRID_H, GRID_W) float32
        self._path_poses: list              = []      # List[PoseStamped]
        self._dyn_pts:    list              = []      # List[(x, y)]
        self._passable:   bool              = True

        # ── 구독자 ────────────────────────────────────────────────── #
        rospy.Subscriber(
            TOPIC_COLOR_COMPRESSED, CompressedImage,
            self._cam_cb, queue_size=2,
        )
        rospy.Subscriber(
            TOPIC_TRAVERSABILITY_COSTMAP, OccupancyGrid,
            self._costmap_cb, queue_size=1,
        )
        rospy.Subscriber(
            TOPIC_LOCAL_PATH, Path,
            self._path_cb, queue_size=1,
        )
        rospy.Subscriber(
            TOPIC_DYNAMIC_OBSTACLES, PointCloud2,
            self._cloud_cb, queue_size=1,
        )
        rospy.Subscriber(
            TOPIC_PASSABLE, Bool,
            self._passable_cb, queue_size=5,
        )

        rospy.loginfo("visualizer_node ready. Press 'q' in the window to quit.")

    # ══════════════════════════════════════════════════════════════════ #
    #  콜백
    # ══════════════════════════════════════════════════════════════════ #

    def _cam_cb(self, msg: CompressedImage) -> None:
        try:
            img = self._bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
        with self._lock:
            self._cam_img = img

    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        if msg.info.width != GRID_W or msg.info.height != GRID_H:
            return
        grid = np.array(msg.data, dtype=np.float32).reshape(GRID_H, GRID_W)
        with self._lock:
            self._costmap = grid

    def _path_cb(self, msg: Path) -> None:
        with self._lock:
            self._path_poses = list(msg.poses)

    def _cloud_cb(self, msg: PointCloud2) -> None:
        pts = [
            (float(p[0]), float(p[1]))
            for p in pc2.read_points(msg, field_names=("x", "y"), skip_nans=True)
        ]
        with self._lock:
            self._dyn_pts = pts

    def _passable_cb(self, msg: Bool) -> None:
        with self._lock:
            self._passable = bool(msg.data)

    # ══════════════════════════════════════════════════════════════════ #
    #  BEV costmap 렌더링
    # ══════════════════════════════════════════════════════════════════ #

    def _render_bev(
        self,
        costmap: np.ndarray,
        path_poses: list,
        dyn_pts: list,
    ) -> np.ndarray:
        bev = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

        # ── 셀 색상 ────────────────────────────────────────────────── #
        for gy in range(GRID_H):
            for gx in range(GRID_W):
                color = cost_to_color(costmap[gy, gx])
                y0, y1 = gy * CELL_PX, (gy + 1) * CELL_PX
                x0, x1 = gx * CELL_PX, (gx + 1) * CELL_PX
                bev[y0:y1, x0:x1] = color

        # ── 그리드 선 ──────────────────────────────────────────────── #
        for i in range(GRID_H + 1):
            cv2.line(bev, (0, i * CELL_PX), (BEV_W, i * CELL_PX), (60, 60, 60), 1)
        for j in range(GRID_W + 1):
            cv2.line(bev, (j * CELL_PX, 0), (j * CELL_PX, BEV_H), (60, 60, 60), 1)

        # ── 동적 장애물 (빨간 원) ──────────────────────────────────── #
        for xr, yr in dyn_pts:
            gy_f = ORIGIN_GY - xr / SRE_CELL_SIZE_M
            gx_f = ORIGIN_GX - yr / SRE_CELL_SIZE_M
            px = int(gx_f * CELL_PX + CELL_PX // 2)
            py = int(gy_f * CELL_PX + CELL_PX // 2)
            if 0 <= px < BEV_W and 0 <= py < BEV_H:
                cv2.circle(bev, (px, py), CELL_PX // 2, (0, 0, 255), -1)

        # ── local path (파란 선 + 점) ──────────────────────────────── #
        path_pixels = []
        for pose in path_poses:
            xr = pose.pose.position.x
            yr = pose.pose.position.y
            gy_f = ORIGIN_GY - xr / SRE_CELL_SIZE_M
            gx_f = ORIGIN_GX - yr / SRE_CELL_SIZE_M
            px = int(gx_f * CELL_PX + CELL_PX // 2)
            py = int(gy_f * CELL_PX + CELL_PX // 2)
            path_pixels.append((px, py))

        for i in range(1, len(path_pixels)):
            cv2.line(bev, path_pixels[i - 1], path_pixels[i], (255, 100, 0), 2)
        for px, py in path_pixels:
            if 0 <= px < BEV_W and 0 <= py < BEV_H:
                cv2.circle(bev, (px, py), 3, (255, 200, 0), -1)

        # ── 로봇 위치 (흰 삼각형) ──────────────────────────────────── #
        robot_px = ORIGIN_GX * CELL_PX + CELL_PX // 2
        robot_py = ORIGIN_GY * CELL_PX + CELL_PX // 2
        tri = np.array([
            [robot_px,          robot_py - CELL_PX],
            [robot_px - CELL_PX // 2, robot_py + CELL_PX // 2],
            [robot_px + CELL_PX // 2, robot_py + CELL_PX // 2],
        ])
        cv2.fillPoly(bev, [tri], (255, 255, 255))

        # ── 범례 ───────────────────────────────────────────────────── #
        cv2.putText(bev, "BEV Costmap", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(bev, f"{SRE_MAX_RANGE_M:.0f}m", (4, BEV_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(bev, "robot", (robot_px - 15, robot_py + CELL_PX + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # 거리 눈금 (1m 단위)
        for d_m in range(1, int(SRE_MAX_RANGE_M) + 1):
            row = ORIGIN_GY - int(d_m / SRE_CELL_SIZE_M)
            if 0 <= row < GRID_H:
                y_px = row * CELL_PX
                cv2.line(bev, (0, y_px), (BEV_W, y_px), (100, 100, 100), 1)
                cv2.putText(bev, f"{d_m}m", (2, y_px - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        return bev

    # ══════════════════════════════════════════════════════════════════ #
    #  메인 루프
    # ══════════════════════════════════════════════════════════════════ #

    def spin(self) -> None:
        rate = rospy.Rate(15)   # 15 Hz 렌더링

        while not rospy.is_shutdown():
            with self._lock:
                cam_img    = self._cam_img.copy()    if self._cam_img    is not None else None
                costmap    = self._costmap.copy()    if self._costmap    is not None else None
                path_poses = list(self._path_poses)
                dyn_pts    = list(self._dyn_pts)
                passable   = self._passable

            # ── 카메라 패널 ───────────────────────────────────────── #
            if cam_img is not None:
                cam_panel = cv2.resize(cam_img, (CAM_DISPLAY_W, CAM_DISPLAY_H))
            else:
                cam_panel = np.zeros((CAM_DISPLAY_H, CAM_DISPLAY_W, 3), dtype=np.uint8)
                cv2.putText(cam_panel, "Waiting for camera...",
                            (10, CAM_DISPLAY_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

            # 통과 가능 여부 표시
            label  = "PASS" if passable else "BLOCKED"
            color  = (0, 200, 0) if passable else (0, 0, 220)
            cv2.putText(cam_panel, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # ── BEV 패널 ──────────────────────────────────────────── #
            if costmap is not None:
                bev_panel = self._render_bev(costmap, path_poses, dyn_pts)
            else:
                bev_panel = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
                cv2.putText(bev_panel, "Waiting for costmap...",
                            (4, BEV_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # ── 합성 ──────────────────────────────────────────────── #
            # BEV를 cam_panel 높이에 맞춤
            bev_scaled = cv2.resize(bev_panel, (BEV_W, CAM_DISPLAY_H))
            combined   = np.hstack([cam_panel, bev_scaled])

            cv2.imshow("Camera Passability", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                rospy.signal_shutdown("user quit")
                break

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    VisualizerNode().spin()
