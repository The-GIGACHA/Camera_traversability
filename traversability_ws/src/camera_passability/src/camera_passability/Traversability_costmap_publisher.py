from __future__ import annotations

"""
costmap_publisher.py
====================
Step 4 — Score Fusion & Costmap Generation

obstacle_publisher.py의 진화 버전.
동적 장애물 PointCloud2 발행은 obstacle_publisher.py가 계속 담당.
이 노드는 traversability OccupancyGrid 생성·발행만 담당.

융합 규칙 (우선순위 순):
  1. YOLO bbox 셀 (사람/자전거)       → COST_LETHAL (100), 무조건 덮어쓰기
  2. seg free_path 분류 셀            → sre_cost × SEG_FREE_DISCOUNT (완화)
  3. seg obstacle 분류 셀             → max(sre_cost, SEG_OBSTACLE_COST)
  4. seg 미분류(unknown)              → sre_cost 그대로
  5. 전체에 대해 max(semantic, geo×0.3) 보수적 병합

라이다 팀 연동 (stub):
  /traversability/lidar_layer 토픽을 구독 준비만 해두고
  실제 로직은 라이다 팀이 채울 수 있도록 _lidar_cb stub 제공.
"""

from typing import List, Optional, Tuple

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

from .config import (
    BASE_LINK_FRAME,
    SRE_CELL_SIZE_M, SRE_MAX_RANGE_M, SRE_HALF_WIDTH_M,
    COST_LETHAL, SEG_FREE_DISCOUNT, SEG_OBSTACLE_COST,
    TOPIC_TRAVERSABILITY_COSTMAP,
)

GRID_H    = int(SRE_MAX_RANGE_M  / SRE_CELL_SIZE_M)
GRID_W    = int(2 * SRE_HALF_WIDTH_M / SRE_CELL_SIZE_M)
ORIGIN_GX = GRID_W // 2
ORIGIN_GY = GRID_H - 1


class CostmapPublisher:
    """
    sre_cost (float32 grid) + seg_grid (int8 grid) + dynamic_pts
    → nav_msgs/OccupancyGrid 발행.

    seg_grid 값 규약 (yolo_detector / seg 모델 출력):
      0   = free_path
      50  = unknown
      100 = obstacle
    """

    SEG_FREE     = 0
    SEG_UNKNOWN  = 50
    SEG_OBSTACLE = 100

    def __init__(self, queue_size: int = 1) -> None:
        self._pub = rospy.Publisher(
            TOPIC_TRAVERSABILITY_COSTMAP,
            OccupancyGrid,
            queue_size=queue_size,
        )

        # 라이다 팀 연동 stub — 라이다 팀이 이 토픽에 발행하면 자동으로 받아짐
        self._lidar_grid: Optional[np.ndarray] = None
        rospy.Subscriber(
            "/traversability/lidar_layer",
            OccupancyGrid,
            self._lidar_cb,
            queue_size=1,
        )

        # 동적 객체 포인트클라우드 (obstacle_publisher.py가 발행)
        self._dynamic_cells: List[Tuple[int, int]] = []
        rospy.Subscriber(
            "/dynamic_obstacles/pointcloud",
            PointCloud2,
            self._dynamic_cb,
            queue_size=1,
        )

    # ══════════════════════════════════════════════════════════════════ #
    #  Public API
    # ══════════════════════════════════════════════════════════════════ #

    def publish(
        self,
        header,
        sre_cost: np.ndarray,           # (GRID_H, GRID_W) float32, 0~1
        seg_grid: Optional[np.ndarray] = None,  # (GRID_H, GRID_W) int8
    ) -> None:
        """
        융합 후 OccupancyGrid 발행.

        seg_grid가 None이면 SRE 단독 모드로 동작
        (seg 모델 출력이 아직 준비 안 된 경우).
        """
        final = self._fuse(sre_cost, seg_grid)
        self._publish_grid(final, header)

    # ══════════════════════════════════════════════════════════════════ #
    #  융합 로직
    # ══════════════════════════════════════════════════════════════════ #

    def _fuse(
        self,
        sre_cost: np.ndarray,
        seg_grid: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        우선순위 규칙대로 최종 cost 산출.
        반환: int8 ndarray (GRID_H × GRID_W), 값 범위 0~100
        """
        # sre_cost를 0~100 정수로 변환
        geo = np.clip(sre_cost * 100, 0, 100).astype(np.float32)
        final = geo.copy()

        # ── seg 융합 ─────────────────────────────────────────────────── #
        if seg_grid is not None:
            is_free = seg_grid == self.SEG_FREE
            is_obs  = seg_grid == self.SEG_OBSTACLE

            # free_path: geometric cost 감쇠
            final[is_free] = geo[is_free] * SEG_FREE_DISCOUNT

            # obstacle: 높은 값으로 올림
            final[is_obs] = np.maximum(geo[is_obs], SEG_OBSTACLE_COST)

            # 보수적 병합 (seg 외 영역도 geometric이 높으면 유지)
            final = np.maximum(final, np.where(is_obs, 0, geo * 0.3))

        # ── 라이다 레이어 오버레이 (stub) ─────────────────────────────── #
        # 라이다 팀이 /traversability/lidar_layer를 발행하면 자동 반영
        if self._lidar_grid is not None:
            final = np.maximum(final, self._lidar_grid.astype(np.float32))

        # ── 동적 객체: LETHAL 강제 덮어쓰기 + 1셀 인플레이션 ──────────── #
        for (gy, gx) in self._dynamic_cells:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < GRID_H and 0 <= nx < GRID_W:
                        if dy == 0 and dx == 0:
                            final[ny, nx] = COST_LETHAL
                        else:
                            final[ny, nx] = max(
                                float(final[ny, nx]),
                                COST_LETHAL * 0.7,
                            )

        return np.clip(final, 0, COST_LETHAL).astype(np.int8)

    # ══════════════════════════════════════════════════════════════════ #
    #  발행
    # ══════════════════════════════════════════════════════════════════ #

    def _publish_grid(self, cost: np.ndarray, header) -> None:
        msg = OccupancyGrid()
        msg.header.stamp    = header.stamp
        msg.header.frame_id = BASE_LINK_FRAME

        msg.info.resolution = SRE_CELL_SIZE_M
        msg.info.width      = GRID_W
        msg.info.height     = GRID_H
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = -SRE_HALF_WIDTH_M
        msg.info.origin.position.z = 0.0

        msg.data = cost.flatten().tolist()
        self._pub.publish(msg)

    # ══════════════════════════════════════════════════════════════════ #
    #  구독 콜백
    # ══════════════════════════════════════════════════════════════════ #

    def _dynamic_cb(self, cloud_msg: PointCloud2) -> None:
        """동적 객체 위치 → 그리드 인덱스 캐시."""
        cells = []
        for p in pc2.read_points(
            cloud_msg, field_names=("x", "y"), skip_nans=True
        ):
            xr, yr = float(p[0]), float(p[1])
            gy = int(ORIGIN_GY - xr / SRE_CELL_SIZE_M)
            gx = int(ORIGIN_GX - yr / SRE_CELL_SIZE_M)
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                cells.append((gy, gx))
        self._dynamic_cells = cells

    def _lidar_cb(self, grid_msg: OccupancyGrid) -> None:
        """
        라이다 팀 연동 stub.
        /traversability/lidar_layer (OccupancyGrid) 수신 시 캐시.
        라이다 팀이 이 토픽에 맞는 그리드를 발행하면 자동으로 오버레이됨.
        """
        if (grid_msg.info.width  != GRID_W or
                grid_msg.info.height != GRID_H):
            rospy.logwarn_throttle(
                5.0,
                f"[costmap_publisher] lidar_layer grid size mismatch "
                f"({grid_msg.info.width}x{grid_msg.info.height} "
                f"vs {GRID_W}x{GRID_H})",
            )
            return
        self._lidar_grid = np.array(
            grid_msg.data, dtype=np.int8
        ).reshape(GRID_H, GRID_W)
