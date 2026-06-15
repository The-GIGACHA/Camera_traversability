#!/usr/bin/env python3
"""
measure_node.py  (ROS1)
========================
정량 지표 측정용 독립 노드. 기존 파이프라인 노드는 건드리지 않고
모든 출력 토픽을 구독해 통계만 수집한다.

수집 지표:
  1. E2E latency  : camera frame 도착 시각 → 같은 stamp 의 costmap 발행
                    시각 까지의 wall-clock 차이. 카메라부터 costmap 까지 전체
                    파이프라인이 소요하는 실시간 지연.
  2. FOV coverage : 매 costmap 프레임의 default(노랑, cost=50) 아닌 셀 비율.
                    SRE 가 실제 데이터로 채운 영역 비중 — 카메라가 보는 범위.
  3. Dynamic count: dynamic_obstacles/pointcloud 프레임당 점 개수.
  4. Path length  : local_path 프레임당 waypoint 개수.

사용:
  # 한 터미널
  roslaunch camera_passability rosbag_test.launch bag:=/path/to/your.bag

  # 다른 터미널
  rosrun camera_passability measure_node.py

  # rosbag 재생 끝나거나 Ctrl+C → 통계 요약 출력
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import CompressedImage, PointCloud2


# stamp(ns) → wall_arrival(ns) 의 매핑이 너무 커지지 않도록 cap
_CAM_BUF_SIZE = 600          # ~20s @ 30Hz, 충분히 넉넉
_INTERIM_PERIOD_SEC = 10.0   # interim 통계 출력 주기

# SRE_DEFAULT_COST=0.35 × 100 = 35. 정확히 35인 셀을 default 로 본다.
# (config.py 가 변경되면 여기도 함께 갱신할 것)
_DEFAULT_COST_INT = 35


class MeasureNode:
    def __init__(self) -> None:
        rospy.init_node("measure_node", anonymous=False)

        # camera frame 의 stamp → 도착 wall time (ns)
        self._cam_arrival: Dict[int, int] = {}
        self._cam_arrival_order: Deque[int] = deque(maxlen=_CAM_BUF_SIZE)

        # 누적 통계
        self._costmap_latencies_ms: List[float] = []
        self._coverage_ratios: List[float] = []
        self._dyn_counts: List[int] = []
        self._path_lengths: List[int] = []

        # 카메라 / costmap / dynamic / path 의 단순 카운트 — 발행 주기 검증용
        self._cam_msg_count = 0
        self._costmap_msg_count = 0
        self._dyn_msg_count = 0
        self._path_msg_count = 0
        self._start_wall = rospy.Time.now().to_sec()

        # ── 구독자 ────────────────────────────────────────────────── #
        rospy.Subscriber(
            "/camera/color/image_raw/compressed",
            CompressedImage, self._cam_cb, queue_size=30,
        )
        rospy.Subscriber(
            "/traversability/costmap",
            OccupancyGrid, self._costmap_cb, queue_size=5,
        )
        rospy.Subscriber(
            "/dynamic_obstacles/pointcloud",
            PointCloud2, self._dyn_cb, queue_size=5,
        )
        rospy.Subscriber(
            "/camera/local_path",
            Path, self._path_cb, queue_size=5,
        )

        rospy.on_shutdown(self._final_report)

        # interim report 타이머
        self._interim_timer = rospy.Timer(
            rospy.Duration(_INTERIM_PERIOD_SEC), self._interim_cb
        )

        rospy.loginfo(
            "[measure] running. Ctrl+C 종료 시 최종 통계 출력. "
            f"interim 통계 {_INTERIM_PERIOD_SEC:.0f}초마다 출력."
        )

    # ══════════════════════════════════════════════════════════════════ #
    #  콜백
    # ══════════════════════════════════════════════════════════════════ #

    def _cam_cb(self, msg: CompressedImage) -> None:
        stamp_ns = msg.header.stamp.to_nsec()
        wall_ns = rospy.Time.now().to_nsec()
        self._cam_arrival[stamp_ns] = wall_ns
        self._cam_arrival_order.append(stamp_ns)
        self._cam_msg_count += 1

        # 오래된 entry 제거 (deque maxlen 으로 빠지는 stamp 도 dict 에서 함께)
        if len(self._cam_arrival) > _CAM_BUF_SIZE * 2:
            keep = set(self._cam_arrival_order)
            self._cam_arrival = {
                k: v for k, v in self._cam_arrival.items() if k in keep
            }

    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        self._costmap_msg_count += 1

        # E2E latency — 같은 stamp 의 카메라 frame 도착 시각과 비교
        stamp_ns = msg.header.stamp.to_nsec()
        if stamp_ns in self._cam_arrival:
            wall_ns = rospy.Time.now().to_nsec()
            latency_ms = (wall_ns - self._cam_arrival[stamp_ns]) / 1e6
            self._costmap_latencies_ms.append(latency_ms)

        # FOV coverage — default cost 가 아닌 셀 비율
        grid = np.asarray(msg.data, dtype=np.int16)
        if grid.size > 0:
            non_default = int(np.sum(grid != _DEFAULT_COST_INT))
            self._coverage_ratios.append(non_default / grid.size)

    def _dyn_cb(self, msg: PointCloud2) -> None:
        self._dyn_msg_count += 1
        self._dyn_counts.append(int(msg.width))   # 점 개수

    def _path_cb(self, msg: Path) -> None:
        self._path_msg_count += 1
        self._path_lengths.append(len(msg.poses))

    # ══════════════════════════════════════════════════════════════════ #
    #  통계 출력
    # ══════════════════════════════════════════════════════════════════ #

    def _interim_cb(self, _event) -> None:
        elapsed = rospy.Time.now().to_sec() - self._start_wall
        if elapsed < 1.0:
            return
        rospy.loginfo(
            f"[measure] interim @ {elapsed:.0f}s — "
            f"cam={self._cam_msg_count} "
            f"costmap={self._costmap_msg_count} "
            f"({self._costmap_msg_count / elapsed:.1f}Hz) "
            f"dyn={self._dyn_msg_count} "
            f"path={self._path_msg_count} "
            f"matched_latency_n={len(self._costmap_latencies_ms)}"
        )

    def _final_report(self) -> None:
        elapsed = max(1e-3, rospy.Time.now().to_sec() - self._start_wall)

        def fmt_stats(arr, name: str, unit: str = "") -> str:
            if not arr:
                return f"  {name:<35} : no data"
            a = np.asarray(arr, dtype=np.float64)
            return (
                f"  {name:<35} : n={len(a):<5} "
                f"mean={a.mean():.2f}{unit}  "
                f"p50={np.median(a):.2f}{unit}  "
                f"p95={np.percentile(a, 95):.2f}{unit}  "
                f"p99={np.percentile(a, 99):.2f}{unit}  "
                f"max={a.max():.2f}{unit}"
            )

        rospy.loginfo("=" * 76)
        rospy.loginfo("정량 지표 — 최종 요약")
        rospy.loginfo("=" * 76)
        rospy.loginfo(f"  측정 wall-clock 시간 [s]              : {elapsed:.1f}")
        rospy.loginfo(
            f"  발행 메시지 수 (cam / costmap / dyn / path) : "
            f"{self._cam_msg_count} / {self._costmap_msg_count} / "
            f"{self._dyn_msg_count} / {self._path_msg_count}"
        )
        rospy.loginfo(
            f"  발행 주기 (Hz, 평균)                          : "
            f"cam={self._cam_msg_count / elapsed:.2f}  "
            f"costmap={self._costmap_msg_count / elapsed:.2f}  "
            f"dyn={self._dyn_msg_count / elapsed:.2f}  "
            f"path={self._path_msg_count / elapsed:.2f}"
        )
        rospy.loginfo("-" * 76)
        rospy.loginfo(fmt_stats(
            self._costmap_latencies_ms,
            "E2E latency (cam arrival -> costmap)", " ms"
        ))
        rospy.loginfo(fmt_stats(
            [r * 100 for r in self._coverage_ratios],
            "BEV FOV coverage", " %"
        ))
        rospy.loginfo(fmt_stats(
            self._dyn_counts,
            "Dynamic obstacle count / frame", ""
        ))
        rospy.loginfo(fmt_stats(
            self._path_lengths,
            "Local path waypoint count", ""
        ))
        rospy.loginfo("=" * 76)


if __name__ == "__main__":
    try:
        MeasureNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
