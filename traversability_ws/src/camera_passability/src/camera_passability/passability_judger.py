from __future__ import annotations

import math
from typing import List, Tuple

from .config import PASS_THRESHOLD_M, VIRTUAL_WALL_STEP_M
from .tf_transformer import RobotPoint


class PassabilityJudger:
    """
    Step 4-c: 통과성 판단 + 가상 벽 점 생성.

    기존 ROS2 노드 로직을 그대로 유지:
    - 객체 0개: Pass=True
    - 객체 1개: Pass=True, wall_pts=[]
    - 객체 2개 이상: 거리-정렬 인접 쌍 중 최소 간격 D_min 사용
      - D_min > PASS_THRESHOLD_M: Pass=True
      - D_min <= PASS_THRESHOLD_M: Pass=False + critical pair 사이 벽 생성
    """

    def judge(self, robot_points: List[RobotPoint]) -> Tuple[bool, List[Tuple[float, float, float]]]:
        n = len(robot_points)

        if n == 0:
            return True, []
        if n == 1:
            return True, []

        # n>=2: 인접 쌍 중 최소 간격을 찾음
        min_dist = float("inf")
        critical_pair = None

        for i in range(n - 1):
            a = robot_points[i]
            b = robot_points[i + 1]
            d = math.hypot(a.x - b.x, a.y - b.y)
            if d < min_dist:
                min_dist = d
                critical_pair = (a, b)

        if min_dist > PASS_THRESHOLD_M:
            return True, []

        # Pass=False: critical pair 사이 가상 벽 포인트 생성
        wall_pts = self._make_virtual_wall(critical_pair[0], critical_pair[1])
        return False, wall_pts

    def _make_virtual_wall(
        self,
        pt_a: RobotPoint,
        pt_b: RobotPoint,
    ) -> List[Tuple[float, float, float]]:
        ax, ay = pt_a.x, pt_a.y
        bx, by = pt_b.x, pt_b.y

        dist = math.hypot(ax - bx, ay - by)
        n_pts = max(2, int(dist / VIRTUAL_WALL_STEP_M))

        wall: List[Tuple[float, float, float]] = []
        for i in range(n_pts + 1):
            t = i / n_pts
            wx = ax + t * (bx - ax)
            wy = ay + t * (by - ay)
            wall.append((wx, wy, 0.0))

        return wall

