from __future__ import annotations

import math
from typing import List, Tuple

from .config import PASS_THRESHOLD_M, ROBOT_WIDTH_M, VIRTUAL_WALL_STEP_M
from .tf_transformer import RobotPoint


class PassabilityJudger:
    """
    Step 4-c: 통과성 판단 + 가상 벽 점 생성 (corridor-based).

    로봇 forward corridor (x>0 AND |y| <= PASS_THRESHOLD_M/2) 안에 들어온
    장애물만 판단에 사용. corridor 바깥(멀리 좌·우측 사람, 가로지른 차 등)은
    로봇 경로와 무관하므로 무시한다.

    이전 구현은 거리축 정렬 인접 쌍의 Euclidean 최소 간격만 봐서, 좌측
    멀리 모여있는 사람 무리에도 BLOCKED를 띄우는 false positive가 잦았다.

    판단:
      0) corridor 안 장애물 0개  →  PASS
      1) corridor 안 lateral 인접 gap (좌측 가장자리 → 장애물들 → 우측 가장자리)
         중 max(gap) >= ROBOT_WIDTH_M  →  PASS (한 측면으로 통과 가능)
      2) 그 외  →  BLOCKED + critical 위치에 가상 벽
    """

    def judge(self, robot_points: List[RobotPoint]) -> Tuple[bool, List[Tuple[float, float, float]]]:
        half_corridor = PASS_THRESHOLD_M / 2.0

        in_corridor = [
            p for p in robot_points
            if p.x > 0.0 and abs(p.y) <= half_corridor
        ]

        n = len(in_corridor)
        if n == 0:
            return True, []

        sorted_in = sorted(in_corridor, key=lambda p: p.y)

        # corridor 양 끝을 포함한 모든 lateral gap
        edges_y = [-half_corridor] + [p.y for p in sorted_in] + [half_corridor]
        gaps = [edges_y[i + 1] - edges_y[i] for i in range(len(edges_y) - 1)]

        if max(gaps) >= ROBOT_WIDTH_M:
            return True, []

        # BLOCKED — 가상 벽 생성
        if n >= 2:
            # 인접 쌍 중 lateral gap 최소를 critical 로 채택
            min_gap = float("inf")
            critical = (sorted_in[0], sorted_in[1])
            for i in range(n - 1):
                gap = sorted_in[i + 1].y - sorted_in[i].y
                if gap < min_gap:
                    min_gap = gap
                    critical = (sorted_in[i], sorted_in[i + 1])
            wall_pts = self._make_virtual_wall(critical[0], critical[1])
        else:
            # 단일 장애물 → 가까운 corridor 가장자리까지 벽으로 연결
            p = sorted_in[0]
            edge_y = half_corridor if p.y >= 0.0 else -half_corridor
            edge_pt = RobotPoint(x=p.x, y=edge_y, cls=p.cls)
            wall_pts = self._make_virtual_wall(p, edge_pt)

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

