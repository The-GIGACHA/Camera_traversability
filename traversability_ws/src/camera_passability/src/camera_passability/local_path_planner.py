from __future__ import annotations

"""
local_path_planner.py
=====================
OccupancyGrid traversability costmap → local path waypoints.

알고리즘: Forward-DP (동적 계획법)
  - 로봇 위치(그리드 하단 중앙)에서 전방으로 LOOKAHEAD_ROWS 행까지 탐색
  - 매 행마다 이전 행의 인접 3개 셀(좌/직진/우)에서 전환 가능
  - 비용 = 셀 cost + 좌우 이동당 LATERAL_COST 페널티 (직진 선호)
  - 목표 행에서 최소 누적 비용 셀을 역추적 → 경로 복원
  - 이동 평균으로 경로 스무딩 후 base_link 좌표로 변환

그리드 ↔ base_link 좌표 변환:
  xr (전방, m) = (ORIGIN_GY - gy) × CELL_SIZE
  yr (좌측, m) = (ORIGIN_GX - gx) × CELL_SIZE
"""

from typing import List, NamedTuple, Tuple

import numpy as np

from .config import (
    SRE_CELL_SIZE_M,
    SRE_MAX_RANGE_M,
    SRE_HALF_WIDTH_M,
    LOCAL_PATH_LOOKAHEAD_M,
    LOCAL_PATH_LATERAL_COST,
    LOCAL_PATH_SMOOTH_WINDOW,
)

# ── 그리드 상수 (sre_mapper / costmap_publisher 와 동일) ──────────────── #

GRID_H    = int(SRE_MAX_RANGE_M  / SRE_CELL_SIZE_M)
GRID_W    = int(2 * SRE_HALF_WIDTH_M / SRE_CELL_SIZE_M)
ORIGIN_GX = GRID_W // 2
ORIGIN_GY = GRID_H - 1   # 로봇 위치: 그리드 맨 아래

LOOKAHEAD_ROWS = min(
    int(LOCAL_PATH_LOOKAHEAD_M / SRE_CELL_SIZE_M),
    GRID_H - 1,
)


class Waypoint(NamedTuple):
    x: float   # base_link 전방 (m), +X
    y: float   # base_link 좌측 (m), +Y


class LocalPathPlanner:
    """
    OccupancyGrid (int8, 0~100) → Waypoint 리스트 (base_link 좌표).

    사용:
        planner = LocalPathPlanner()
        waypoints = planner.plan(cost_grid)   # cost_grid: np.ndarray (GRID_H, GRID_W)
    """

    def plan(self, cost_grid: np.ndarray) -> List[Waypoint]:
        """
        cost_grid: (GRID_H, GRID_W) int8 또는 float32, 값 범위 0~100.
        반환: Waypoint 리스트 (로봇 위치 → 전방 순서).
        """
        grid = cost_grid.astype(np.float32)

        # ── 1. Forward DP ─────────────────────────────────────────── #
        INF  = 1e9
        dp   = np.full((GRID_H, GRID_W), INF, dtype=np.float32)
        prev = np.full((GRID_H, GRID_W, 2), -1, dtype=np.int32)  # (gy, gx)

        # 시작: 로봇 위치
        dp[ORIGIN_GY, ORIGIN_GX] = float(grid[ORIGIN_GY, ORIGIN_GX])

        target_gy = max(ORIGIN_GY - LOOKAHEAD_ROWS, 0)

        for gy in range(ORIGIN_GY - 1, target_gy - 1, -1):
            for gx in range(GRID_W):
                cell_cost = float(grid[gy, gx])
                best      = INF
                best_from = (-1, -1)

                prev_gy = gy + 1
                if prev_gy > ORIGIN_GY:
                    continue

                for dgx in (-1, 0, 1):
                    pgx = gx + dgx
                    if pgx < 0 or pgx >= GRID_W:
                        continue
                    if dp[prev_gy, pgx] >= INF:
                        continue
                    lateral  = LOCAL_PATH_LATERAL_COST * abs(dgx)
                    candidate = dp[prev_gy, pgx] + cell_cost + lateral
                    if candidate < best:
                        best      = candidate
                        best_from = (prev_gy, pgx)

                if best < INF:
                    dp[gy, gx]   = best
                    prev[gy, gx] = best_from

        # ── 2. 목표 행에서 최소 cost 셀 선택 ─────────────────────── #
        target_row = dp[target_gy]
        if np.all(target_row >= INF):
            return self._straight_fallback()

        best_gx = int(np.argmin(target_row))
        if target_row[best_gx] >= INF:
            return self._straight_fallback()

        # ── 3. 역추적으로 경로 복원 ──────────────────────────────── #
        path_cells: List[Tuple[int, int]] = []
        gy, gx = target_gy, best_gx

        while 0 <= gy <= ORIGIN_GY:
            path_cells.append((gy, gx))
            pg = prev[gy, gx]
            if int(pg[0]) == -1:
                break
            gy, gx = int(pg[0]), int(pg[1])

        path_cells.reverse()  # 로봇 → 전방 순서

        # ── 4. 그리드 인덱스 → base_link 좌표 ───────────────────── #
        waypoints = [
            Waypoint(
                x=float((ORIGIN_GY - r) * SRE_CELL_SIZE_M),
                y=float((ORIGIN_GX - c) * SRE_CELL_SIZE_M),
            )
            for r, c in path_cells
        ]

        # ── 5. 이동 평균 스무딩 ──────────────────────────────────── #
        return self._smooth(waypoints)

    # ══════════════════════════════════════════════════════════════════ #
    #  내부 유틸
    # ══════════════════════════════════════════════════════════════════ #

    @staticmethod
    def _smooth(waypoints: List[Waypoint]) -> List[Waypoint]:
        """이동 평균 스무딩. 시작점(로봇 위치)은 원본 고정."""
        win = LOCAL_PATH_SMOOTH_WINDOW
        if len(waypoints) < win:
            return waypoints

        xs = np.array([w.x for w in waypoints])
        ys = np.array([w.y for w in waypoints])

        kernel = np.ones(win) / win
        xs_s   = np.convolve(xs, kernel, mode="same")
        ys_s   = np.convolve(ys, kernel, mode="same")

        xs_s[0] = xs[0]   # 로봇 위치 고정
        ys_s[0] = ys[0]

        return [Waypoint(x=float(x), y=float(y)) for x, y in zip(xs_s, ys_s)]

    @staticmethod
    def _straight_fallback() -> List[Waypoint]:
        """유효한 경로를 찾지 못했을 때 직진 경로로 대체."""
        return [
            Waypoint(x=float(i) * SRE_CELL_SIZE_M, y=0.0)
            for i in range(LOOKAHEAD_ROWS + 1)
        ]
