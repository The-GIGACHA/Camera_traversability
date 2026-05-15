from __future__ import annotations

"""
sre_mapper.py
=============
Step 3 — Geometric Perception (SRE 기반 지면 거칠기 분석)

depth_projector.py의 진화 버전.

핵심 설계 (너의 아이디어 반영):
  1. YOLO bbox 마스킹된 depth를 입력으로 받음
     → 사람이 있던 셀은 NaN이므로 자동으로 bincount에서 제외됨
     → costmap_publisher에서 해당 셀에 COST_LETHAL 강제 할당 (별도 처리)

  2. 역투영: pixel(u,v) + depth(Z) → camera 3D (X,Y,Z)
             → base_link TF → (xr, yr, zr)
     X, Y는 그리드 셀 인덱스 계산에만 사용하고 버림.
     Z(높이)만 각 셀 배열에 누적 → 분산 계산.

  3. np.bincount 로 Python 루프 없이 고속 셀 통계 계산
     (sum_z, sum_z² → mean, variance in one pass)

  4. 세 지표 융합:
     - variance : 거칠기 (잔디, 수풀, 돌부리)
     - delta    : 단차   (연석, 계단)
     - abs      : 절대높이 조건부 (벽, 큰 장애물) — 오르막 오탐 방지 포함

  5. D455 내장 IMU pitch/roll 임계값 초과 시 프레임 스킵
     (흔들림이 너무 심한 프레임은 Z값 전체가 틀어져 cost 왜곡됨)

출력: 0.0~1.0 float32 ndarray (GRID_H × GRID_W)
      costmap_publisher가 0~100 OccupancyGrid로 변환해 발행
"""

from typing import Optional, Tuple

import numpy as np
import rospy
import tf

from sensor_msgs.msg import CameraInfo

from .config import (
    BASE_LINK_FRAME,
    SRE_CELL_SIZE_M, SRE_MAX_RANGE_M, SRE_HALF_WIDTH_M,
    SRE_DEPTH_MIN_M, SRE_DEPTH_MAX_M, SRE_SUBSAMPLE,
    SRE_MIN_POINTS, SRE_DEFAULT_COST,
    SRE_VAR_SAT, SRE_DELTA_SAT,
    SRE_ABS_H_THRESH, SRE_SLOPE_FLAT,
    SRE_W_VAR, SRE_W_DELTA, SRE_W_ABS,
)

# ── 그리드 크기 (config 값에서 계산) ─────────────────────────────────── #

GRID_H    = int(SRE_MAX_RANGE_M  / SRE_CELL_SIZE_M)   # 전후 셀 수
GRID_W    = int(2 * SRE_HALF_WIDTH_M / SRE_CELL_SIZE_M)  # 좌우 셀 수
ORIGIN_GX = GRID_W // 2   # base_link가 그리드 좌우 중앙
ORIGIN_GY = GRID_H - 1    # base_link가 그리드 맨 아래 (전방 = 위)


class SREMapper:
    """
    마스킹된 depth + camera_info → geometric cost grid (float32, 0~1).

    사용 방법:
        mapper = SREMapper()
        cost_grid = mapper.compute(depth_masked, info_msg, header)
        # cost_grid: np.ndarray shape (GRID_H, GRID_W), 0.0~1.0
    """

    def __init__(self) -> None:
        self._tf_listener = tf.TransformListener()

    # ══════════════════════════════════════════════════════════════════ #
    #  Public API
    # ══════════════════════════════════════════════════════════════════ #

    def compute(
        self,
        depth_masked: np.ndarray,   # float32, mm 단위, NaN = 마스킹된 픽셀
        info_msg: CameraInfo,
        header,                     # 원본 depth 메시지 header (TF stamp)
    ) -> np.ndarray:
        """
        마스킹된 depth → geometric cost grid.

        반환: float32 ndarray (GRID_H × GRID_W), 값 범위 0.0~1.0
              포인트 부족 셀은 SRE_DEFAULT_COST.
        """
        # ── 1. mm → m 변환, 유효 픽셀 마스크 ───────────────────────── #
        depth_m = depth_masked / 1000.0  # NaN은 그대로 NaN

        valid = (
            np.isfinite(depth_m) &
            (depth_m >= SRE_DEPTH_MIN_M) &
            (depth_m <= SRE_DEPTH_MAX_M)
        )

        # ── 2. 서브샘플 픽셀 좌표 생성 ──────────────────────────────── #
        h, w = depth_m.shape
        us, vs = np.meshgrid(
            np.arange(0, w, SRE_SUBSAMPLE),
            np.arange(0, h, SRE_SUBSAMPLE),
        )
        us = us.flatten()
        vs = vs.flatten()

        valid_sub = valid[vs, us]
        us = us[valid_sub]
        vs = vs[valid_sub]

        if len(us) == 0:
            return np.full((GRID_H, GRID_W), SRE_DEFAULT_COST, dtype=np.float32)

        # ── 3. 역투영: pixel → camera 3D ────────────────────────────── #
        # X, Y: 그리드 셀 찾는 용도로만 사용 후 버림
        # Z: 각 셀에 모아서 분산 계산
        fx = float(info_msg.K[0])
        fy = float(info_msg.K[4])
        cx = float(info_msg.K[2])
        cy = float(info_msg.K[5])

        Z_cam = depth_m[vs, us]
        X_cam = (us - cx) * Z_cam / fx
        Y_cam = (vs - cy) * Z_cam / fy

        pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1).astype(np.float32)

        # ── 4. camera → base_link TF (배치 변환) ────────────────────── #
        pts_base = self._transform_to_base_link(pts_cam, header)
        if pts_base is None:
            return np.full((GRID_H, GRID_W), SRE_DEFAULT_COST, dtype=np.float32)

        xr = pts_base[:, 0]   # 전방 (+X)
        yr = pts_base[:, 1]   # 좌측 (+Y)
        zr = pts_base[:, 2]   # 높이 — 이 값만 분산 계산에 사용

        # ── 5. 그리드 인덱싱 ─────────────────────────────────────────── #
        gy = (ORIGIN_GY - (xr / SRE_CELL_SIZE_M)).astype(int)
        gx = (ORIGIN_GX - (yr / SRE_CELL_SIZE_M)).astype(int)

        in_range = (gx >= 0) & (gx < GRID_W) & (gy >= 0) & (gy < GRID_H)
        gx = gx[in_range]
        gy = gy[in_range]
        zr = zr[in_range]

        if len(gx) == 0:
            return np.full((GRID_H, GRID_W), SRE_DEFAULT_COST, dtype=np.float32)

        # ── 6. 셀별 Z 통계 (np.bincount 고속 누적) ───────────────────── #
        mean_z, var_z, count = self._accumulate_cells(gx, gy, zr)

        # ── 7. 세 지표 계산 ───────────────────────────────────────────── #
        norm_var    = self._variance_score(var_z)
        norm_delta  = self._delta_score(mean_z)
        abs_penalty = self._abs_score(mean_z, norm_delta, count)

        # ── 8. 융합 ──────────────────────────────────────────────────── #
        reliable = count >= SRE_MIN_POINTS
        raw  = SRE_W_VAR * norm_var + SRE_W_DELTA * norm_delta + SRE_W_ABS * abs_penalty
        cost = np.where(reliable, raw, SRE_DEFAULT_COST)

        return np.clip(cost, 0.0, 1.0).astype(np.float32)

    # ══════════════════════════════════════════════════════════════════ #
    #  셀 통계 누적 (루프 없음)
    # ══════════════════════════════════════════════════════════════════ #

    @staticmethod
    def _accumulate_cells(
        gx: np.ndarray,
        gy: np.ndarray,
        zr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        np.bincount으로 셀별 Z 평균·분산·포인트 수를 한 번에 계산.

        Var(Z) = E[Z²] - E[Z]²  (수치 안정성 위해 양수 클리핑)
        """
        flat = gy * GRID_W + gx
        n    = GRID_H * GRID_W

        count  = np.bincount(flat, minlength=n).reshape(GRID_H, GRID_W)
        sum_z  = np.bincount(flat, weights=zr,    minlength=n).reshape(GRID_H, GRID_W)
        sum_z2 = np.bincount(flat, weights=zr**2, minlength=n).reshape(GRID_H, GRID_W)

        safe_count = np.maximum(count, 1)
        mean_z = np.where(count > 0, sum_z / safe_count, np.nan)
        var_z  = np.where(
            count > 0,
            np.maximum(sum_z2 / safe_count - mean_z**2, 0.0),
            0.0,
        )
        return mean_z, var_z, count

    # ══════════════════════════════════════════════════════════════════ #
    #  지표 1: Z 분산 — 거칠기 (잔디·수풀·돌부리)
    # ══════════════════════════════════════════════════════════════════ #

    @staticmethod
    def _variance_score(var_z: np.ndarray) -> np.ndarray:
        return np.clip(var_z / SRE_VAR_SAT, 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════ #
    #  지표 2: 인접 셀 높이 차이 — 단차 (연석·계단)
    # ══════════════════════════════════════════════════════════════════ #

    @staticmethod
    def _delta_score(mean_z: np.ndarray) -> np.ndarray:
        """
        인접 셀 간 높이 차이(절댓값) 계산.

        NaN(데이터 없음) 셀과 valid 셀의 경계에서는 가짜 delta가 생기지
        않도록 양쪽 모두 valid 한 경우에만 diff를 사용한다.

        과거 구현은 NaN을 0으로 채워서 |0 − ground_noise| 만큼의 fake delta가
        FOV 경계 모든 셀에 발생, SRE_DELTA_SAT(=0.08m)을 쉽게 넘어
        saturate → "유령 벽" halo가 visualizer에 보였음.
        """
        valid  = ~np.isnan(mean_z)
        filled = np.where(valid, mean_z, 0.0)

        # 아래 이웃 (axis=0): 양쪽 valid 인 경우에만 diff 채택.
        # np.diff(..., append=arr[-1:]) → 마지막 행은 자기 자신과 비교(=0).
        dz_dy = np.abs(np.diff(filled, axis=0, append=filled[-1:, :]))
        pair_valid_y = np.zeros_like(mean_z, dtype=bool)
        pair_valid_y[:-1] = valid[:-1] & valid[1:]
        dz_dy = np.where(pair_valid_y, dz_dy, 0.0)

        # 오른쪽 이웃 (axis=1): 마지막 열도 자기 자신과 비교(=0).
        dz_dx = np.abs(np.diff(filled, axis=1, append=filled[:, -1:]))
        pair_valid_x = np.zeros_like(mean_z, dtype=bool)
        pair_valid_x[:, :-1] = valid[:, :-1] & valid[:, 1:]
        dz_dx = np.where(pair_valid_x, dz_dx, 0.0)

        delta = np.maximum(dz_dy, dz_dx)
        return np.clip(delta / SRE_DELTA_SAT, 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════ #
    #  지표 3: 조건부 절대 높이 — 벽·큰 장애물 (오르막 오탐 방지)
    # ══════════════════════════════════════════════════════════════════ #

    @staticmethod
    def _abs_score(
        mean_z: np.ndarray,
        norm_delta: np.ndarray,
        count: np.ndarray,
    ) -> np.ndarray:
        """
        평지(slope 낮음)인데 절대 높이가 높은 셀에만 페널티.

        오르막 오탐 방지 로직:
          slope = norm_delta * SRE_DELTA_SAT / SRE_CELL_SIZE_M  (m/m 근사)
          slope < SRE_SLOPE_FLAT  →  평지 판정
          평지 + 높이 초과  →  페널티 적용
          오르막(slope ↑)   →  페널티 면제
        """
        slope_approx = norm_delta * SRE_DELTA_SAT / SRE_CELL_SIZE_M

        is_flat  = slope_approx < SRE_SLOPE_FLAT
        is_high  = mean_z > SRE_ABS_H_THRESH
        has_data = count >= SRE_MIN_POINTS
        not_nan  = ~np.isnan(mean_z)

        excess  = np.maximum(mean_z - SRE_ABS_H_THRESH, 0.0)
        penalty = np.clip(excess / (SRE_ABS_H_THRESH * 3), 0.0, 1.0)

        apply = is_flat & is_high & has_data & not_nan
        return np.where(apply, penalty, 0.0)

    # ══════════════════════════════════════════════════════════════════ #
    #  TF 배치 변환 (카메라 → base_link)
    # ══════════════════════════════════════════════════════════════════ #

    def _transform_to_base_link(
        self,
        pts_cam: np.ndarray,    # (N, 3) float32
        header,
    ) -> Optional[np.ndarray]:
        """
        TF를 한 번만 조회해 행렬 곱으로 일괄 변환 (포인트당 조회 x).
        실패 시 None 반환 → 호출부에서 빈 grid 반환.
        """
        try:
            self._tf_listener.waitForTransform(
                BASE_LINK_FRAME,
                header.frame_id,
                header.stamp,
                rospy.Duration(0.05),
            )
            (trans, rot) = self._tf_listener.lookupTransform(
                BASE_LINK_FRAME,
                header.frame_id,
                header.stamp,
            )
        except Exception as e:
            rospy.logdebug(f"[sre_mapper] TF lookup failed: {e}")
            return None

        import tf.transformations as tft
        T = tft.quaternion_matrix(rot)
        T[:3, 3] = trans

        ones = np.ones((len(pts_cam), 1), dtype=np.float32)
        hom  = np.hstack([pts_cam, ones])          # (N, 4)
        out  = (T @ hom.T).T                       # (N, 4)
        return out[:, :3].astype(np.float32)
