#!/usr/bin/env python3
"""
traversability_node.py  (ROS1)
===============================
Part 1 하이브리드 통과성 판단 파이프라인 메인 노드.

Step 1: CameraSynchronizer — Color + Depth 시간 동기화
Step 2: YoloDetector       — 동적 객체 탐지 + depth 마스킹
Step 3: SREMapper          — 마스킹된 depth → geometric cost grid
Step 4: CostmapPublisher   — seg + sre 융합 → OccupancyGrid 발행

동적 장애물 파이프라인 (사람/자전거 통과성 판단) 은
camera_passability_node.py 가 별도로 담당.
이 노드는 traversability costmap만 생성.

D455 내장 IMU 흔들림 보정:
  /camera/imu 구독 → pitch/roll 임계값 초과 프레임 스킵.
  임계값은 config.py IMU_PITCH_MAX_DEG / IMU_ROLL_MAX_DEG.
"""

import math

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage, Imu

from camera_passability.camera_sync import CameraSynchronizer
from camera_passability.yolo_detector import YoloDetector
from camera_passability.Traversability_sre_mapper import SREMapper
from camera_passability.Traversability_costmap_publisher import CostmapPublisher
from camera_passability.config import (
    IMU_TOPIC,
    IMU_PITCH_MAX_DEG,
    IMU_ROLL_MAX_DEG,
    TOPIC_COLOR_COMPRESSED,
)


class TraversabilityNode:

    def __init__(self) -> None:
        rospy.init_node("traversability_node", anonymous=False)

        # ── 파이프라인 구성 요소 ──────────────────────────────────────── #
        # ~yolo_model 파라미터로 .pt / .onnx / .engine 등 swap 가능.
        # 미지정 시 YoloDetector 기본값(config.YOLO_MODEL_PATH).
        yolo_model = rospy.get_param("~yolo_model", "")
        self._yolo      = YoloDetector(model_path=yolo_model) if yolo_model else YoloDetector()
        self._sre       = SREMapper()
        self._costmap   = CostmapPublisher()

        # ── D455 내장 IMU 흔들림 보정 ─────────────────────────────────── #
        self._imu_ok     = True   # False이면 해당 프레임 스킵
        self._pitch_max  = math.radians(IMU_PITCH_MAX_DEG)
        self._roll_max   = math.radians(IMU_ROLL_MAX_DEG)
        rospy.Subscriber(IMU_TOPIC, Imu, self._imu_cb, queue_size=20)

        # ── 진단: raw 토픽 수신 확인 (카메라 토픽이 이 노드에 도달하는지) #
        rospy.Subscriber(
            TOPIC_COLOR_COMPRESSED,
            CompressedImage,
            self._diag_color_cb,
            queue_size=2,
        )

        # ── Step 1: 카메라 동기화 (마지막에 초기화 — 콜백 바로 시작) ──── #
        self._sync = CameraSynchronizer(callback=self._pipeline_cb)

        rospy.loginfo("traversability_node ready.")

    # ══════════════════════════════════════════════════════════════════ #
    #  진단: raw 토픽 수신 확인
    # ══════════════════════════════════════════════════════════════════ #

    def _diag_color_cb(self, msg: CompressedImage) -> None:
        """raw 컬러 토픽이 이 노드에 도달하는지 5초마다 로그로 확인."""
        rospy.loginfo_throttle(
            5.0,
            f"[traversability] RAW color msg received — "
            f"stamp={msg.header.stamp.to_sec():.3f}",
        )

    # ══════════════════════════════════════════════════════════════════ #
    #  IMU 흔들림 보정
    # ══════════════════════════════════════════════════════════════════ #

    def _imu_cb(self, msg: Imu) -> None:
        """
        D455 내장 IMU quaternion → pitch/roll 변환 후 임계값 비교.
        임계값 초과 시 _imu_ok=False → 파이프라인 콜백에서 프레임 스킵.

        quaternion → euler 변환 (tf.transformations 대신 수식 직접 사용,
        의존성 최소화).
        """
        q = msg.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        # roll (x-axis)
        sinr = 2.0 * (qw * qx + qy * qz)
        cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr, cosr)

        # pitch (y-axis)
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        self._imu_ok = (
            abs(pitch) <= self._pitch_max and
            abs(roll)  <= self._roll_max
        )

        if not self._imu_ok:
            rospy.logwarn_throttle(
                1.0,
                f"[traversability] IMU shake detected "
                f"pitch={math.degrees(pitch):.1f}° "
                f"roll={math.degrees(roll):.1f}° — frame skipped",
            )

    # ══════════════════════════════════════════════════════════════════ #
    #  메인 파이프라인 콜백
    # ══════════════════════════════════════════════════════════════════ #

    def _pipeline_cb(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        info_msg: CameraInfo,
    ) -> None:
        
        rospy.loginfo_throttle(3.0, "[traversability] Pipeline callback fired — processing frame")
        # ── IMU 흔들림 체크 ──────────────────────────────────────────── #
        if not self._imu_ok:
            return   # Z값 전체가 틀어진 프레임 — 스킵

        try:
            # ── Step 2: YOLO 탐지 + depth 마스킹 ────────────────────────── #
            # detect_and_mask_depth: 추론 1회로 탐지 + NaN 마스킹 동시 처리
            detections, depth_masked = self._yolo.detect_and_mask_depth(
                color_img, depth_img
            )
            # depth_masked: 사람/자전거 bbox → NaN
            # detections: 동적 장애물 파이프라인과 공유 가능
            # (이 노드에서는 costmap_publisher의 dynamic_cb가 별도로 포인트클라우드 수신)

            # ── Step 3: SRE geometric cost 계산 ─────────────────────────── #
            sre_cost = self._sre.compute(
                depth_masked, info_msg, info_msg.header
            )
            # sre_cost: float32 (GRID_H × GRID_W), 0.0~1.0
            # NaN 마스킹된 셀(사람 있던 곳)은 포인트 없음 → SRE_DEFAULT_COST
            # costmap_publisher._dynamic_cb에서 LETHAL로 덮어씀

            # ── Step 4: 융합 + OccupancyGrid 발행 ───────────────────────── #
            # seg_grid=None: seg 모델 출력이 아직 없는 경우 SRE 단독 동작
            # seg 모델 준비되면 seg_grid를 여기에 넘기면 됨
            self._costmap.publish(
                header=info_msg.header,
                sre_cost=sre_cost,
                seg_grid=None,   # TODO: YOLOv8-seg 팀원 출력 연결 시 교체
            )
            rospy.loginfo_throttle(
                3.0, "[traversability] SRE costmap published"
            )

        except Exception as e:
            rospy.logerr(f"[traversability] Pipeline error: {e}")
            # 실패 시 빈 costmap 발행 (모든 셀 SRE_DEFAULT_COST)
            import traceback
            rospy.logerr(traceback.format_exc())

    def spin(self) -> None:
        rospy.spin()


if __name__ == "__main__":
    TraversabilityNode().spin()
