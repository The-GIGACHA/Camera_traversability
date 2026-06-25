#!/usr/bin/env python3
"""
vision_auth_node.py  (ROS1)
============================
비전 기반 수령인 인증 및 적재함 제어 — 메인 상태 머신 노드.

ROS1 포팅 변경사항:
  - rospy.init_node() / rospy.spin() 사용
  - Node 클래스 상속 없음 — 일반 Python 클래스
  - rospy.Subscriber / rospy.Publisher 사용
  - rospy.Timer(rospy.Duration(period), callback) 사용
  - rclpy.duration.Duration → rospy.Duration
  - get_logger() → rospy.loginfo/logwarn/logdebug
  - AuthState 전이 로직은 ROS2 버전과 동일
"""

import enum
import time
from typing import Optional

import numpy as np
import rospy
from std_msgs.msg import Bool, String
from sensor_msgs.msg import CameraInfo

from campus_delivery_auth.camera_sync import CameraSync
from campus_delivery_auth.qr_scanner import QRScanner
from campus_delivery_auth.gesture_recognizer import GestureRecognizer


# ── 파라미터 상수 ──────────────────────────────────────────────────────── #

QR_TIMEOUT_SEC       = 15.0
GESTURE_MAX_FAIL     = 3
AUTH_FAILED_WAIT_SEC = 30.0
UNLOCK_PUB_COUNT     = 3
UNLOCK_PUB_INTERVAL  = 1.0
TIMER_PERIOD_SEC     = 0.1


# ── 상태 정의 ──────────────────────────────────────────────────────────── #

class AuthState(enum.Enum):
    IDLE          = 'IDLE'
    QR_SCANNING   = 'QR_SCANNING'
    GESTURE_AUTH  = 'GESTURE_AUTH'
    AUTHENTICATED = 'AUTHENTICATED'
    AUTH_FAILED   = 'AUTH_FAILED'


# ── 노드 ───────────────────────────────────────────────────────────────── #

class VisionAuthNode:

    def __init__(self) -> None:
        rospy.init_node('vision_auth_node', anonymous=False)

        # ── 파라미터 ──────────────────────────────────────────────────── #
        # auth_bypass: 카메라/모델 없이 ARRIVED → 즉시 AUTHENTICATED (벤치 테스트용)
        self._auth_bypass = bool(rospy.get_param('~auth_bypass', False))
        if self._auth_bypass:
            rospy.logwarn('auth_bypass=True — 인증 단계를 건너뜁니다 (테스트 모드)')

        # ── 서브모듈 ──────────────────────────────────────────────────── #
        self._camera_sync = CameraSync()
        self._qr_scanner  = QRScanner()
        self._gesture_rec = GestureRecognizer()

        self._camera_sync.register_callback(self._on_camera_frame)

        # ── 구독 (ROS1: rospy.Subscriber(topic, type, callback)) ──────── #
        rospy.Subscriber('/robot_state', String, self._on_robot_state, queue_size=10)

        # ── 발행 ─────────────────────────────────────────────────────── #
        self._pub_unlock = rospy.Publisher('/cargo_unlock', Bool, queue_size=10)

        # ── 상태 머신 변수 ───────────────────────────────────────────── #
        self._state         = AuthState.IDLE
        self._state_enter_t = time.monotonic()
        self._unlock_sent   = 0
        self._last_unlock_t = 0.0

        # ── 메인 루프 타이머 (ROS1: rospy.Timer) ─────────────────────── #
        rospy.Timer(rospy.Duration(TIMER_PERIOD_SEC), self._tick)

        rospy.loginfo('vision_auth_node started. State: IDLE')

    # ═══════════════════════════════════════════════════════════════════ #
    #  외부 입력 핸들러
    # ═══════════════════════════════════════════════════════════════════ #

    def _on_robot_state(self, msg: String) -> None:
        cmd = msg.data.strip().upper()

        if cmd == 'NAVIGATING':
            if self._state != AuthState.IDLE:
                rospy.loginfo(
                    f'NAVIGATING received → force IDLE (was {self._state.value})'
                )
                self._transition(AuthState.IDLE)

        elif cmd == 'ARRIVED':
            if self._state == AuthState.IDLE:
                if self._auth_bypass:
                    rospy.logwarn('ARRIVED + auth_bypass → AUTHENTICATED')
                    self._transition(AuthState.AUTHENTICATED)
                else:
                    rospy.loginfo('ARRIVED → QR_SCANNING')
                    self._transition(AuthState.QR_SCANNING)
            else:
                rospy.logwarn(
                    f'ARRIVED ignored: current state is {self._state.value}'
                )

    def _on_camera_frame(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        info_msg:  CameraInfo,
    ) -> None:
        if self._state == AuthState.QR_SCANNING:
            self._handle_qr(color_img, depth_img, info_msg)
        elif self._state == AuthState.GESTURE_AUTH:
            self._handle_gesture(color_img, depth_img, info_msg)

    # ═══════════════════════════════════════════════════════════════════ #
    #  인증 처리
    # ═══════════════════════════════════════════════════════════════════ #

    def _handle_qr(self, color_img, depth_img, info_msg) -> None:
        result = self._qr_scanner.scan(color_img, depth_img, info_msg)
        if result.success:
            rospy.loginfo(f'QR auth success (depth={result.depth_m:.2f}m)')
            self._transition(AuthState.AUTHENTICATED)
        else:
            rospy.logdebug(f'QR scan: {result.reason}')

    def _handle_gesture(self, color_img, depth_img, info_msg) -> None:
        result = self._gesture_rec.recognize(color_img, depth_img, info_msg)
        if result.detected:
            rospy.loginfo(
                f'Gesture auth success: {result.gesture} '
                f'(conf={result.confidence:.2f}, depth={result.depth_m:.2f}m)'
            )
            self._transition(AuthState.AUTHENTICATED)
        else:
            rospy.logdebug(f'Gesture: {result.reason}')

    # ═══════════════════════════════════════════════════════════════════ #
    #  메인 루프 타이머
    #  ROS1 rospy.Timer 콜백은 TimerEvent 인자를 받으므로 _=None 처리
    # ═══════════════════════════════════════════════════════════════════ #

    def _tick(self, event=None) -> None:
        now     = time.monotonic()
        elapsed = now - self._state_enter_t

        if self._state == AuthState.QR_SCANNING:
            if elapsed >= QR_TIMEOUT_SEC:
                rospy.loginfo(f'QR timeout ({QR_TIMEOUT_SEC}s) → GESTURE_AUTH')
                self._transition(AuthState.GESTURE_AUTH)

        elif self._state == AuthState.GESTURE_AUTH:
            if elapsed >= QR_TIMEOUT_SEC * GESTURE_MAX_FAIL:
                rospy.logwarn('Gesture timeout → AUTH_FAILED')
                self._transition(AuthState.AUTH_FAILED)

        elif self._state == AuthState.AUTHENTICATED:
            if self._unlock_sent < UNLOCK_PUB_COUNT:
                if now - self._last_unlock_t >= UNLOCK_PUB_INTERVAL:
                    msg = Bool(); msg.data = True
                    self._pub_unlock.publish(msg)
                    self._last_unlock_t = now
                    self._unlock_sent  += 1
                    rospy.loginfo(
                        f'Unlock published ({self._unlock_sent}/{UNLOCK_PUB_COUNT})'
                    )
            else:
                rospy.loginfo('Unlock complete → IDLE')
                self._transition(AuthState.IDLE)

        elif self._state == AuthState.AUTH_FAILED:
            if elapsed >= AUTH_FAILED_WAIT_SEC:
                rospy.loginfo('AUTH_FAILED wait done → IDLE')
                self._transition(AuthState.IDLE)

    # ═══════════════════════════════════════════════════════════════════ #
    #  상태 전환
    # ═══════════════════════════════════════════════════════════════════ #

    def _transition(self, new_state: AuthState) -> None:
        old_state = self._state

        # 퇴장
        if old_state in (AuthState.QR_SCANNING, AuthState.GESTURE_AUTH,
                         AuthState.AUTHENTICATED):
            self._camera_sync.off()
            self._gesture_rec.reset()

        self._state         = new_state
        self._state_enter_t = time.monotonic()

        # 진입
        if new_state == AuthState.IDLE:
            self._unlock_sent = 0

        elif new_state == AuthState.QR_SCANNING:
            self._camera_sync.on()

        elif new_state == AuthState.GESTURE_AUTH:
            self._gesture_rec.reset()
            self._camera_sync.on()

        elif new_state == AuthState.AUTHENTICATED:
            self._unlock_sent   = 0
            self._last_unlock_t = 0.0
            self._camera_sync.off()

        elif new_state == AuthState.AUTH_FAILED:
            self._camera_sync.off()
            rospy.logwarn(
                f'Authentication failed. Waiting {AUTH_FAILED_WAIT_SEC}s.'
            )

        rospy.loginfo(f'State: {old_state.value} → {new_state.value}')

    # ─────────────────────────────────────────────────────────────────── #

    def spin(self) -> None:
        rospy.spin()


# ──────────────────────────────────────────────────────────────────────── #

if __name__ == '__main__':
    node = VisionAuthNode()
    node.spin()
