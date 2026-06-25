"""
camera_sync.py  (ROS1)
======================
Color + Depth 이미지 시간 동기화 모듈.

ROS1 포팅 변경사항:
  - message_filters.Subscriber 인자 순서: (topic, type)  ← ROS1
    (ROS2는 (node, type, topic))
  - 구독 해제: sub.unregister()  ← ROS1
    (ROS2는 sub.sub.destroy())
  - rospy.loginfo() 사용
"""

from __future__ import annotations
from typing import Callable, Optional

import message_filters
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo
import rospy


SYNC_SLOP_SEC   = 0.05
SYNC_QUEUE_SIZE = 10


class CameraSync:
    """
    Color / Depth 동기화 래퍼 (ROS1).

    사용법:
        sync = CameraSync()
        sync.register_callback(my_callback)  # (color_img, depth_img, info) → None
        sync.on()   # 구독 시작
        sync.off()  # 구독 해제 (NAVIGATING 시 호출)
    """

    def __init__(self) -> None:
        self._bridge  = CvBridge()
        self._cb: Optional[Callable] = None

        self._sub_color: Optional[message_filters.Subscriber] = None
        self._sub_depth: Optional[message_filters.Subscriber] = None
        self._sub_info:  Optional[message_filters.Subscriber] = None
        self._sync:      Optional[message_filters.ApproximateTimeSynchronizer] = None
        self._active = False

    # ── 외부 인터페이스 ─────────────────────────────────────────────────── #

    def register_callback(self, callback: Callable) -> None:
        self._cb = callback

    def on(self) -> None:
        if self._active:
            return
        if self._cb is None:
            rospy.logwarn('CameraSync: callback not registered')
            return

        # ROS1: message_filters.Subscriber(topic, type)
        self._sub_color = message_filters.Subscriber(
            '/camera/color/image_raw/compressed', CompressedImage
        )
        self._sub_depth = message_filters.Subscriber(
            '/camera/aligned_depth_to_color/image_raw/compressedDepth', CompressedImage
        )
        self._sub_info = message_filters.Subscriber(
            '/camera/color/camera_info', CameraInfo
        )

        # slop : 동기화 허용 시간차 -> 타임스탬프에 맞는 프레임만 _internal_callback()으로 넘김
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_color, self._sub_depth, self._sub_info],
            queue_size=SYNC_QUEUE_SIZE,
            slop=SYNC_SLOP_SEC,
        )
        self._sync.registerCallback(self._internal_callback)
        self._active = True
        rospy.loginfo('CameraSync: ON')

    def off(self) -> None:
        if not self._active:
            return

        # ROS1: unregister()로 구독 해제
        for sub in (self._sub_color, self._sub_depth, self._sub_info):
            if sub is not None:
                sub.unregister()

        self._sub_color = None
        self._sub_depth = None
        self._sub_info  = None
        self._sync      = None
        self._active    = False
        rospy.loginfo('CameraSync: OFF')

    @property
    # 현재 구독 상태 확인
    def is_active(self) -> bool:
        return self._active

    # ── 내부 ────────────────────────────────────────────────────────────── #

    def _internal_callback(self, color_msg, depth_msg, info_msg):
        try:
            color_img = self._bridge.compressed_imgmsg_to_cv2(
                color_msg, desired_encoding='bgr8'
            )
        except Exception as e:
            rospy.logwarn(f'CameraSync: color decode error: {e}')
            return

        try:
            depth_img = self._bridge.compressed_imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough'
            ).astype(np.float32)
        except Exception as e:
            rospy.logwarn(f'CameraSync: depth decode error: {e}')
            return

        if self._cb is not None:
            self._cb(color_img, depth_img, info_msg)
