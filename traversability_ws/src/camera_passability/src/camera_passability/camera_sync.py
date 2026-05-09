from __future__ import annotations

"""
camera_sync.py
==============
Step 1: Color + Depth CompressedImage를 2-way ApproximateTimeSynchronizer로
동기화하고, CameraInfo는 별도 캐시로 관리한 뒤
파이프라인 콜백으로 전달합니다.

3-way 동기화 → 2-way 변경 이유
  ApproximateTimeSynchronizer로 Color + Depth + CameraInfo 세 토픽을 동시에
  맞추려 하면 바쁜 노드 환경(YOLO/TF 리스너/다중 구독자)에서 내부 큐가
  꽉 차거나 타임스탬프 편차 판정 타이밍이 어긋나 sync가 영구히 발화되지
  않을 수 있습니다. CameraInfo는 카메라 파라미터가 거의 변하지 않으므로
  별도 rospy.Subscriber로 최신 메시지를 캐시하고, 핵심 영상 두 토픽만
  2-way 동기화하면 훨씬 안정적으로 동작합니다.
"""

from typing import Callable, Optional

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, CompressedImage

import numpy as np

from .config import (
    SYNC_SLOP_SEC,
    TOPIC_COLOR_COMPRESSED,
    TOPIC_DEPTH_COMPRESSED,
    TOPIC_CAMERA_INFO,
)


class CameraSynchronizer:
    """
    Color / Depth CompressedImage를 2-way ApproximateTimeSynchronizer로
    동기화하고, CameraInfo는 별도 캐시로 관리합니다.

    callback 시그니처:
        callback(color_img: np.ndarray,
                 depth_img: np.ndarray,
                 info_msg:  CameraInfo) -> None
    """

    def __init__(
        self,
        callback: Callable[[np.ndarray, np.ndarray, CameraInfo], None],
        queue_size: int = 10,
        slop_sec: float = SYNC_SLOP_SEC,
        color_topic: str = TOPIC_COLOR_COMPRESSED,
        depth_topic: str = TOPIC_DEPTH_COMPRESSED,
        info_topic: str = TOPIC_CAMERA_INFO,
    ):
        self._callback = callback
        self._bridge = CvBridge()

        # CameraInfo 캐시 (파라미터가 거의 변하지 않으므로 최신값 유지)
        self._latest_info: Optional[CameraInfo] = None

        # self.에 저장 — 지역변수로 두면 __init__ 종료 후 GC가 수거해
        # message_filters 내부 콜백 체인이 끊길 수 있음
        self._sub_color = message_filters.Subscriber(color_topic, CompressedImage)
        self._sub_depth = message_filters.Subscriber(depth_topic, CompressedImage)

        # CameraInfo는 별도 rospy.Subscriber로 캐시 (2-way 동기화 안정성 향상)
        self._sub_info = rospy.Subscriber(
            info_topic, CameraInfo, self._info_cb, queue_size=5
        )

        # 2-way ApproximateTimeSynchronizer (Color + Depth)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_color, self._sub_depth],
            queue_size=queue_size,
            slop=slop_sec,
        )
        self._sync.registerCallback(self._synced_cb)

        rospy.loginfo(
            f"[camera_sync] Initialized — color='{color_topic}' "
            f"depth='{depth_topic}' info='{info_topic}' "
            f"slop={slop_sec}s queue={queue_size}"
        )

    # ── CameraInfo 캐시 ───────────────────────────────────────────────── #

    def _info_cb(self, msg: CameraInfo) -> None:
        self._latest_info = msg

    # ── 이미지 디코딩 ─────────────────────────────────────────────────── #

    def _decode_images(
        self,
        color_msg: CompressedImage,
        depth_msg: CompressedImage,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            color_img = self._bridge.compressed_imgmsg_to_cv2(
                color_msg,
                desired_encoding="bgr8",
            )
        except Exception as e:
            rospy.logwarn(f"[camera_sync] color decode failed: {e}")
            return None, None

        try:
            depth_img = self._bridge.compressed_imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )
        except Exception as e:
            rospy.logwarn(f"[camera_sync] depth decode failed: {e}")
            return None, None

        # depth는 보통 (H, W) 또는 (H, W, 1). (H, W)로 정규화
        if depth_img is not None and depth_img.ndim == 3 and depth_img.shape[-1] == 1:
            depth_img = depth_img[:, :, 0]

        return color_img, depth_img

    # ── 2-way 동기화 콜백 ─────────────────────────────────────────────── #

    def _synced_cb(
        self,
        color_msg: CompressedImage,
        depth_msg: CompressedImage,
    ) -> None:
        # 진단: sync가 실제로 발화되는지 확인
        rospy.loginfo_throttle(10.0, "[camera_sync] SYNC FIRED — decoding images...")

        if self._latest_info is None:
            rospy.logwarn_throttle(
                5.0,
                "[camera_sync] CameraInfo not yet received — skipping frame. "
                "Check that the camera_info topic is publishing.",
            )
            return

        color_img, depth_img = self._decode_images(color_msg, depth_msg)
        if color_img is None or depth_img is None:
            return

        self._callback(color_img, depth_img, self._latest_info)
