"""
qr_scanner.py  (ROS1)
=====================
OpenCV WeChat QR 기반 1차 인증 모듈.

WeChat QR 디코딩 + Depth 거리 필터링 + 코드 해시 검증을 하나의 scan() 호출로 처리. 
인증 결과를 QRResult 데이터클래스로 반환해서 호출부가 if result.success만 보면 되도록 설계

ROS1 포팅 사항:
  - rospy.logwarn/logdebug 사용 (Node.get_logger() 제거)
  - CameraInfo.K (1D list, 9 elements) — ROS1/ROS2 동일하나 명시
  - 나머지 로직은 ROS-독립적이므로 변경 없음
"""

from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo
import rospy


DEPTH_VALID_MAX_M = 0.5
DEPTH_VALID_MIN_M = 0.1
QR_ROI_DEPTH_PAD  = 20

_SECRET_CODE_HASH = os.environ.get(
    'AUTH_QR_CODE_HASH',
    hashlib.sha256(b'campus_delivery_secret').hexdigest(),
)


@dataclass
class QRResult:
    success: bool
    data:    str   = ''
    depth_m: float = 0.0
    reason:  str   = ''


class QRScanner:

    def __init__(self) -> None:
        self._detector = self._init_wechat_qr()

    def scan(
        self,
        color_img:   np.ndarray,
        depth_img:   np.ndarray,
        camera_info: CameraInfo,
    ) -> QRResult:
        decoded_list, points_list = self._detector.detectAndDecode(color_img)

        # 디코딩 결과가 없으면 인증 실패
        if not decoded_list:
            return QRResult(success=False, reason='no_qr_detected')

        # 디코딩 결과가 있으면 Depth 거리 필터링 및 코드 해시 검증
        for decoded, points in zip(decoded_list, points_list):
            if not decoded:
                continue

            # Depth 거리 필터링
            depth_m = self._measure_depth(depth_img, points)
            if depth_m is None:
                continue
            if not (DEPTH_VALID_MIN_M <= depth_m <= DEPTH_VALID_MAX_M):
                return QRResult(
                    success=False, data=decoded, depth_m=depth_m,
                    reason=f'depth_out_of_range:{depth_m:.2f}m',
                )

            # 코드 해시 검증
            if self._verify_code(decoded):
                return QRResult(success=True, data=decoded, depth_m=depth_m)
            else:
                # 코드 해시 검증 실패
                return QRResult(
                    success=False, data='[REDACTED]', depth_m=depth_m,
                    reason='invalid_code',
                )

        # 디코딩 결과가 없으면 인증 실패
        return QRResult(success=False, reason='decode_empty')

    @staticmethod
    # WeChat QR 모델 초기화
    def _init_wechat_qr():
        model_dir = os.path.join(os.path.dirname(__file__), 'models', 'wechat_qr')
        paths = [
            os.path.join(model_dir, 'detect.prototxt'),
            os.path.join(model_dir, 'detect.caffemodel'),
            os.path.join(model_dir, 'sr.prototxt'),
            os.path.join(model_dir, 'sr.caffemodel'),
        ]
        if all(os.path.exists(p) for p in paths):
            return cv2.wechat_qrcode_WeChatQRCode(*paths)
        rospy.logwarn('WeChatQR: model files not found, using fallback mode')
        return cv2.wechat_qrcode_WeChatQRCode()

    @staticmethod
    # Depth 거리 측정
    def _measure_depth(depth_img, points) -> Optional[float]:
        if points is None or len(points) == 0:
            return None

        # ROI 영역 설정
        pts = points.reshape(-1, 2).astype(int)
        x1 = max(0, pts[:, 0].min() - QR_ROI_DEPTH_PAD)
        y1 = max(0, pts[:, 1].min() - QR_ROI_DEPTH_PAD)
        x2 = min(depth_img.shape[1], pts[:, 0].max() + QR_ROI_DEPTH_PAD)
        y2 = min(depth_img.shape[0], pts[:, 1].max() + QR_ROI_DEPTH_PAD)
        roi   = depth_img[y1:y2, x1:x2]

        # 유효 depth 값들 추출
        valid = roi[roi > 0]

        # 유효 depth 값이 없으면 None 반환
        if valid.size == 0:
            return None
        return float(np.median(valid)) / 1000.0

    @staticmethod
    def _verify_code(decoded: str) -> bool:
        return hashlib.sha256(decoded.encode()).hexdigest() == _SECRET_CODE_HASH
