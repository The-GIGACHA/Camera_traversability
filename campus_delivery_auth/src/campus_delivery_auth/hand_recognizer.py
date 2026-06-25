"""
hand_recognizer.py  (ROS1)
==========================
MediaPipe Hands 기반 손 제스처/수화 인식 모듈 (QR 실패 시 fallback 인증).

설계:
  - MediaPipe(`mediapipe`)는 지연 import → 미설치 환경에서도 모듈 import 가능.
  - 21개 hand landmark → 제스처 분류는 **순수 함수**(classify_hand_gesture)로 분리.
    카메라/MediaPipe 없이 합성 landmark 로 단위테스트 가능.
  - GestureRecognizer 와 동일한 인터페이스:
        recognize(color_img, depth_img, camera_info) -> HandResult
        reset()
    → vision_auth_node 에 드롭인 교체/추가 가능.

MediaPipe landmark 인덱스 (21점):
  0 wrist
  1-4   thumb  (CMC, MCP, IP, TIP)
  5-8   index  (MCP, PIP, DIP, TIP)
  9-12  middle
  13-16 ring
  17-20 pinky
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from sensor_msgs.msg import CameraInfo
import rospy


# ── 파라미터 상수 ──────────────────────────────────────────────────────── #

HAND_DETECTION_CONF = 0.6
HAND_TRACKING_CONF  = 0.5
MAX_NUM_HANDS       = 1

DEPTH_VALID_MAX_M   = 0.5
DEPTH_VALID_MIN_M   = 0.1
DEPTH_PERCENTILE    = 20
CONFIRM_FRAMES      = 5

# Tasks API 모델 파일 (campus_delivery_auth/models/hand_landmarker.task)
_DEFAULT_TASK = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models', 'hand_landmarker.task'
)

# 인증으로 인정할 제스처 (테스트로 최적 동작 선정)
VALID_GESTURES: set = {'open_palm'}

# landmark 인덱스
WRIST = 0
THUMB_IP, THUMB_TIP            = 3, 4
INDEX_PIP, INDEX_TIP           = 6, 8
MIDDLE_PIP, MIDDLE_TIP         = 10, 12
RING_PIP, RING_TIP             = 14, 16
PINKY_PIP, PINKY_TIP           = 18, 20


@dataclass
class HandResult:
    detected:   bool
    gesture:    str   = ''
    confidence: float = 0.0
    depth_m:    float = 0.0
    num_hands:  int   = 0
    reason:     str   = ''


# ═══════════════════════════════════════════════════════════════════════ #
#  순수 함수: landmark → 제스처  (ROS/MediaPipe 비의존, 단위테스트 대상)
# ═══════════════════════════════════════════════════════════════════════ #

def finger_states(lm: np.ndarray, handedness: str) -> dict:
    """
    lm: (21, 3) 정규화 landmark 배열 (x, y, z), y는 위로 갈수록 작음.
    handedness: 'Right' | 'Left' (MediaPipe 라벨, 카메라 시점 기준).
    반환: {thumb, index, middle, ring, pinky: bool}  (펴짐=True)
    """
    def ext_vert(tip_i, pip_i) -> bool:
        # 손가락이 위를 향한다는 가정: tip 이 pip 보다 위(=y 작음)면 펴진 것
        return bool(lm[tip_i, 1] < lm[pip_i, 1])

    # 엄지는 좌우(x)로 펴짐 — handedness 에 따라 방향 반전
    if handedness == 'Right':
        thumb = bool(lm[THUMB_TIP, 0] < lm[THUMB_IP, 0])
    else:
        thumb = bool(lm[THUMB_TIP, 0] > lm[THUMB_IP, 0])

    return {
        'thumb':  thumb,
        'index':  ext_vert(INDEX_TIP,  INDEX_PIP),
        'middle': ext_vert(MIDDLE_TIP, MIDDLE_PIP),
        'ring':   ext_vert(RING_TIP,   RING_PIP),
        'pinky':  ext_vert(PINKY_TIP,  PINKY_PIP),
    }


def classify_hand_gesture(
    lm: np.ndarray, handedness: str = 'Right'
) -> Tuple[str, float]:
    """21점 landmark → (gesture, confidence[0..1]). 미분류는 ('', 0.0)."""
    st  = finger_states(lm, handedness)
    ext = {f for f, v in st.items() if v}
    n   = len(ext)

    if n == 5:
        return 'open_palm', 0.9
    if n == 0:
        return 'fist', 0.9
    if ext == {'index', 'middle'}:
        return 'victory', 0.8
    if ext == {'index'}:
        return 'point', 0.7
    if ext == {'thumb'}:
        return 'thumbs_up', 0.7
    return '', 0.0


# ═══════════════════════════════════════════════════════════════════════ #
#  N-프레임 확정 버퍼
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class _FrameBuffer:
    maxlen: int
    buf: deque = field(default_factory=deque)

    def push(self, gesture: str) -> None:
        self.buf.append(gesture)
        if len(self.buf) > self.maxlen:
            self.buf.popleft()

    def is_confirmed(self, gesture: str) -> bool:
        return len(self.buf) == self.maxlen and all(g == gesture for g in self.buf)

    def reset(self) -> None:
        self.buf.clear()


# ═══════════════════════════════════════════════════════════════════════ #
#  MediaPipe 래퍼
# ═══════════════════════════════════════════════════════════════════════ #

class HandRecognizer:

    def __init__(self) -> None:
        self._hands  = self._init_mediapipe()
        self._buffer = _FrameBuffer(maxlen=CONFIRM_FRAMES)
        rospy.loginfo('HandRecognizer: MediaPipe Hands ready')

    @staticmethod
    def _init_mediapipe():
        """MediaPipe Tasks API(HandLandmarker, IMAGE 모드) 초기화."""
        try:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError(
                'mediapipe 미설치 — `pip install mediapipe` 후 사용하세요'
            ) from e

        task_path = os.environ.get('HAND_LANDMARKER_TASK', _DEFAULT_TASK)
        if not os.path.exists(task_path):
            raise RuntimeError(
                f'hand_landmarker.task 없음: {task_path}\n'
                '  → models/README.md 의 URL 에서 받아 두세요'
            )

        options = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=task_path),
            num_hands=MAX_NUM_HANDS,
            min_hand_detection_confidence=HAND_DETECTION_CONF,
            min_hand_presence_confidence=HAND_DETECTION_CONF,
            min_tracking_confidence=HAND_TRACKING_CONF,
            running_mode=vision.RunningMode.IMAGE,
        )
        return vision.HandLandmarker.create_from_options(options)

    def recognize(
        self,
        color_img:   np.ndarray,
        depth_img:   np.ndarray,
        camera_info: CameraInfo,
    ) -> HandResult:
        import cv2  # 지연 import (ROS 환경 가정)
        import mediapipe as mp

        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._hands.detect(mp_image)

        if not res.hand_landmarks:
            self._buffer.push('')
            return HandResult(detected=False, reason='no_hand')

        h, w = color_img.shape[:2]
        best = None  # (gesture, conf, depth_m)

        for lms, handed in zip(res.hand_landmarks, res.handedness):
            lm    = np.array([[p.x, p.y, p.z] for p in lms])
            label = handed[0].category_name  # 'Right'/'Left'

            gesture, conf = classify_hand_gesture(lm, label)
            if gesture not in VALID_GESTURES:
                continue

            depth_m = self._measure_depth(depth_img, lm, w, h)
            if depth_m is None or not (DEPTH_VALID_MIN_M <= depth_m <= DEPTH_VALID_MAX_M):
                continue

            if best is None or conf > best[1]:
                best = (gesture, conf, depth_m)

        if best is None:
            self._buffer.push('')
            return HandResult(
                detected=False, num_hands=len(res.hand_landmarks),
                reason='no_valid_gesture',
            )

        gesture, conf, depth_m = best
        self._buffer.push(gesture)

        if self._buffer.is_confirmed(gesture):
            return HandResult(
                detected=True, gesture=gesture, confidence=conf,
                depth_m=depth_m, num_hands=len(res.hand_landmarks),
            )

        done = len(self._buffer.buf)
        return HandResult(
            detected=False, gesture=gesture, confidence=conf, depth_m=depth_m,
            num_hands=len(res.hand_landmarks),
            reason=f'confirming:{done}/{CONFIRM_FRAMES}',
        )

    def reset(self) -> None:
        self._buffer.reset()

    # ── depth: landmark bbox 영역 percentile ──────────────────────────── #

    @staticmethod
    def _measure_depth(depth_img, lm, w, h) -> Optional[float]:
        xs = (lm[:, 0] * w).astype(int)
        ys = (lm[:, 1] * h).astype(int)
        x1, x2 = max(0, xs.min()), min(w, xs.max())
        y1, y2 = max(0, ys.min()), min(h, ys.max())
        roi   = depth_img[y1:y2, x1:x2]
        valid = roi[roi > 0]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, DEPTH_PERCENTILE)) / 1000.0
