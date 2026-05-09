"""
gesture_recognizer.py  (ROS1)
==============================
YOLOv8 기반 2차 제스처 인증 모듈.

ROS1 포팅 변경사항:
  - rospy.logwarn/logdebug 사용
  - 나머지 로직은 ROS-독립적이므로 변경 없음
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sensor_msgs.msg import CameraInfo
from ultralytics import YOLO
import rospy


GESTURE_CONF_THRESH = 0.6 # confidence test하면서 최적화
DEPTH_VALID_MAX_M   = 0.5 
DEPTH_VALID_MIN_M   = 0.1
CONFIRM_FRAMES      = 5
DEPTH_PERCENTILE    = 20

VALID_GESTURES: set = {'both_hands_up'}

KP_LEFT_WRIST    = 9
KP_RIGHT_WRIST   = 10
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER= 6
KP_NOSE          = 0


@dataclass
class GestureResult:
    detected:   bool
    gesture:    str   = ''
    confidence: float = 0.0
    depth_m:    float = 0.0
    reason:     str   = ''


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


class GestureRecognizer:

    def __init__(self, model_path: str = 'yolov8n-pose.pt') -> None:
        self._model  = YOLO(model_path)
        self._buffer = _FrameBuffer(maxlen=CONFIRM_FRAMES)
        rospy.loginfo(f'GestureRecognizer: model={model_path}')

    def recognize(
        self,
        color_img:   np.ndarray,
        depth_img:   np.ndarray,
        camera_info: CameraInfo,
    ) -> GestureResult:
        results = self._model.predict(
            color_img, conf=GESTURE_CONF_THRESH, verbose=False
        )

        best = None
        for person in results[0]:
            if person.keypoints is None:
                continue
            kps     = person.keypoints.xy[0].cpu().numpy()
            kp_conf = person.keypoints.conf
            if kp_conf is None:
                continue
            kp_conf = kp_conf[0].cpu().numpy()

            box     = person.boxes.xyxy[0].cpu().numpy()
            depth_m = self._measure_depth(depth_img, box)
            if depth_m is None or not (DEPTH_VALID_MIN_M <= depth_m <= DEPTH_VALID_MAX_M):
                continue

            gesture, conf = self._classify_gesture(kps, kp_conf)
            if gesture not in VALID_GESTURES:
                continue
            if best is None or conf > best[1]:
                best = (gesture, conf, depth_m)

        if best is None:
            self._buffer.push('')
            return GestureResult(detected=False, reason='no_valid_gesture')

        gesture, conf, depth_m = best
        self._buffer.push(gesture)

        if self._buffer.is_confirmed(gesture):
            return GestureResult(
                detected=True, gesture=gesture,
                confidence=conf, depth_m=depth_m,
            )

        done = len(self._buffer.buf)
        return GestureResult(
            detected=False, gesture=gesture, confidence=conf, depth_m=depth_m,
            reason=f'confirming:{done}/{CONFIRM_FRAMES}',
        )

    def reset(self) -> None:
        self._buffer.reset()

    @staticmethod
    def _measure_depth(depth_img, box) -> Optional[float]:
        x1, y1, x2, y2 = map(int, box)
        h, w = depth_img.shape[:2]
        roi   = depth_img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        valid = roi[roi > 0]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, DEPTH_PERCENTILE)) / 1000.0

    @staticmethod
    def _classify_gesture(kps, kp_conf, conf_thresh=0.5):
        def ok(i): return float(kp_conf[i]) > conf_thresh
        if not (ok(KP_LEFT_WRIST) and ok(KP_RIGHT_WRIST) and ok(KP_NOSE)):
            return '', 0.0
        if kps[KP_LEFT_WRIST][1] < kps[KP_NOSE][1] and \
           kps[KP_RIGHT_WRIST][1] < kps[KP_NOSE][1]:
            avg = float(kp_conf[KP_LEFT_WRIST] + kp_conf[KP_RIGHT_WRIST]) / 2.0
            return 'both_hands_up', avg
        return '', 0.0
