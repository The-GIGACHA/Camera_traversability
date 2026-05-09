from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

import numpy as np
from sensor_msgs.msg import CameraInfo

from .config import DEPTH_MIN_M, DEPTH_MAX_M, DEPTH_MEDIAN_RATIO

if TYPE_CHECKING:
    from .yolo_detector import Detection


@dataclass(frozen=True)
class CamPoint:
    """카메라 좌표계 3D 점 (X: 우, Y: 하, Z: 전방/depth)."""
    X: float
    Y: float
    Z: float
    cls: int


class DepthProjector:
    """
    Step 3: YOLO Detection bbox 내 depth → 카메라 3D 좌표 (CamPoint).

    bbox 하단 DEPTH_MEDIAN_RATIO 구간의 depth 중앙값을 대표 거리로 사용.
    발/바퀴 부분이 지면에 가장 가까우므로 하단 픽셀이 더 정확한 거리를 줌.
    """

    def project(
        self,
        detections: List["Detection"],
        depth_img: np.ndarray,      # uint16 또는 float32, mm 단위
        info_msg: CameraInfo,
    ) -> List[CamPoint]:
        """
        detections + depth_img + CameraInfo → CamPoint 리스트.

        depth_img가 uint16이면 mm 정수값, float32면 NaN 포함 가능.
        유효 depth 범위 [DEPTH_MIN_M, DEPTH_MAX_M] 밖의 픽셀은 무시.
        """
        fx   = float(info_msg.K[0])
        fy   = float(info_msg.K[4])
        cx_c = float(info_msg.K[2])
        cy_c = float(info_msg.K[5])

        img_h, img_w = depth_img.shape[:2]
        cam_points: List[CamPoint] = []

        for det in detections:
            x1 = max(0, int(det.x1))
            y1 = max(0, int(det.y1))
            x2 = min(img_w, int(det.x2))
            y2 = min(img_h, int(det.y2))

            if x2 <= x1 or y2 <= y1:
                continue

            # bbox 하단 DEPTH_MEDIAN_RATIO 구간만 ROI로 사용
            y_split = int(y1 + (y2 - y1) * (1.0 - DEPTH_MEDIAN_RATIO))
            y_split = max(y_split, y1)

            roi_mm = depth_img[y_split:y2, x1:x2].astype(np.float32)
            roi_m  = roi_mm / 1000.0

            valid = roi_m[np.isfinite(roi_m) & (roi_m >= DEPTH_MIN_M) & (roi_m <= DEPTH_MAX_M)]
            if len(valid) == 0:
                continue

            Z = float(np.median(valid))
            # bbox 중심 픽셀(cx, cy)을 대표 방향으로 역투영
            X = (det.cx - cx_c) / fx * Z
            Y = (det.cy - cy_c) / fy * Z

            cam_points.append(CamPoint(X=X, Y=Y, Z=Z, cls=det.cls))

        return cam_points
