from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from .config import YOLO_CONF_THRESH, TARGET_CLASSES, MASK_CLASSES


@dataclass(frozen=True)
class Detection:
    """동적 장애물 파이프라인용 탐지 결과."""
    cx: float
    cy: float
    bw: float
    bh: float
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class YoloDetector:
    """
    Step 2: YOLOv8 추론.

    두 가지 출력을 제공:
      1. detect()          — 동적 장애물 파이프라인용 Detection 리스트
      2. mask_depth()      — SRE용 depth 마스킹
                             사람/자전거 bbox 영역을 NaN으로 치환해
                             Z 분산 오염을 원천 차단.

    SRE 파이프라인 흐름:
      color_img → detect() → bbox 목록
                           → mask_depth(depth_img, bbox 목록)
                           → 마스킹된 depth → sre_mapper
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thresh: float = YOLO_CONF_THRESH,
    ):
        self._model      = YOLO(model_path)
        self._conf_thresh = conf_thresh

    # ── 동적 장애물 파이프라인용 ──────────────────────────────────────── #

    def detect(self, color_img: np.ndarray) -> List[Detection]:
        """person / bicycle bbox 탐지."""
        results = self._model.predict(
            color_img, conf=self._conf_thresh, verbose=False
        )
        dets: List[Detection] = []
        if not results:
            return dets

        boxes = results[0].boxes
        if boxes is None:
            return dets

        for box in boxes:
            cls = int(box.cls[0])
            if cls not in TARGET_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dets.append(Detection(
                cx=cx, cy=cy,
                bw=x2-x1, bh=y2-y1,
                cls=cls, conf=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))
        return dets

    # ── SRE 파이프라인용 ──────────────────────────────────────────────── #

    def detect_and_mask_depth(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray,
    ) -> Tuple[List[Detection], np.ndarray]:
        """
        YOLO 추론 + depth 마스킹을 한 번의 추론으로 처리.

        반환:
          detections : 동적 장애물 Detection 리스트 (동적 장애물 파이프라인 공유)
          depth_masked: MASK_CLASSES bbox 영역이 NaN으로 치환된 float32 depth (mm)

        마스킹 이유:
          사람/자전거가 지면 위에 있으면 해당 그리드 셀의 Z값이
          지면 + 사람 높이가 섞여 분산이 폭발적으로 커짐.
          NaN으로 치환하면 sre_mapper의 bincount가 이 픽셀을 무시하고,
          costmap_publisher에서 해당 셀에 COST_LETHAL을 별도 할당함.
        """
        results = self._model.predict(
            color_img, conf=self._conf_thresh, verbose=False
        )

        # depth float32 복사 (원본 보존)
        depth_masked = depth_img.astype(np.float32).copy()

        dets: List[Detection] = []
        if not results:
            return dets, depth_masked

        boxes = results[0].boxes
        if boxes is None:
            return dets, depth_masked

        h, w = depth_masked.shape[:2]

        for box in boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # SRE depth 마스킹 (MASK_CLASSES)
            if cls in MASK_CLASSES:
                x1i = max(0, int(x1))
                y1i = max(0, int(y1))
                x2i = min(w, int(x2))
                y2i = min(h, int(y2))
                depth_masked[y1i:y2i, x1i:x2i] = np.nan

            # 동적 장애물 Detection 수집 (TARGET_CLASSES)
            if cls in TARGET_CLASSES:
                conf = float(box.conf[0])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                dets.append(Detection(
                    cx=cx, cy=cy,
                    bw=x2-x1, bh=y2-y1,
                    cls=cls, conf=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))

        return dets, depth_masked
