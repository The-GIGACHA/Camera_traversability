#!/usr/bin/env python3
"""
hand_demo.py
============
ROS 없이 웹캠으로 MediaPipe(Tasks API) 손 제스처 인식을 실시간 확인하는 데모.
인식 로직(classify_hand_gesture)은 ROS 노드와 동일.

설치:  pip install mediapipe opencv-python
모델:  campus_delivery_auth/models/hand_landmarker.task (models/README.md 참고)
실행:  python3 tools/hand_demo.py
종료:  q
"""

import os
import sys

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from campus_delivery_auth.hand_recognizer import classify_hand_gesture  # noqa: E402

TASK = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')


def main() -> None:
    if not os.path.exists(TASK):
        print(f'모델 없음: {TASK}\n  models/README.md 의 URL 에서 받으세요.')
        return

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=TASK),
            num_hands=1, running_mode=vision.RunningMode.IMAGE,
        )
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('웹캠을 열 수 없습니다.')
        return

    print('q 로 종료. 손을 카메라에 보여주세요.')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        label_text = 'no hand'
        if res.hand_landmarks:
            for lms, handed in zip(res.hand_landmarks, res.handedness):
                lm = np.array([[p.x, p.y, p.z] for p in lms])
                for p in lm:                                    # landmark 점 그리기
                    cv2.circle(frame, (int(p[0] * w), int(p[1] * h)), 3,
                               (0, 200, 255), -1)
                side = handed[0].category_name
                gesture, conf = classify_hand_gesture(lm, side)
                label_text = f'{side}: {gesture or "?"} ({conf:.2f})'

        cv2.putText(frame, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('hand_demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
