# campus_delivery_auth — 모델 파일

- `models/hand_landmarker.task` : 손 제스처/수화 인증용 (MediaPipe Tasks API)
    ```bash
    curl -L -o models/hand_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    ```
- `models/wechat_qr/` : WeChat QR 4종 파일 필요
    - detect.prototxt / detect.caffemodel
    - sr.prototxt / sr.caffemodel
- `weights/yolov8n-pose.pt` : (구) 몸 제스처 인증용 — 손가락 수화는 hand_landmarker 사용
