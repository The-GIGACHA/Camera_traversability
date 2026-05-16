# camera_passability_in_real_world

1:5 scale RC car의 실주행 환경에서 RealSense D455 한 대만으로 **정적 지면 거칠기(Traversability) costmap + 동적 객체 통과성(Passability) 판단 + Local Path 생성**까지 처리하는 ROS1 패키지.

장기적으로 **(1) 통과성·주행 모듈** 과 **(2) 비전 인증·적재함 개방 모듈** 두 축으로 구성되지만, 현재 레포의 활성 작업 범위는 **(1) 통과성·주행 모듈** 입니다. (2) 모듈은 future work — [Future Work](#future-work) 참고.

---

## 실행 환경 (Environment)

| 항목 | 값 |
|---|---|
| OS | Ubuntu 20.04 LTS |
| ROS | ROS 1 Noetic |
| Python | 3.8 |
| 카메라 | Intel RealSense D455 (Color + Aligned Depth + IMU) |
| 차량 | 1:5 Scale RC car (실측 차폭 0.72 m) |
| 가속 | YOLOv8n CPU 추론 (필요 시 CUDA) | -> 추후 모델 학습 후에 TensorRT FP16으로 추론 최적화

> 현재 검증은 **rosbag 재생**으로 진행 중. compressedDepth 토픽을 그대로 처리하므로 실주행 시 카메라 노드 raw 전환 없이도 동일 코드가 동작합니다 (Compressed 토픽 사용 유지가 대역폭 측면에서 유리).

---

## Architecture PipeLine

```
RealSense D455
   │
   ├─ /camera/color/image_raw/compressed
   ├─ /camera/aligned_depth_to_color/image_raw/compressedDepth
   └─ /camera/color/camera_info
                │
                ▼
        ┌──── CameraSynchronizer (Step 1) ────┐
        │   Color/Depth 2-way ApproxSync      │
        │   CameraInfo 별도 캐시              │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
 ┌─────────────────────┐         ┌─────────────────────┐
 │  Dynamic Pipeline   │         │ Traversability      │
 │  (사람·자전거 회피)     │         │ Pipeline (SRE)      │
 ├─────────────────────┤         ├─────────────────────┤
 │ YoloDetector        │         │ YoloDetector mask   │
 │ DepthProjector      │         │ SREMapper           │
 │ TfPointTransformer  │         │ CostmapPublisher    │
 │ fov_filter          │         └─────────┬───────────┘
 │ PassabilityJudger   │                   │
 └──────────┬──────────┘                   ▼
            │                    /traversability/costmap
            ▼                    (nav_msgs/OccupancyGrid)
 /dynamic_obstacles/passable                │
 (std_msgs/Bool)                            ▼
            │                       LocalPathPlanner
 /dynamic_obstacles/pointcloud      (Forward DP)
 (sensor_msgs/PointCloud2)                  │
            └────────┐         ┌────────────┘
                     ▼         ▼
                 /camera/local_path
                 (nav_msgs/Path, base_footprint frame)
                          │
                          ▼
                  VisualizerNode (OpenCV BEV)
```

두 파이프라인은 **camera_sync 만 공유** 하고 그 외 독립적으로 동작합니다. 결과는 `CostmapPublisher` 에서 **dynamic LETHAL 오버레이 + SRE geometric cost + (옵션) seg + (옵션) lidar layer** 4겹으로 융합돼 단일 OccupancyGrid 가 발행됩니다.

---

## 모듈 단위 책임 (Module Responsibilities)

`traversability_ws/src/camera_passability/` 패키지 안에 모든 코드가 모여 있습니다. ROS 노드 (entry point) 는 `scripts/`, 재사용 가능한 클래스는 `src/camera_passability/` 에 둡니다.

### scripts/ — ROS 노드

| 파일 | 역할 |
|---|---|
| `Traversability_traversability_node.py` | SRE costmap 메인 노드. IMU 흔들림 게이팅 → SRE 계산 → OccupancyGrid 발행 |
| `camera_passability_node.py` | 동적 객체 메인 노드. YOLO → 3D 역투영 → TF → FOV → passability → PointCloud2 + Bool 발행 |
| `local_path_node.py` | costmap 수신 → Forward DP → `nav_msgs/Path` 발행 |
| `visualizer_node.py` | OpenCV 한 창에 카메라 영상 + BEV costmap + path + dynamic obstacle 마커 표시 |

### src/camera_passability/ — 라이브러리 모듈

| 파일 | 역할 | 핵심 함수 / 클래스 |
|---|---|---|
| `camera_sync.py` | Color + Depth `CompressedImage` 2-way ApproxSync. `compressedDepth` 페이로드(12B 헤더 + PNG)를 cv_bridge 우회로 직접 디코드. 16UC1 / 32FC1 분기. | `CameraSynchronizer` |
| `yolo_detector.py` | YOLOv8n 추론 + 동적 클래스 bbox 영역의 depth NaN 마스킹 (SRE 오염 방지) | `YoloDetector.detect_and_mask_depth` |
| `depth_projector.py` | bbox → 카메라 3D 좌표. bbox 하단부 depth 중앙값을 대표 거리로 사용 | `DepthProjector.project` |
| `tf_transformer.py` | `camera_color_optical_frame` → `base_footprint` 점 변환. tf2 사용 | `TfPointTransformer.transform_points` |
| `fov_filter.py` | base_footprint 기준 전방 부채꼴 + `MAX_RANGE_M` 거리 필터. 거리순 정렬 | `filter_fov` |
| `passability_judger.py` | **Corridor-based** 통과성 판단. `\|y\| ≤ PASS_THRESHOLD_M/2` 안 장애물의 lateral gap이 `ROBOT_WIDTH_M` 이상이면 PASS | `PassabilityJudger.judge` |
| `Traversability_sre_mapper.py` | depth → camera 3D → base_footprint 변환 → 셀별 Z 통계(`np.bincount`) → variance / delta / abs 세 지표 융합 → float32 0~1 cost grid | `SREMapper.compute` |
| `Traversability_costmap_publisher.py` | SRE cost + (seg 옵션) + (lidar stub) + dynamic LETHAL 오버레이 → `nav_msgs/OccupancyGrid` 발행. dynamic 셀은 1셀 인플레이션 포함 | `CostmapPublisher.publish` |
| `local_path_planner.py` | Forward DP. 매 행마다 좌/직진/우 3-인접 전이, `LOCAL_PATH_LATERAL_COST` 페널티로 직진 선호 | `LocalPathPlanner.plan` |
| `config.py` | 모든 토픽/프레임/임계값 상수 중앙 집중 | (상수만) |

---

## I/O Contract (인터페이스)

다른 팀(주행 제어, 라이다)과 맞물리는 부분 — **이 토픽/프레임 계약은 절대 깨지지 않게 유지**합니다.

### Subscribed

| Topic | Type | 비고 |
|---|---|---|
| `/camera/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | jpeg |
| `/camera/aligned_depth_to_color/image_raw/compressedDepth` | `sensor_msgs/CompressedImage` | 16UC1 / 32FC1 + PNG |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | fx/fy/cx/cy 출처 |
| `/camera/imu` | `sensor_msgs/Imu` | D455 내장. pitch/roll 임계 초과 프레임 skip |
| `/traversability/lidar_layer` *(stub)* | `nav_msgs/OccupancyGrid` | 라이다 팀이 채울 슬롯. 현재 비어있어도 동작 |

### Published

| Topic | Type | Frame | 비고 |
|---|---|---|---|
| `/traversability/costmap` | `nav_msgs/OccupancyGrid` | `base_footprint` | 25×20 셀, 0.20 m 해상도, 5 m × 4 m |
| `/camera/local_path` | `nav_msgs/Path` | `base_footprint` | DP로 뽑은 waypoint 시퀀스 |
| `/dynamic_obstacles/pointcloud` | `sensor_msgs/PointCloud2` | `base_footprint` | 사람·자전거 점 + (BLOCKED 시) 가상 벽 |
| `/dynamic_obstacles/passable` | `std_msgs/Bool` | — | `True`=PASS, `False`=BLOCKED |

### TF 요구사항

`base_footprint → camera_color_optical_frame` 경로가 완성되어 있어야 합니다. realsense2_camera 노드(또는 rosbag tf_static)가 카메라 내부 체인 `camera_link → … → camera_color_optical_frame` 을 발행한다면, **launch 파일이 `base_footprint → camera_link` 한 단계만 채워주면** 됩니다 (`launch/rosbag_test.launch` 참고).

⚠️ `base_footprint → camera_color_optical_frame` 을 직접 static publish 하면 안 됩니다. tf_static 의 기존 부모와 충돌해 lookup이 실패합니다.

---

## Configuration

`src/camera_passability/config.py` 가 단일 진실의 원천(single source of truth)입니다. 자주 조정하는 파라미터 값:

### 그리드 / 거리

```python
SRE_CELL_SIZE_M  = 0.20    # 한 셀 = 20 cm × 20 cm
SRE_MAX_RANGE_M  = 5.0     # 전방 5 m
SRE_HALF_WIDTH_M = 2.0     # 좌우 ±2 m
```

### 차량 물리 (실측 — 마운트 변경 시 갱신)

```python
ROBOT_WIDTH_M     = 0.72   # 1:5 RC카 차폭
SAFETY_MARGIN_M   = 0.08   # 좌우 각 4 cm 여유
PASS_THRESHOLD_M  = 0.80   # ROBOT_WIDTH + MARGIN
```

### SRE 임계값

```python
SRE_VAR_SAT      = 0.03    # Z 분산 포화 (거칠기 — 잔디·돌부리)
SRE_DELTA_SAT    = 0.08    # 인접 셀 높이차 포화 (단차 — 연석·계단)
SRE_ABS_H_THRESH = 0.15    # 절대 높이 임계 (벽·큰 장애물)
SRE_DEFAULT_COST = 0.35    # 미관측 셀의 default cost (0~1)
```

### 경로 계획

```python
LOCAL_PATH_LOOKAHEAD_M   = 3.0
LOCAL_PATH_LATERAL_COST  = 10.0   # 클수록 직진 선호
LOCAL_PATH_SMOOTH_WINDOW = 5
```

### YOLO

```python
YOLO_CONF_THRESH = 0.5
TARGET_CLASSES   = {0, 1}   # person, bicycle (COCO)
MASK_CLASSES     = {0, 1}   # SRE 마스킹 대상
```

---

## 실행 방법 (Usage)

### Workspace 빌드

```bash
cd traversability_ws
catkin_make            # 또는 catkin build
source devel/setup.bash
```

### rosbag 으로 통합 실행

```bash
roslaunch camera_passability rosbag_test.launch bag:=/path/to/your.bag
```

launch 인자:
- `bag` (필수): rosbag 파일 경로
- `rate` (기본 1.0): 재생 속도 배율
- `loop_flag` (기본 비활성): `--loop` 옵션 전달
- `yolo_model` (기본 `yolov8n.pt`): YOLO 가중치

### TF 측정값 갱신

`launch/rosbag_test.launch` 의 `static_transform_publisher` 인자를 실측값으로:

```xml
<node pkg="tf" type="static_transform_publisher" name="cam_tf"
      args="X Y Z   YAW PITCH ROLL   base_footprint camera_link 100"/>
```

| 인자 | 의미 | 단위 |
|---|---|---|
| X | base_footprint → 카메라 forward 거리 | m |
| Y | 좌우 오프셋 (+ 가 LEFT) | m |
| Z | 지면 → 카메라 높이 | m |
| YAW / PITCH / ROLL | 마운트 회전 | rad |

> 90° 광학 프레임 회전은 rosbag(또는 realsense 드라이버)의 tf_static 이 처리하므로 여기선 **로봇 ↔ 카메라 마운트 회전만** 선언합니다.

### 진단 명령

```bash
# TF 체인 살아있는지
rosrun tf tf_echo base_footprint camera_color_optical_frame

# costmap 발행 주기 확인
rostopic hz /traversability/costmap

# 동적 장애물 cloud 발행 주기 + 점 내용
rostopic hz /dynamic_obstacles/pointcloud
rostopic echo /dynamic_obstacles/pointcloud -n 1 | head -30

# Visualizer 로그 (SYNC FIRED / costmap published 가 일정 주기로 찍혀야 함)
rosnode info /visualizer_node
```

### Visualizer 색상 규약

- **노랑** = 미관측 셀 (`SRE_DEFAULT_COST`)
- **초록** = 관측 + 통과 가능 (cost ~0)
- **노랑→빨강 그라데이션** = SRE cost 증가
- **빨강 셀** = LETHAL (동적 장애물 + 인플레이션) 또는 SRE max cost
- **마젠타 원 + 검은 외곽** = `/dynamic_obstacles/pointcloud` 의 개별 동적 객체 (LETHAL 셀과 시각 분리)
- **파랑 선/점** = `/camera/local_path` waypoint
- **흰 삼각형** = 로봇 위치 (BEV 하단 중앙)

---

## 최근 개선 사항 (Recent Improvements)

이 레포는 rosbag 검증 단계에서 다음 문제들을 순차적으로 잡았습니다. 각 항목은 별도 PR 로 merge됨.

| # | 문제 | 해결 |
|---|---|---|
| 1 | `compressedDepth` 토픽이 cv_bridge 에서 silently `None` 반환 → SRE 콜백 영구 중단 → costmap 발행 0회 | `camera_sync._decode_depth` 신설. 12B 헤더 직접 파싱 + `msg.format` 으로 16UC1/32FC1 분기 + 32FC1 inverse-depth 역양자화 |
| 2 | launch 의 `static_transform_publisher` 가 `base_footprint → camera_color_optical_frame` 으로 잡혀있어 tf_static 의 기존 부모와 충돌 → TF lookup 실패 → 전체 BEV가 `SRE_DEFAULT_COST` 노란색 | child 를 `camera_link` 로 변경. rosbag tf_static 의 90° optical 회전을 그대로 활용 |
| 3 | `_delta_score` 의 NaN→0 fill 이 FOV 가장자리에 fake wall halo 생성 | NaN/valid 경계 diff 를 0 으로 무시하는 pair-validity 마스크 적용 |
| 4 | TF placeholder 값이 실측과 안 맞아 좌측에 systematic red strip → planner 우측 편향 | 실측 마운트 값 + yaw -5° 가설 적용. 더불어 `SRE_DEFAULT_COST 0.5→0.35`, `LATERAL_COST 5→10` 으로 잔여 편향 완화 |
| 5 | `ROBOT_WIDTH_M` 가 0.4 placeholder. 실제는 0.72 → 통과 판정이 너무 관대 | 실측 0.72 반영, `SAFETY_MARGIN` 0.4→0.08 로 줄여 `PASS_THRESHOLD_M` 은 동일 유지 |
| 6 | `PassabilityJudger` 가 거리축 인접 쌍의 Euclidean 만 봐서 옆에 사람 무리만 있어도 BLOCKED. 위치 무관 | **corridor 기반 재설계**. `\|y\| ≤ PASS_THRESHOLD_M/2` 안 장애물만 고려, lateral gap 으로 PASS/BLOCKED 판단 |
| 7 | Dynamic obstacle 원이 LETHAL 셀과 같은 빨강이라 시각화 묻힘 | 마젠타 + 검은 외곽으로 변경 |

---

## 알려진 한계 (Known Limitations)

- **카메라 자연 사각지대**: D455 RGB 수직 FOV 58° + 카메라 높이 0.995 m + pitch≈0 → 전방 ~2.2 m 이하 ground 가 카메라에 안 보임. BEV 하단부가 항상 default 노랑. 가까운 정적 장애물(연석, 쓰레기통 등)은 동적 파이프라인(YOLO)만 잡을 수 있어 안전 사각지대. **카메라 down-pitch 또는 보조 센서(소나/lidar) 가 권장 해결책(약 5~10°도 정도)**.
- **YOLO 한정 클래스**: 현재 `person`, `bicycle` 만 동적으로 처리. 추후 차량·전동킥보드 등 확장 예정.
- **Seg layer**: `CostmapPublisher.publish(seg_grid=None)` 으로 비활성. YOLOv8-seg 팀원 출력 연결 시 활성화.
- **Lidar layer**: `/traversability/lidar_layer` 구독 stub 만 존재. 라이다 팀이 같은 그리드 규약으로 발행하면 자동 오버레이.
- **Corridor judge 의 단순성**: 현재 corridor 는 직선(차량 forward)만 가정. 실제로는 local_path 가 휘어 있을 때 그 곡선을 따라 corridor 가 휘는 게 더 정확. 추후 path-aware 판정으로 고도화 가능.

---

## Future Work

### Vision Authentication + Cargo Unlock (`campus_delivery_auth/`)

배송 로봇이 목적지 도착(`/robot_state == "ARRIVED"`) 후 수령인을 비전으로 인증하고 적재함을 개방하는 별도 파이프라인. 현재 통과성 모듈 완성 후 착수 예정이라 이 레포에서는 placeholder 디렉터리만 존재(Vibe Coding). 설계 의도:

- 상태 머신 기반 (idle / ARRIVED / NAVIGATING) — 비활성 구간에 카메라 구독 완전 해제로 자원 양보
- WeChat QR (1차) + Gesture (2차, YOLOv8-gesture) 이중 인증
- aligned_depth 로 카메라 ≤ 0.5 m 객체만 유효 인증으로 처리 (혼잡 환경 오인식 방지)
- 인증 성공 시 `/cargo_unlock` (`std_msgs/Bool`) 발행 → 제어팀이 CAN/Arduino 로 적재함 잠금 해제

상세 설계는 통과성 모듈 안정화 후 별도 PR.

### Local path → 경로 추종

현재는 path 발행까지만. 실제 차량 조향·속도 제어는 주행 제어팀과 인터페이스 합의 후 별도 노드 (예: `path_tracker_node.py`) 로 분리 예정.

---

## 패키지 구조 한눈에

```
traversability_ws/
├── src/
│   └── camera_passability/
│       ├── launch/
│       │   └── rosbag_test.launch         ← TF 측정값 반영하는 곳
│       ├── scripts/                        ← ROS 노드 entrypoint
│       │   ├── Traversability_traversability_node.py
│       │   ├── camera_passability_node.py
│       │   ├── local_path_node.py
│       │   └── visualizer_node.py
│       ├── src/camera_passability/         ← 라이브러리 모듈
│       │   ├── camera_sync.py
│       │   ├── yolo_detector.py
│       │   ├── depth_projector.py
│       │   ├── tf_transformer.py
│       │   ├── fov_filter.py
│       │   ├── passability_judger.py
│       │   ├── Traversability_sre_mapper.py
│       │   ├── Traversability_costmap_publisher.py
│       │   ├── local_path_planner.py
│       │   └── config.py                    ← 모든 상수
│       ├── weights/                         ← YOLO 모델 (yolov8n.pt)
│       ├── CMakeLists.txt
│       ├── package.xml
│       └── setup.py
└── (devel/, build/ 등 빌드 산출물)

campus_delivery_auth/                        ← future work, 현재 stub
```
