"""
config.py
=========
공통 파라미터/토픽/프레임 계약.
기존 동적 장애물 파이프라인 상수 + SRE traversability 파라미터 추가.
"""
import os as _os

# ── YOLO 모델 경로 ────────────────────────────────────────────────────── #
# 파일 위치 계층 (config.py 기준):
#   _SRC_CA_DIR : <ws>/src/camera_passability/src/camera_passability/
#   _SRC_DIR    : <ws>/src/camera_passability/src/
#   _PKG_DIR    : <ws>/src/camera_passability/          (ROS 패키지 루트)
#   _WS_DIR     : <ws>/                                 (catkin 워크스페이스 루트)
_SRC_CA_DIR = _os.path.dirname(_os.path.abspath(__file__))
_SRC_DIR    = _os.path.dirname(_SRC_CA_DIR)
_PKG_DIR    = _os.path.dirname(_SRC_DIR)
_WS_DIR     = _os.path.dirname(_os.path.dirname(_PKG_DIR))

# 모델은 워크스페이스 루트(traversability_ws/yolov8n.pt) 에 둔다고 가정.
# 패키지 내부 weights/ 디렉터리에 두는 게 더 깔끔하지만 사용자가 ws 루트 방식 선호.
# launch 인자 yolo_model:=<absolute path> 로 언제든지 override 가능
# (Traversability/camera_passability 노드가 ~yolo_model 파라미터 읽음).
def _resolve_yolo_model() -> str:
    """존재하는 후보 중 첫 번째 반환. 없으면 ws 루트의 yolov8n.pt 를 default 로."""
    candidates = [
        _os.path.join(_WS_DIR, "yolov8n.pt"),                 # 사용자 현재 배치 (1순위)
        _os.path.join(_PKG_DIR, "weights", "yolov8n.pt"),    # 관례적 위치
        _os.path.join(_WS_DIR, "weights", "yolov8n.pt"),     # ws/weights/ 가능성
    ]
    for c in candidates:
        if _os.path.exists(c):
            return c
    return candidates[0]   # 못 찾으면 ws 루트 경로 반환 (Ultralytics 가 자동 다운로드 시도)

YOLO_MODEL_PATH = _resolve_yolo_model()

# ── 프레임 / 토픽 ─────────────────────────────────────────────────────── #

BASE_LINK_FRAME = "base_footprint" # 로봇 베이스 프레임 (tf 기준)

TOPIC_COLOR_COMPRESSED = "/camera/color/image_raw/compressed"
TOPIC_DEPTH_COMPRESSED = "/camera/aligned_depth_to_color/image_raw/compressedDepth"
TOPIC_CAMERA_INFO      = "/camera/color/camera_info"

TOPIC_DYNAMIC_OBSTACLES = "/dynamic_obstacles/pointcloud"
TOPIC_PASSABLE          = "/dynamic_obstacles/passable"

# SRE 출력 토픽
TOPIC_TRAVERSABILITY_COSTMAP = "/traversability/costmap"

# ── 동적 장애물 파이프라인 ────────────────────────────────────────────── #

# 실측 1:5 RC카 차폭 (이전 placeholder 0.4 → 0.72)
ROBOT_WIDTH_M      = 0.72
# 좌우 각 4 cm 여유. 실측 차폭이 반영되면서 마진은 줄여 PASS_THRESHOLD_M을
# 이전과 동일한 0.80 m로 유지 (실측 정확도 ↑, 통과성 판단 강도는 유지).
SAFETY_MARGIN_M    = 0.08
PASS_THRESHOLD_M   = ROBOT_WIDTH_M + SAFETY_MARGIN_M   # 0.80 m

FOV_DEG     = 120.0
MAX_RANGE_M = 5.0

VIRTUAL_WALL_STEP_M = 0.05

# ── YOLO 공통 ─────────────────────────────────────────────────────────── #

YOLO_CONF_THRESH = 0.5
TARGET_CLASSES   = {0, 1}   # person=0, bicycle=1 (COCO)

# SRE 마스킹 대상 클래스 (Z 분산 오염 방지)
# 지면 위에 있으면 Z 분산을 발산시키는 객체 전부 포함
MASK_CLASSES = {0, 1}       # person, bicycle — TARGET_CLASSES와 동일

# ── Depth 공통 ────────────────────────────────────────────────────────── #

DEPTH_MIN_M        = 0.3
DEPTH_MAX_M        = 6.0
DEPTH_MEDIAN_RATIO = 0.2    # bbox 내 하위 percentile (동적 장애물 파이프라인용)

# ── Step 1: 시간 동기화 ───────────────────────────────────────────────── #

SYNC_SLOP_SEC = 0.2

# ── SRE (Surface Roughness Estimation) ───────────────────────────────── #

# 그리드 설계
SRE_CELL_SIZE_M  = 0.20     # 셀 크기 20 cm
SRE_MAX_RANGE_M  = 5.0      # 전방 최대 거리
SRE_HALF_WIDTH_M = 2.0      # 좌우 범위 ±2 m

# depth 유효 범위 (SRE 전용 — 동적 장애물용과 별도 관리)
SRE_DEPTH_MIN_M  = 0.3
SRE_DEPTH_MAX_M  = 5.5

# 서브샘플링 (속도 vs 밀도 트레이드오프)
SRE_SUBSAMPLE    = 3        # N픽셀마다 1개 사용

# 셀 신뢰도 최소 포인트 수
SRE_MIN_POINTS   = 4

# ── SRE 세 가지 지표 포화값 (정규화 상한) ────────────────────────────── #

# 1. Z 분산 (거칠기): 잔디·수풀·돌부리
SRE_VAR_SAT      = 0.03     # m² — std 약 17 cm 에서 포화

# 2. 인접 셀 높이 차이 (단차): 연석·계단
SRE_DELTA_SAT    = 0.08     # m — 8 cm 단차에서 포화

# 3. 조건부 절대 높이 (벽·큰 장애물): 오르막 오탐 방지 조건 포함
SRE_ABS_H_THRESH  = 0.15   # m — 지면보다 15 cm 높으면 의심
SRE_SLOPE_FLAT    = 0.06    # m/m — 이 이하면 "평지"로 간주
                            # 오르막(slope ↑)이면 abs 페널티 면제

# ── SRE 지표 가중치 (합산 = 1.0) ─────────────────────────────────────── #

SRE_W_VAR   = 0.50
SRE_W_DELTA = 0.35
SRE_W_ABS   = 0.15

# ── Costmap 융합 ──────────────────────────────────────────────────────── #

# 동적 객체(person/bicycle) 셀 cost — OccupancyGrid 최대
COST_LETHAL         = 100

# seg free_path 셀에 대해 geometric cost를 얼마나 완화할지
# 0.0 = seg 완전 신뢰 / 1.0 = seg 무시
SEG_FREE_DISCOUNT   = 0.6

# seg obstacle 셀 고정 cost
SEG_OBSTACLE_COST   = 85

# 포인트 부족 셀 (unknown) 기본값
# 0.5 → 0.35 로 낮춤. 카메라 yaw 마운트 오차로 FOV가 비대칭이라
# 우측 close-range 셀만 SRE 데이터를 받아 cost가 낮게 계산되고, 좌측은
# DEFAULT 50으로 남아 planner가 우측 경로를 항상 선호하던 문제 완화용.
# (안전성 trade-off: 미관측 영역의 추정 위험도가 낮아짐. 실주행 시 재검토 필요)
SRE_DEFAULT_COST    = 0.35  # 0~1 정규화 기준

# ── IMU 보정 (D455 내장 IMU) ─────────────────────────────────────────── #

IMU_TOPIC           = "/camera/imu"
# pitch/roll 임계값 — 이 이상 흔들리면 해당 프레임 스킵
IMU_PITCH_MAX_DEG   = 45.0   # 테스트용 완화 (실서비스 시 10.0으로 복원)
IMU_ROLL_MAX_DEG    = 45.0   # 테스트용 완화 (실서비스 시 10.0으로 복원)

# ── Local Path Planning ───────────────────────────────────────────────── #

TOPIC_LOCAL_PATH         = "/camera/local_path"

# 전방 lookahead 거리 (m) — 이 거리만큼의 경로 점을 생성
LOCAL_PATH_LOOKAHEAD_M   = 3.0

# 좌우 이동 1셀당 추가 비용 — 클수록 직진 선호
# 5.0 → 10.0 으로 올림. 카메라 yaw 비대칭으로 인한 우측 편향을 상쇄해
# planner가 더 강하게 중앙(직진)을 고수하도록 함.
LOCAL_PATH_LATERAL_COST  = 10.0

# 이동 평균 스무딩 윈도우 (셀 수)
LOCAL_PATH_SMOOTH_WINDOW = 5
