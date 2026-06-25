#!/usr/bin/env python3
"""
test_hand_recognizer_offline.py
===============================
MediaPipe/카메라/ROS 없이 손 제스처 분류 순수 로직만 검증.
합성 landmark(21×3) 를 만들어 classify_hand_gesture 를 직접 호출.

실행:  python3 test/test_hand_recognizer_offline.py
"""

import os
import sys
import types

import numpy as np

# ── ROS 모듈 stub (hand_recognizer 가 import 단계에서 rospy/sensor_msgs 요구) ── #
_rospy = types.ModuleType('rospy')
_rospy.loginfo = _rospy.logwarn = _rospy.logdebug = lambda *a, **k: None
sys.modules['rospy'] = _rospy

_sm     = types.ModuleType('sensor_msgs')
_sm_msg = types.ModuleType('sensor_msgs.msg')
_sm_msg.CameraInfo = object
_sm.msg = _sm_msg
sys.modules['sensor_msgs'] = _sm
sys.modules['sensor_msgs.msg'] = _sm_msg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from campus_delivery_auth.hand_recognizer import (  # noqa: E402
    classify_hand_gesture, finger_states,
)


def _hand(points: dict) -> np.ndarray:
    """필요한 landmark 인덱스만 채운 (21,3) 배열 생성."""
    lm = np.zeros((21, 3), dtype=float)
    for idx, (x, y) in points.items():
        lm[idx] = (x, y, 0.0)
    return lm


# 인덱스: thumb(ip=3,tip=4) index(pip=6,tip=8) middle(10,12) ring(14,16) pinky(18,20)
OPEN_PALM = _hand({
    3: (0.34, 0.72), 4: (0.28, 0.68),   # thumb 펴짐(tip.x < ip.x, Right)
    6: (0.45, 0.50), 8: (0.45, 0.30),   # index up
    10: (0.50, 0.50), 12: (0.50, 0.28), # middle up
    14: (0.55, 0.50), 16: (0.55, 0.32), # ring up
    18: (0.60, 0.50), 20: (0.60, 0.40), # pinky up
})

FIST = _hand({
    3: (0.40, 0.75), 4: (0.45, 0.78),   # thumb 접힘
    6: (0.45, 0.50), 8: (0.45, 0.60),   # index 접힘(tip.y > pip.y)
    10: (0.50, 0.50), 12: (0.50, 0.60),
    14: (0.55, 0.50), 16: (0.55, 0.60),
    18: (0.60, 0.50), 20: (0.60, 0.60),
})

VICTORY = _hand({
    3: (0.40, 0.75), 4: (0.45, 0.78),   # thumb 접힘
    6: (0.45, 0.50), 8: (0.45, 0.30),   # index up
    10: (0.50, 0.50), 12: (0.50, 0.28), # middle up
    14: (0.55, 0.50), 16: (0.55, 0.60), # ring 접힘
    18: (0.60, 0.50), 20: (0.60, 0.60), # pinky 접힘
})


def test_open_palm():
    g, c = classify_hand_gesture(OPEN_PALM, 'Right')
    assert g == 'open_palm', f'got {g}'
    assert c > 0.5
    print('  ✓ open_palm (5 손가락) 인식')


def test_fist():
    g, _ = classify_hand_gesture(FIST, 'Right')
    assert g == 'fist', f'got {g}'
    print('  ✓ fist (0 손가락) 인식')


def test_victory():
    g, _ = classify_hand_gesture(VICTORY, 'Right')
    assert g == 'victory', f'got {g}'
    print('  ✓ victory (검지+중지) 인식')


def test_left_hand_thumb_mirrored():
    # 같은 좌표라도 handedness 가 Left 면 thumb 판정이 반전되어야 함
    st_r = finger_states(OPEN_PALM, 'Right')
    st_l = finger_states(OPEN_PALM, 'Left')
    assert st_r['thumb'] != st_l['thumb'], 'thumb 은 handedness 로 반전되어야 함'
    print('  ✓ handedness 별 thumb 판정 반전 동작')


if __name__ == '__main__':
    print('hand_recognizer 오프라인 분류 테스트')
    test_open_palm()
    test_fist()
    test_victory()
    test_left_hand_thumb_mirrored()
    print('\n✅ 전체 통과')
