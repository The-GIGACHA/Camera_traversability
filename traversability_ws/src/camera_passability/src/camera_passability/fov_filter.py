from __future__ import annotations

import math
from typing import List

from .config import FOV_DEG, MAX_RANGE_M
from .tf_transformer import RobotPoint


def filter_fov(robot_points: List[RobotPoint]) -> List[RobotPoint]:
    """
    Step 4-b: base_link 기준 전방 부채꼴 + 최대 거리 필터.
    - base_link에서 전방 = +X 방향 가정
    """
    half_fov_rad = math.radians(FOV_DEG / 2.0)
    filtered: List[RobotPoint] = []

    for p in robot_points:
        dist = math.hypot(p.x, p.y)
        if dist > MAX_RANGE_M:
            continue
        if p.x <= 0:
            continue

        angle = math.atan2(p.y, p.x)
        if abs(angle) > half_fov_rad:
            continue

        filtered.append(p)

    filtered.sort(key=lambda p: math.hypot(p.x, p.y))
    return filtered

