#!/usr/bin/env python3
"""
cargo_actuator_node.py  (ROS1)
==============================
/cargo_unlock(Bool) 구독 → 시리얼로 적재함 잠금장치(서보/릴레이) 제어.

흐름:
  vision_auth_node 가 UNLOCK_PUB_COUNT(=3)회 True 를 발행하므로,
  DEBOUNCE_SEC 내 중복 True 는 1회 개방으로 합침.
  ~auto_relock_sec > 0 이면 개방 후 해당 시간 뒤 자동 재잠금.

파라미터:
  ~port            (str)   시리얼 포트              기본 /dev/ttyUSB0
  ~baud            (int)   보드레이트              기본 115200
  ~mock            (bool)  하드웨어 없이 로그만      기본 False
  ~unlock_cmd      (str)   개방 명령 문자열          기본 "UNLOCK"
  ~lock_cmd        (str)   잠금 명령 문자열          기본 "LOCK"
  ~auto_relock_sec (float) 자동 재잠금 지연(0=off)  기본 0.0
"""

import rospy
from std_msgs.msg import Bool

from campus_delivery_auth.serial_driver import CargoSerial


DEBOUNCE_SEC = 5.0


class CargoActuatorNode:

    def __init__(self) -> None:
        rospy.init_node('cargo_actuator_node', anonymous=False)

        port = rospy.get_param('~port', '/dev/ttyUSB0')
        baud = int(rospy.get_param('~baud', 115200))
        mock = bool(rospy.get_param('~mock', False))
        self._unlock_cmd  = rospy.get_param('~unlock_cmd', 'UNLOCK')
        self._lock_cmd    = rospy.get_param('~lock_cmd', 'LOCK')
        self._auto_relock = float(rospy.get_param('~auto_relock_sec', 0.0))

        self._serial = CargoSerial(
            port=port, baud=baud, mock=mock, logger=self._ros_log
        )

        self._last_unlock_t = 0.0
        self._relock_timer  = None

        rospy.Subscriber('/cargo_unlock', Bool, self._on_unlock, queue_size=10)

        rospy.loginfo(
            f'cargo_actuator_node ready '
            f'(mock={self._serial.is_mock}, auto_relock={self._auto_relock}s)'
        )

    # ── 로깅 어댑터: serial_driver(level, msg) → rospy ──────────────────── #

    @staticmethod
    def _ros_log(level: str, msg: str) -> None:
        fn = {'info': rospy.loginfo, 'warn': rospy.logwarn,
              'error': rospy.logerr}.get(level, rospy.loginfo)
        fn(f'[serial] {msg}')

    # ── 콜백 ────────────────────────────────────────────────────────────── #

    def _on_unlock(self, msg: Bool) -> None:
        if not msg.data:
            return

        now = rospy.get_time()
        if now - self._last_unlock_t < DEBOUNCE_SEC:
            rospy.logdebug('unlock debounced (중복 명령 무시)')
            return
        self._last_unlock_t = now

        rospy.loginfo('UNLOCK 수신 → 적재함 개방')
        self._serial.send(self._unlock_cmd)

        if self._auto_relock > 0.0:
            if self._relock_timer is not None:
                self._relock_timer.shutdown()
            self._relock_timer = rospy.Timer(
                rospy.Duration(self._auto_relock), self._on_relock, oneshot=True
            )

    def _on_relock(self, event=None) -> None:
        rospy.loginfo(f'auto relock ({self._auto_relock}s 경과) → 적재함 잠금')
        self._serial.send(self._lock_cmd)

    # ─────────────────────────────────────────────────────────────────────── #

    def spin(self) -> None:
        rospy.spin()
        self._serial.close()


if __name__ == '__main__':
    CargoActuatorNode().spin()
