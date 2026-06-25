"""
serial_driver.py  (ROS-독립)
============================
적재함 잠금장치(서보/릴레이) 제어용 시리얼 래퍼.

설계 의도:
  - ROS에 의존하지 않음 → 단독 단위테스트 가능 (logger를 주입받음).
  - pyserial 미설치 / 포트 열기 실패 시 자동으로 mock 모드로 폴백 →
    하드웨어 없이도 노드를 그대로 띄워 명령 흐름을 검증할 수 있음.
  - 명령은 한 줄 텍스트 프로토콜: "UNLOCK\n" / "LOCK\n"
"""

from __future__ import annotations
from typing import Callable, Optional

try:
    import serial  # pyserial
except ImportError:
    serial = None


def _default_logger(level: str, msg: str) -> None:
    print(f'[{level.upper()}] serial_driver: {msg}')


class CargoSerial:
    """적재함 MCU와의 시리얼 통신. 실패 시 mock 모드로 동작."""

    def __init__(
        self,
        port:   str = '/dev/ttyUSB0',
        baud:   int = 115200,
        mock:   bool = False,
        logger: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._port = port
        self._baud = baud
        self._log  = logger or _default_logger
        self._ser  = None
        self._mock = bool(mock)

        if serial is None and not self._mock:
            self._log('warn', 'pyserial 미설치 → mock 모드로 전환')
            self._mock = True

        self._connect()

    # ── 연결 ────────────────────────────────────────────────────────────── #

    def _connect(self) -> None:
        if self._mock:
            self._log('info', f'mock 모드 (하드웨어 없음), port={self._port}')
            return
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=1.0)
            self._log('info', f'연결됨: {self._port}@{self._baud}')
        except Exception as e:  # noqa: BLE001 — 포트 미존재/권한 등 모두 mock 폴백
            self._log('warn', f'시리얼 열기 실패 ({e}) → mock 모드로 전환')
            self._mock = True
            self._ser  = None

    # ── 명령 전송 ──────────────────────────────────────────────────────── #

    def send(self, command: str) -> bool:
        """command(예: 'UNLOCK')에 개행 붙여 전송. 성공 시 True."""
        payload = (command.rstrip('\n') + '\n').encode()

        if self._mock or self._ser is None:
            self._log('info', f'[MOCK] 전송했다고 가정: {command!r}')
            return True

        try:
            self._ser.write(payload)
            self._log('info', f'전송: {command!r}')
            return True
        except Exception as e:  # noqa: BLE001
            self._log('error', f'시리얼 쓰기 실패: {e}')
            return False

    @property
    def is_mock(self) -> bool:
        return self._mock

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # noqa: BLE001
                pass
            self._ser = None
