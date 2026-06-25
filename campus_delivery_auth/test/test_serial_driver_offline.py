#!/usr/bin/env python3
"""
test_serial_driver_offline.py
=============================
ROS 없이 serial_driver 로직만 검증하는 오프라인 스모크 테스트.
실행:  python3 test/test_serial_driver_offline.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from campus_delivery_auth.serial_driver import CargoSerial  # noqa: E402


def _collect_logger(sink):
    return lambda level, msg: sink.append((level, msg))


def test_explicit_mock_sends_ok():
    logs = []
    s = CargoSerial(mock=True, logger=_collect_logger(logs))
    assert s.is_mock is True
    assert s.send('UNLOCK') is True
    assert s.send('LOCK') is True
    assert any('UNLOCK' in m for _, m in logs)
    print('  ✓ mock 모드: send 성공 + 로그 기록')


def test_bad_port_falls_back_to_mock():
    logs = []
    s = CargoSerial(port='/dev/definitely_not_a_real_port',
                    mock=False, logger=_collect_logger(logs))
    # pyserial 미설치 또는 포트 열기 실패 → 둘 다 mock 폴백이어야 함
    assert s.is_mock is True, 'bad 포트는 mock으로 폴백해야 함'
    assert s.send('UNLOCK') is True
    print('  ✓ 잘못된 포트: mock 모드로 안전 폴백')


def test_unlock_burst_idempotent_payload():
    # vision_auth 가 3회 발행하는 상황 모사 — 드라이버는 매번 정상 처리
    logs = []
    s = CargoSerial(mock=True, logger=_collect_logger(logs))
    assert all(s.send('UNLOCK') for _ in range(3))
    sent = [m for lvl, m in logs if 'UNLOCK' in m]
    assert len(sent) == 3
    print('  ✓ UNLOCK 3회 연속 전송 정상 (디바운스는 노드 책임)')


if __name__ == '__main__':
    print('serial_driver 오프라인 테스트')
    test_explicit_mock_sends_ok()
    test_bad_port_falls_back_to_mock()
    test_unlock_burst_idempotent_payload()
    print('\n✅ 전체 통과')
