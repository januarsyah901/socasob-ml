import cv2
import numpy as np

from camera.esp32_camera import decode_websocket_packet
from cv.eye_landmarks import is_looking_at_screen
from realtime.daily_hardware_policy import DailyHardwarePolicy
from services.robot_trigger_service import RobotTriggerService


def jpeg_bytes():
    ok, encoded = cv2.imencode(".jpg", np.zeros((3, 3, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def packet(robot_id, is_dekat, jpeg, distance_mm=None):
    header = bytes([len(robot_id)]) + robot_id.encode() + bytes([is_dekat])
    if distance_mm is not None:
        header += bytes([0xA5]) + distance_mm.to_bytes(2, "big")
    return header + jpeg


def test_decode_v1_has_no_numeric_distance():
    decoded = decode_websocket_packet(packet("r1", 1, jpeg_bytes()))
    assert decoded is not None
    assert decoded.robot_id == "r1"
    assert decoded.is_dekat is True
    assert decoded.distance_cm is None


def test_decode_v2_distance_and_jpeg_offset():
    decoded = decode_websocket_packet(packet("r1", 1, jpeg_bytes(), 1234))
    assert decoded is not None
    assert decoded.distance_cm == 123.4
    assert decoded.frame.shape == (3, 3, 3)


def test_decode_v2_invalid_distance_is_none():
    decoded = decode_websocket_packet(packet("r1", 0, jpeg_bytes(), 0xFFFF))
    assert decoded is not None
    assert decoded.distance_cm is None


def test_near_distance_for_ten_seconds_is_fatigue_5():
    policy = DailyHardwarePolicy()
    policy.update("r1", True, 40.0, False, False, now=0.0)
    result = policy.update("r1", True, 40.0, False, False, now=10.0)
    assert result["hardware_command"] == "5"
    assert result["close_distance_duration_seconds"] == 10.0


def test_fatigue_persists_ten_minutes_as_10():
    policy = DailyHardwarePolicy()
    policy.update("r1", True, 40.0, False, False, now=0.0)
    policy.update("r1", True, 40.0, False, False, now=10.0)
    result = policy.update("r1", True, 40.0, False, False, now=610.0)
    assert result["hardware_command"] == "10"


def test_continuous_gaze_over_twenty_minutes_is_20():
    policy = DailyHardwarePolicy()
    policy.update("r1", True, 100.0, False, False, now=0.0)
    result = policy.update("r1", True, 100.0, False, False, now=1201.0)
    assert result["hardware_command"] == "20"


def test_break_requires_twenty_seconds_without_face():
    policy = DailyHardwarePolicy()
    policy.update("r1", True, 100.0, False, False, now=0.0)
    result = policy.update("r1", True, 100.0, False, False, now=1201.0)
    assert result["hardware_command"] == "20"

    still_facing = policy.update("r1", True, 100.0, False, False, now=1210.0)
    assert still_facing["hardware_command"] == "20"
    assert still_facing["break_remaining_sec"] == 20.0

    incomplete_break = policy.update("r1", False, None, False, False, now=1215.0)
    assert incomplete_break["hardware_command"] == "20"
    assert incomplete_break["break_remaining_sec"] == 20.0

    completed_break = policy.update("r1", False, None, False, False, now=1235.0)
    assert completed_break["hardware_command"] == "normal"
    assert completed_break["break_remaining_sec"] == 0.0


def test_break_counts_gaze_away_and_restarts_when_looking_at_screen():
    policy = DailyHardwarePolicy()
    policy.update("r1", True, 100.0, False, False, now=0.0)
    policy.update("r1", True, 100.0, False, False, now=1200.0)

    away = policy.update(
        "r1", True, 100.0, False, False, looking_at_screen=False, now=1201.0
    )
    assert away["break_remaining_sec"] == 20.0

    halfway = policy.update(
        "r1", True, 100.0, False, False, looking_at_screen=False, now=1211.0
    )
    assert halfway["break_remaining_sec"] == 10.0

    interrupted = policy.update(
        "r1", True, 100.0, False, False, looking_at_screen=True, now=1212.0
    )
    assert interrupted["hardware_command"] == "20"
    assert interrupted["break_remaining_sec"] == 20.0

    restarted = policy.update(
        "r1", True, 100.0, False, False, looking_at_screen=False, now=1213.0
    )
    assert restarted["break_remaining_sec"] == 20.0


def test_gaze_classifier_uses_iris_position():
    landmarks = [(0.0, 0.0)] * 478
    landmarks[33] = (0.3, 0.5)
    landmarks[133] = (0.4, 0.5)
    landmarks[159] = (0.35, 0.48)
    landmarks[145] = (0.35, 0.52)
    landmarks[468] = (0.35, 0.5)
    landmarks[362] = (0.6, 0.5)
    landmarks[263] = (0.7, 0.5)
    landmarks[386] = (0.65, 0.48)
    landmarks[374] = (0.65, 0.52)
    landmarks[473] = (0.65, 0.5)

    assert is_looking_at_screen(landmarks) is True
    landmarks[468] = (0.302, 0.5)
    assert is_looking_at_screen(landmarks) is False
    assert is_looking_at_screen(landmarks[:468]) is None


def test_dry_eye_wins_over_fatigue():
    policy = DailyHardwarePolicy()
    policy.update("r1", True, 40.0, False, False, now=0.0)
    for index in range(5):
        result = policy.update("r1", True, 40.0, True, True, now=1.0 + index)
    assert result["hardware_command"] == "dry"


def test_twenty_wins_over_dry_and_fatigue():
    policy = DailyHardwarePolicy()
    policy.update("r1", True, 40.0, False, False, now=0.0)
    for index in range(5):
        policy.update("r1", True, 40.0, True, True, now=1.0 + index)
    result = policy.update("r1", True, 40.0, False, False, now=1201.0)
    assert result["hardware_command"] == "20"


def test_myopia_is_report_only():
    policy = DailyHardwarePolicy()
    policy.update("r1", True, 15.0, False, False, now=0.0)
    result = policy.update("r1", True, 15.0, False, False, now=241 * 60.0)
    assert result["myopia_report_risk"] is True
    assert result["hardware_command"] in {"normal", "5", "10", "dry", "20"}
    assert result["hardware_command"] != "myopia"


def test_trigger_service_deduplicates_official_text_commands():
    service = RobotTriggerService()
    service.send_trigger("r1", "5")
    service.send_trigger("r1", "5")
    assert service.get_last_trigger("r1") == "5"
    assert service.send_trigger("r1", "invalid") is False
    assert service.get_last_trigger("r1") == "normal"
