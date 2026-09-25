from vision.blink_detector import BlinkEventDetector
from vision.metrics_window import MetricsWindow


def test_open_eye_noise_does_not_emit_repeated_blinks():
    detector = BlinkEventDetector()
    ear_values = [0.27, 0.24, 0.27, 0.24, 0.27, 0.24, 0.27] * 4

    events = [
        detector.update(value, face_confidence=1.0, timestamp=index / 15.0)
        for index, value in enumerate(ear_values)
    ]

    assert not any(events)


def test_three_to_five_closed_frames_emit_one_valid_blink():
    detector = BlinkEventDetector()
    ear_values = [0.27] * 5 + [0.20, 0.19, 0.20, 0.21] + [0.27] * 12

    events = [
        detector.update(value, face_confidence=1.0, timestamp=index / 15.0)
        for index, value in enumerate(ear_values)
    ]

    assert sum(event is not None for event in events) == 1


def test_invalid_landmark_frame_does_not_change_detector():
    detector = BlinkEventDetector()

    detector.update(0.19, face_confidence=0.0, timestamp=0.0)

    assert detector.state.value == "open"
    assert detector.smoother._value is None


def test_metrics_rate_and_incomplete_ratio_use_valid_events():
    metrics = MetricsWindow(window_seconds=60)
    for timestamp in range(61):
        metrics.add_frame(timestamp, is_closed=False, is_valid=True)

    for index in range(20):
        metrics.add_blink({
            "timestamp": 1.0 + index * 2.9,
            "duration": 0.2,
            "incomplete": index < 4,
        })

    assert metrics.smoothed_blink_rate() == 20.0
    assert metrics.incomplete_blink_ratio() == 0.2


def test_warmup_requires_sixty_seconds_and_five_valid_blinks():
    metrics = MetricsWindow(window_seconds=60)
    for timestamp in range(61):
        metrics.add_frame(timestamp, is_closed=False, is_valid=True)

    for index in range(4):
        metrics.add_blink({"timestamp": 10.0 + index, "duration": 0.2})
    assert not metrics.is_warmed_up(60.0)

    metrics.add_blink({"timestamp": 60.0, "duration": 0.2})
    assert metrics.is_warmed_up(60.0)