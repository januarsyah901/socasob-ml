import datetime
import time
from collections import deque
from typing import Any, Dict, Optional

VALID_HARDWARE_COMMANDS = ("normal", "5", "10", "dry", "20")


class DailyHardwarePolicy:
    """Per-robot ML policy that emits the single final hardware command."""

    def __init__(self) -> None:
        self.robots: Dict[str, Dict[str, Any]] = {}

    def get_robot_state(self, robot_id: str) -> Dict[str, Any]:
        today = datetime.date.today()
        state = self.robots.get(robot_id)
        if state is None or state["date"] != today:
            state = {
                "date": today,
                "screen_duration_sec": 0.0,
                "continuous_gaze_sec": 0.0,
                "continuous_distance_below_50_sec": 0.0,
                "continuous_distance_below_20_sec": 0.0,
                "distance_below_20_cm_detected": False,
                "total_blinks": 0,
                "incomplete_blinks": 0,
                "blink_events": deque(),
                "fatigue_start_time": None,
                "last_update_time": None,
                "last_risk_status": "normal",
                "distance_cm": None,
            }
            self.robots[robot_id] = state
        return state

    def update(
        self,
        robot_id: str,
        face_detected: bool,
        distance_cm: Optional[float],
        blink_event: bool,
        incomplete_blink: bool,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Accumulate one observation and return the prioritized final command."""
        state = self.get_robot_state(robot_id)
        now = time.time() if now is None else now
        previous_time = state["last_update_time"]
        delta = 0.0 if previous_time is None else max(0.0, now - previous_time)
        state["last_update_time"] = now
        state["distance_cm"] = distance_cm

        if face_detected:
            state["screen_duration_sec"] += delta
            state["continuous_gaze_sec"] += delta
        else:
            state["continuous_gaze_sec"] = 0.0

        if face_detected and distance_cm is not None:
            if distance_cm < 50.0:
                state["continuous_distance_below_50_sec"] += delta
            else:
                state["continuous_distance_below_50_sec"] = 0.0
            if distance_cm < 20.0:
                state["continuous_distance_below_20_sec"] += delta
                state["distance_below_20_cm_detected"] = True
            else:
                state["continuous_distance_below_20_sec"] = 0.0
        else:
            state["continuous_distance_below_50_sec"] = 0.0
            state["continuous_distance_below_20_sec"] = 0.0

        if blink_event:
            state["total_blinks"] += 1
            state["incomplete_blinks"] += int(incomplete_blink)
            state["blink_events"].append((now, bool(incomplete_blink)))
        cutoff = now - 60.0
        while state["blink_events"] and state["blink_events"][0][0] < cutoff:
            state["blink_events"].popleft()

        return self._evaluate(state, now)

    def _evaluate(self, state: Dict[str, Any], now: float) -> Dict[str, Any]:
        screen_minutes = state["screen_duration_sec"] / 60.0
        fatigue_active = (
            screen_minutes > 360.0
            or state["continuous_distance_below_50_sec"] >= 10.0
        )
        if fatigue_active:
            if state["fatigue_start_time"] is None:
                state["fatigue_start_time"] = now
        else:
            state["fatigue_start_time"] = None

        events = list(state["blink_events"])
        total_window = len(events)
        incomplete_window = sum(1 for _, incomplete in events if incomplete)
        window_start = events[0][0] if events else now
        valid_window_seconds = min(60.0, max(0.0, now - window_start))
        blink_rate = (
            total_window / valid_window_seconds * 60.0
            if valid_window_seconds > 0 else 0.0
        )
        incomplete_ratio = (
            incomplete_window / total_window if total_window >= 5 else 0.0
        )
        dry_active = (
            screen_minutes > 360.0
            or (total_window >= 5 and incomplete_ratio >= 0.40)
            or (total_window >= 5 and valid_window_seconds >= 60.0 and blink_rate <= 10.0)
        )
        fatigue_escalated = (
            fatigue_active
            and state["fatigue_start_time"] is not None
            and now - state["fatigue_start_time"] >= 600.0
        )

        if state["continuous_gaze_sec"] > 1200.0:
            command = "20"
        elif dry_active:
            command = "dry"
        elif fatigue_escalated:
            command = "10"
        elif fatigue_active:
            command = "5"
        else:
            command = "normal"

        return self._build_result(
            state=state,
            command=command,
            screen_minutes=screen_minutes,
            blink_rate=blink_rate,
            incomplete_ratio=incomplete_ratio,
            fatigue_active=fatigue_active,
            dry_active=dry_active,
            now=now,
        )

    @staticmethod
    def _build_result(
        state: Dict[str, Any],
        command: str,
        screen_minutes: float,
        blink_rate: float,
        incomplete_ratio: float,
        fatigue_active: bool,
        dry_active: bool,
        now: float,
    ) -> Dict[str, Any]:
        previous = state["last_risk_status"]
        state["last_risk_status"] = command
        return {
            "hardware_command": command,
            "hardware": command,
            "lcd_command": {
                "normal": "normal",
                "5": "fatigue_5m",
                "10": "fatigue_10m",
                "dry": "dry_eye",
                "20": "break_20m",
            }[command],
            "screen_time_minutes": round(screen_minutes, 2),
            "continuous_gaze_minutes": round(state["continuous_gaze_sec"] / 60.0, 2),
            "distance_cm": state["distance_cm"],
            "close_distance_duration_seconds": round(state["continuous_distance_below_50_sec"], 2),
            "blink_rate_per_minute": round(blink_rate, 2),
            "incomplete_blink_count": state["incomplete_blinks"],
            "total_blink_observed": state["total_blinks"],
            "incomplete_blink_ratio": round(incomplete_ratio, 3),
            "fatigue_risk": fatigue_active,
            "dry_eye_risk": dry_active,
            "myopia_report_risk": (
                screen_minutes >= 240.0
                or state["distance_below_20_cm_detected"]
                or state["continuous_distance_below_20_sec"] > 1200.0
            ),
            "previous_hardware_command": previous,
            "hardware_command_changed": previous != command,
            "last_update_timestamp": now,
        }

    def reset(self, robot_id: Optional[str] = None) -> None:
        if robot_id:
            self.robots.pop(robot_id, None)
        else:
            self.robots.clear()
