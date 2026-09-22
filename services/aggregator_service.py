# services/aggregator_service.py

import time
import threading
from typing import Callable, Optional, Dict, Any
from utils.time_utils import get_current_iso_time
from utils.logger import get_logger

logger = get_logger(__name__)

AGGREGATION_WINDOW_SEC = 60

class AggregatorService:
    def __init__(self, on_summary: Callable[[dict], None], trigger_service=None):
        self._on_summary = on_summary
        self._trigger_service = trigger_service
        self._lock = threading.Lock()
        
        # State per robot
        self._robots: Dict[str, Dict[str, Any]] = {}

        self._thread: threading.Thread | None = None
        self._running = False

    def _get_robot_state(self, robot_id: str) -> Dict[str, Any]:
        if robot_id not in self._robots:
            self._robots[robot_id] = {
                "period_start": get_current_iso_time(),
                "screen_duration_sec": 0.0,
                "blink_count": 0,
                "incomplete_blink_count": 0,
                
                "continuous_distance_below_50_sec": 0.0,
                "distance_below_50_cm_for_at_least_10_seconds": 0,
                
                "distance_below_20_cm_detected": False,
                
                "current_continuous_gaze_sec": 0.0,
                "longest_continuous_gaze_sec": 0.0,
                
                "distance_sum_cm": 0.0,
                "distance_count": 0,
                "policy_summary": {},
                
                "last_frame_time": time.time()
            }
        return self._robots[robot_id]

    def reset(self) -> None:
        with self._lock:
            self._robots.clear()
        logger.info("[Aggregator] State counter berhasil direset.")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True,
            name="aggregator-service"
        )
        self._thread.start()
        logger.info("[Aggregator] Service dimulai. Window = 60 detik.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("[Aggregator] Service dihentikan.")

    def _aggregation_loop(self) -> None:
        while self._running:
            time.sleep(AGGREGATION_WINDOW_SEC)
            if self._running:
                self._flush_summary()

    def ingest(self, 
               robot_id: str, 
               face_detected: bool,
               distance_cm: Optional[float], 
               blink_event: bool, 
               incomplete_blink: bool,
               policy_summary: Optional[Dict[str, Any]] = None,
               **kwargs) -> None:
        """
        Terima data satu frame untuk diakumulasi per robot.
        """
        current_time = time.time()

        with self._lock:
            state = self._get_robot_state(robot_id)
            if policy_summary:
                state["policy_summary"] = {
                    key: policy_summary[key]
                    for key in (
                        "screen_time_minutes", "continuous_gaze_minutes", "distance_cm",
                        "close_distance_duration_seconds", "blink_rate_per_minute",
                        "incomplete_blink_count", "total_blink_observed",
                        "incomplete_blink_ratio", "fatigue_risk", "dry_eye_risk",
                        "myopia_report_risk", "hardware_command",
                    )
                    if key in policy_summary
                }

            delta = current_time - state["last_frame_time"]
            delta = min(delta, 1.0)
            state["last_frame_time"] = current_time

            if face_detected:
                state["screen_duration_sec"] += delta
                state["current_continuous_gaze_sec"] += delta
                
                if state["current_continuous_gaze_sec"] > state["longest_continuous_gaze_sec"]:
                    state["longest_continuous_gaze_sec"] = state["current_continuous_gaze_sec"]
                
                if distance_cm is not None:
                    state["distance_sum_cm"] += distance_cm
                    state["distance_count"] += 1
                    
                    if distance_cm < 20.0:
                        state["distance_below_20_cm_detected"] = True
                        
                    if distance_cm < 50.0:
                        state["continuous_distance_below_50_sec"] += delta
                        if state["continuous_distance_below_50_sec"] >= 10.0:
                            state["distance_below_50_cm_for_at_least_10_seconds"] = 1
                    else:
                        state["continuous_distance_below_50_sec"] = 0.0
            else:
                state["current_continuous_gaze_sec"] = 0.0
                state["continuous_distance_below_50_sec"] = 0.0

            if blink_event:
                state["blink_count"] += 1
                if incomplete_blink:
                    state["incomplete_blink_count"] += 1


    def _flush_summary(self) -> None:
        """
        Hitung payload ringkasan dari data yang terakumulasi,
        panggil callback, lalu reset state untuk window berikutnya.
        """
        with self._lock:
            robots_to_flush = dict(self._robots)
            self._robots.clear()

        period_end = get_current_iso_time()

        for robot_id, state in robots_to_flush.items():
            avg_distance_cm = 0.0
            if state["distance_count"] > 0:
                avg_distance_cm = round(state["distance_sum_cm"] / state["distance_count"], 1)

            longest_continuous_gaze_minutes = round(state["longest_continuous_gaze_sec"] / 60.0, 2)

            summary = {
                "robot_id": robot_id,
                "period_start": state["period_start"],
                "period_end": period_end,
                "screen_duration_sec": round(state["screen_duration_sec"]),
                "blink_count": state["blink_count"],
                "incomplete_blink_count": state["incomplete_blink_count"],
                "distance_below_50_cm_for_at_least_10_seconds": state["distance_below_50_cm_for_at_least_10_seconds"],
                "distance_below_20_cm_detected": state["distance_below_20_cm_detected"],
                "longest_continuous_gaze_minutes": longest_continuous_gaze_minutes,
                "avg_distance_cm": avg_distance_cm
            }
            summary.update(state["policy_summary"])

            logger.info(
                f"[Aggregator] Summary robot={robot_id} | "
                f"Screen={summary['screen_duration_sec']}s | Blink={summary['blink_count']} | IncompBlink={summary['incomplete_blink_count']}"
            )

            try:
                self._on_summary(summary)
            except Exception as e:
                logger.error(f"[Aggregator] Error saat memanggil on_summary callback: {e}")
