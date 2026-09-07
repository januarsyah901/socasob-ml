# services/aggregator_service.py
#
# Modul ini mengakumulasi data hasil analisis per-frame selama 60 detik,
# lalu menghasilkan satu payload ringkasan (summary) yang dikirim ke Backend
# via Channel B (py-minute-summary).
#
# Data yang diakumulasi per menit:
#   - Durasi "Dekat" dan "Jauh" dalam detik
#   - Total kedipan & rata-rata blink rate
#   - Rata-rata PERCLOS & Fatigue Score komposit
#   - Health status & kondisi mata dari EyeConditionAnalyzer

import time
import threading
from typing import Callable, Optional
from utils.time_utils import get_current_iso_time
from utils.logger import get_logger

logger = get_logger(__name__)

# Durasi satu window agregasi (detik)
AGGREGATION_WINDOW_SEC = 60


class AggregatorService:
    """
    Mengakumulasi data deteksi per-frame selama satu menit,
    lalu memanggil callback dengan payload ringkasan untuk dikirim ke BE.
    """

    def __init__(self, on_summary: Callable[[dict], None]):
        """
        Args:
            on_summary: Callback yang dipanggil setiap 1 menit dengan payload summary.
                        Biasanya ini memanggil be_socket_client.emit_minute_summary().
        """
        self._on_summary = on_summary
        self._lock = threading.Lock()
        # Multi-robot: satu bucket akumulasi per robot_id
        self._buckets: dict = {}
        self._period_start: str = get_current_iso_time()

        self._thread: threading.Thread | None = None
        self._running = False

    def _new_bucket(self) -> dict:
        return {
            "near_sec": 0.0,
            "far_sec": 0.0,
            "blink_count": 0,
            "blink_rate_samples": [],
            "perclos_samples": [],
            "composite_score_samples": [],
            "health_statuses": [],
            "eye_conditions": [],
            "recommendations": [],
            "last_frame_time": time.time(),
        }

    def _reset_state(self) -> None:
        """Reset semua bucket (kompatibilitas API lama)."""
        self._buckets = {}
        self._period_start = get_current_iso_time()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mulai background thread yang menjalankan timer 1 menit."""
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
        """Hentikan background thread aggregator."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("[Aggregator] Service dihentikan.")

    def _aggregation_loop(self) -> None:
        """Loop yang menunggu 60 detik lalu memanggil _flush_summary."""
        while self._running:
            time.sleep(AGGREGATION_WINDOW_SEC)
            if self._running:
                self._flush_summary()

    # ------------------------------------------------------------------
    # Data Ingestion (dipanggil tiap frame dari VisionPipelineService)
    # ------------------------------------------------------------------

    def ingest(self, robot_id: str, distance: str, blink_event: bool,
               blink_rate: float, health_status: str,
               eye_conditions: list[str], recommendations: list[str],
               perclos: float = 0.0, composite_score: float = 0.0) -> None:
        """
        Terima data satu frame untuk diakumulasi.
        """
        current_time = time.time()

        with self._lock:
            bucket = self._buckets.get(robot_id)
            if bucket is None:
                bucket = self._new_bucket()
                self._buckets[robot_id] = bucket

            # Hitung delta waktu antar frame untuk akurasi durasi
            delta = current_time - bucket["last_frame_time"]
            # Batasi delta maksimal 1 detik untuk mencegah jump jika lag
            delta = min(delta, 1.0)
            bucket["last_frame_time"] = current_time

            # Akumulasi durasi berdasarkan status jarak
            if distance == "Dekat":
                bucket["near_sec"] += delta
            else:
                bucket["far_sec"] += delta

            # Akumulasi blink
            if blink_event:
                bucket["blink_count"] += 1

            # Simpan sample blink rate & komposit
            if blink_rate > 0:
                bucket["blink_rate_samples"].append(blink_rate)
            bucket["perclos_samples"].append(perclos)
            bucket["composite_score_samples"].append(composite_score)

            # Simpan health status dan kondisi
            bucket["health_statuses"].append(health_status)
            bucket["eye_conditions"].extend(eye_conditions)
            bucket["recommendations"].extend(recommendations)

    # ------------------------------------------------------------------
    # Summary Flush
    # ------------------------------------------------------------------

    def _flush_summary(self) -> None:
        """
        Hitung payload ringkasan per robot dari data yang terakumulasi,
        panggil callback untuk tiap robot, lalu reset bucket untuk window berikutnya.
        """
        with self._lock:
            if not self._buckets:
                logger.info("[Aggregator] Tidak ada data dalam window ini. Summary dilewati.")
                self._period_start = get_current_iso_time()
                return
            items = list(self._buckets.items())
            self._buckets = {}
            period_start = self._period_start
            self._period_start = get_current_iso_time()

        period_end = get_current_iso_time()
        for robot_id, b in items:
            near_sec = round(b["near_sec"])
            far_sec = round(b["far_sec"])
            total_sec = near_sec + far_sec
            blink_count = b["blink_count"]
            avg_blink_rate = round(
                sum(b["blink_rate_samples"]) / len(b["blink_rate_samples"]), 2
            ) if b["blink_rate_samples"] else 0.0

            avg_perclos = round(
                sum(b["perclos_samples"]) / len(b["perclos_samples"]), 3
            ) if b["perclos_samples"] else 0.0

            avg_composite_score = round(
                sum(b["composite_score_samples"]) / len(b["composite_score_samples"]), 1
            ) if b["composite_score_samples"] else 0.0

            dominant_distance = "Dekat" if near_sec >= far_sec else "Jauh"

            # Ambil health_status yang paling sering muncul
            health_status = "Aman"
            if b["health_statuses"]:
                health_status = max(
                    set(b["health_statuses"]),
                    key=b["health_statuses"].count
                )

            # Deduplicate kondisi dan rekomendasi
            eye_conditions = list(dict.fromkeys(b["eye_conditions"]))
            recommendations = list(dict.fromkeys(b["recommendations"]))

            near_percentage = round((near_sec / total_sec) * 100, 1) if total_sec > 0 else 0.0

            summary = {
                "robot_id": robot_id,
                "period_start": period_start,
                "period_end": period_end,
                "near_duration_sec": near_sec,
                "far_duration_sec": far_sec,
                "near_percentage": near_percentage,
                "blink_count": blink_count,
                "avg_blink_rate": avg_blink_rate,
                "avg_perclos": avg_perclos,
                "avg_fatigue_score": avg_composite_score,
                "dominant_distance": dominant_distance,
                "health_status": health_status,
                "eye_conditions": eye_conditions,
                "recommendations": recommendations
            }

            logger.info(
                f"[Aggregator] Summary robot={robot_id} | "
                f"Dekat={near_sec}s | Jauh={far_sec}s | Blink={blink_count} | FatigueScore={avg_composite_score}"
            )

            # Panggil callback (emit ke BE)
            try:
                self._on_summary(summary)
            except Exception as e:
                logger.error(f"[Aggregator] Error saat memanggil on_summary callback: {e}")
