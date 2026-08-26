# storage/database.py
"""
database.py — Penyimpanan riwayat deteksi ke SQLite.

Tiga tabel terpisah per jenis deteksi:
1. fatigue_logs   — timestamp, composite_score, status, data_quality
2. dry_eye_logs   — timestamp, perclos, avg_blink_duration, status
3. myopia_risk_logs — timestamp, screen_time_today_minutes,
                      risk_percentage, distance_warning_count,
                      break_reminder_count

Skema dirancang portable — bisa dimigrasi ke database lain
(PostgreSQL, MySQL) dengan perubahan minimal pada koneksi.

Menyediakan fungsi query "ambil riwayat N hari terakhir" per tabel
untuk laporan historis via WebSocket.
"""

import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Default path database — di direktori project
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "eye_health.db",
)

# ─────────────────────────────────────────────
# Schema SQL
# ─────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fatigue_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    composite_score REAL    NOT NULL,
    status          TEXT    NOT NULL,
    data_quality    REAL    NOT NULL,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dry_eye_logs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               REAL    NOT NULL,
    perclos                 REAL    NOT NULL,
    avg_blink_duration      REAL    NOT NULL,
    incomplete_blink_ratio  REAL    NOT NULL DEFAULT 0.0,
    blink_rate              REAL    NOT NULL DEFAULT 0.0,
    status                  TEXT    NOT NULL,
    created_at              TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS myopia_risk_logs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               REAL    NOT NULL,
    screen_time_today_min   REAL    NOT NULL,
    risk_percentage         REAL    NOT NULL,
    distance_warning_count  INTEGER NOT NULL DEFAULT 0,
    break_reminder_count    INTEGER NOT NULL DEFAULT 0,
    created_at              TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fatigue_ts ON fatigue_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_dry_eye_ts ON dry_eye_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_myopia_ts  ON myopia_risk_logs(timestamp);
"""


class HealthDatabase:
    """
    Database SQLite untuk riwayat deteksi kesehatan mata.

    Thread-safe: menggunakan lock internal karena SQLite
    tidak mendukung concurrent writes dari banyak thread.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or DEFAULT_DB_PATH
        self._lock = threading.Lock()

        # Pastikan direktori ada
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        # Inisialisasi schema
        self._init_schema()
        logger.info(f"[Database] Terhubung ke SQLite: {self._db_path}")

    def _init_schema(self) -> None:
        """Buat tabel jika belum ada."""
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _get_conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager untuk koneksi database thread-safe."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ─────────────────────────────────────────
    # Insert methods
    # ─────────────────────────────────────────

    def log_fatigue(self, result: Dict[str, Any]) -> None:
        """Simpan hasil deteksi fatigue."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO fatigue_logs
                       (timestamp, composite_score, status, data_quality)
                       VALUES (?, ?, ?, ?)""",
                    (
                        result.get("timestamp", time.time()),
                        result.get("composite_score", 0.0),
                        result.get("status", "Unknown"),
                        result.get("data_quality", 0.0),
                    ),
                )

    def log_dry_eye(self, result: Dict[str, Any]) -> None:
        """Simpan hasil deteksi dry eye."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO dry_eye_logs
                       (timestamp, perclos, avg_blink_duration,
                        incomplete_blink_ratio, blink_rate, status)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        result.get("timestamp", time.time()),
                        result.get("perclos", 0.0),
                        result.get("avg_blink_duration", 0.0),
                        result.get("incomplete_blink_ratio", 0.0),
                        result.get("blink_rate", 0.0),
                        result.get("status", "Unknown"),
                    ),
                )

    def log_myopia_risk(self, result: Dict[str, Any]) -> None:
        """Simpan hasil estimasi risiko miopia."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO myopia_risk_logs
                       (timestamp, screen_time_today_min, risk_percentage,
                        distance_warning_count, break_reminder_count)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        result.get("timestamp", time.time()),
                        result.get("screen_time_minutes", 0.0),
                        result.get("risk_percentage", 0.0),
                        result.get("distance_warning_count", 0),
                        result.get("break_reminder_count", 0),
                    ),
                )

    def log_all(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Simpan hasil semua detector sekaligus (dari InferenceEngine.run())."""
        if "fatigue" in results:
            self.log_fatigue(results["fatigue"])
        if "dry_eye" in results:
            self.log_dry_eye(results["dry_eye"])
        if "myopia_risk" in results:
            self.log_myopia_risk(results["myopia_risk"])

    # ─────────────────────────────────────────
    # Query methods — riwayat N hari terakhir
    # ─────────────────────────────────────────

    def get_fatigue_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Ambil riwayat fatigue N hari terakhir."""
        return self._query_history("fatigue_logs", days)

    def get_dry_eye_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Ambil riwayat dry eye N hari terakhir."""
        return self._query_history("dry_eye_logs", days)

    def get_myopia_risk_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Ambil riwayat risiko miopia N hari terakhir."""
        return self._query_history("myopia_risk_logs", days)

    def _query_history(
        self, table: str, days: int
    ) -> List[Dict[str, Any]]:
        """Query generik untuk mengambil riwayat N hari terakhir."""
        cutoff = time.time() - (days * 86400)

        # Whitelist nama tabel untuk mencegah SQL injection
        allowed = {"fatigue_logs", "dry_eye_logs", "myopia_risk_logs"}
        if table not in allowed:
            raise ValueError(f"Tabel tidak dikenal: {table}")

        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    f"SELECT * FROM {table} WHERE timestamp >= ? ORDER BY timestamp DESC",
                    (cutoff,),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def get_summary(self, days: int = 1) -> Dict[str, Any]:
        """
        Ringkasan statistik untuk dashboard — data N hari terakhir.

        Returns:
            Dict dengan summary per jenis deteksi.
        """
        cutoff = time.time() - (days * 86400)

        with self._lock:
            with self._get_conn() as conn:
                # Fatigue summary
                fat = conn.execute(
                    """SELECT COUNT(*) as total,
                              AVG(composite_score) as avg_score,
                              MAX(composite_score) as max_score
                       FROM fatigue_logs WHERE timestamp >= ?""",
                    (cutoff,),
                ).fetchone()

                # Dry eye summary
                dry = conn.execute(
                    """SELECT COUNT(*) as total,
                              AVG(perclos) as avg_perclos,
                              AVG(incomplete_blink_ratio) as avg_incomplete
                       FROM dry_eye_logs WHERE timestamp >= ?""",
                    (cutoff,),
                ).fetchone()

                # Myopia risk summary
                myo = conn.execute(
                    """SELECT COUNT(*) as total,
                              MAX(screen_time_today_min) as max_screen_time,
                              MAX(risk_percentage) as max_risk,
                              SUM(distance_warning_count) as total_distance_warns,
                              SUM(break_reminder_count) as total_break_reminders
                       FROM myopia_risk_logs WHERE timestamp >= ?""",
                    (cutoff,),
                ).fetchone()

        return {
            "period_days": days,
            "fatigue": dict(fat) if fat else {},
            "dry_eye": dict(dry) if dry else {},
            "myopia_risk": dict(myo) if myo else {},
        }
