# services/feature_store.py

import time
import threading
from typing import Dict, Any, Optional, List


class FeatureStore:
    """
    Cache in-memory yang thread-safe untuk menyimpan data fitur ekstraksi terbaru.
    
    Menyimpan fitur global terbaru serta data per-robot untuk dashboard multi-robot.
    """

    def __init__(self):
        """Inisialisasi penyimpan fitur dalam keadaan kosong."""
        self.lock = threading.Lock()
        self._features: Optional[Dict[str, Any]] = None
        self._by_robot: Dict[str, Dict[str, Any]] = {}
        self._last_seen: Dict[str, float] = {}

    def update(self, new_features: Dict[str, Any]) -> None:
        """
        Memperbarui data fitur dengan hasil ekstraksi terbaru secara aman.
        """
        with self.lock:
            self._features = new_features.copy()
            robot_id = new_features.get("robot_id")
            if robot_id:
                self._by_robot[robot_id] = new_features.copy()
                self._last_seen[robot_id] = time.time()

    def get(self, robot_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Mengambil salinan data fitur terbaru untuk dikirim sebagai JSON API.
        Jika robot_id dispesifikasikan, ambil data robot tersebut.
        """
        with self.lock:
            if robot_id:
                data = self._by_robot.get(robot_id)
                return data.copy() if data else None
            if self._features is None:
                return None
            return self._features.copy()

    def list_robots(self) -> List[Dict[str, Any]]:
        """
        Mengembalikan daftar semua robot yang tercatat di store beserta status aktifnya.
        """
        now = time.time()
        result = []
        with self.lock:
            for rid, last_ts in self._last_seen.items():
                is_active = (now - last_ts) < 30.0
                result.append({
                    "robot_id": rid,
                    "last_seen": last_ts,
                    "is_active": is_active,
                })
        return result

    def update_hardware_trigger(
        self,
        trigger: str,
        robot_id: Optional[str] = None,
        lcd_cmd: Optional[str] = None,
        speaker_cmd: Optional[str] = None,
        lcd_label: Optional[str] = None,
        speaker_label: Optional[str] = None,
    ) -> None:
        """
        Memperbarui trigger manual hardware pada cache feature store.
        """
        now = time.time()
        with self.lock:
            targets = []
            if robot_id and robot_id in self._by_robot:
                targets.append(self._by_robot[robot_id])
            if self._features:
                targets.append(self._features)

            for feat in targets:
                feat["robot_trigger"] = trigger
                hw = feat.setdefault("hardware", {})
                if lcd_cmd is not None:
                    hw["lcd_command"] = lcd_cmd
                if speaker_cmd is not None:
                    hw["speaker_command"] = speaker_cmd
                if lcd_label is not None:
                    hw["lcd_label"] = lcd_label
                if speaker_label is not None:
                    hw["speaker_label"] = speaker_label
                hw["robot_trigger"] = trigger
                hw["timestamp"] = now

    def has_data(self) -> bool:
        """Mengecek apakah sudah ada data fitur yang masuk."""
        with self.lock:
            return self._features is not None

    def reset(self, robot_id: Optional[str] = None) -> None:
        """Reset data fitur dan cache robot ke kondisi awal."""
        with self.lock:
            if robot_id:
                self._by_robot.pop(robot_id, None)
                self._last_seen.pop(robot_id, None)
                if self._features and self._features.get("robot_id") == robot_id:
                    self._features = None
            else:
                self._features = None
                self._by_robot.clear()
                self._last_seen.clear()