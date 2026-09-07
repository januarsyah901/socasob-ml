# services/feature_store.py

import threading
import time
from typing import Dict, Any, Optional


class FeatureStore:
    """
    Cache in-memory yang thread-safe untuk menyimpan data fitur ekstraksi terbaru.

    Multi-robot: menyimpan satu payload per robot_id sehingga dashboard dengan
    tab per-robot tidak saling menimpa. API lama tanpa robot_id tetap kompatibel
    (mengembalikan robot yang terakhir update).

    Jika belum ada robot yang connect dan mengirim frame, `get()` mengembalikan None
    sehingga API tidak menampilkan data palsu/dummy.
    """

    def __init__(self):
        """Inisialisasi penyimpan fitur dalam keadaan kosong."""
        self.lock = threading.Lock()

        # {robot_id: {"features": dict, "updated_at": float}}
        self._stores: Dict[str, Dict[str, Any]] = {}
        self._last_robot: Optional[str] = None

        # Kompatibilitas lama: single-slot view (robot terakhir)
        self._features: Optional[Dict[str, Any]] = None

    def update(self, new_features: Dict[str, Any]) -> None:
        """
        Memperbarui data fitur dengan hasil ekstraksi terbaru secara aman.

        Args:
            new_features (Dict[str, Any]): Dictionary fitur terbaru dari pipeline.
                Harus mengandung `robot_id`; jika tidak ada, disimpan di slot legacy.
        """
        with self.lock:
            # Menyimpan salinan agar tidak terjadi mutasi referensi dari luar
            snapshot = new_features.copy()
            robot_id = snapshot.get("robot_id")
            if robot_id:
                self._stores[robot_id] = {
                    "features": snapshot,
                    "updated_at": time.time(),
                }
                self._last_robot = robot_id
            self._features = snapshot

    def get(self, robot_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Mengambil salinan data fitur terbaru untuk dikirim sebagai JSON API.

        Args:
            robot_id: bila diisi, kembalikan fitur robot tersebut (None jika
                robot belum pernah mengirim frame). Bila None, kembalikan robot
                yang terakhir update (perilaku lama).

        Returns:
            Optional[Dict[str, Any]]: Salinan dictionary fitur, atau None jika belum ada data.
        """
        with self.lock:
            if robot_id:
                slot = self._stores.get(robot_id)
                if slot is None:
                    return None
                return slot["features"].copy()
            if self._last_robot and self._last_robot in self._stores:
                return self._stores[self._last_robot]["features"].copy()
            if self._features is None:
                return None
            return self._features.copy()

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Semua fitur terakhir per robot_id."""
        with self.lock:
            return {rid: slot["features"].copy() for rid, slot in self._stores.items()}

    def list_robots(self, active_sec: float = 30.0) -> list:
        """Daftar robot_id + waktu update terakhir + flag aktif."""
        now = time.time()
        with self.lock:
            out = [
                {
                    "robot_id": rid,
                    "last_seen": slot["updated_at"],
                    "is_active": (now - slot["updated_at"]) <= active_sec,
                    "has_data": True,
                }
                for rid, slot in self._stores.items()
            ]
            out.sort(key=lambda r: r["last_seen"], reverse=True)
            return out

    def has_data(self, robot_id: Optional[str] = None) -> bool:
        """Mengecek apakah sudah ada data fitur yang masuk."""
        with self.lock:
            if robot_id:
                return robot_id in self._stores
            return self._features is not None

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
        Memperbarui status hardware (robot_trigger, lcd_command, speaker_command)
        pada snapshot fitur yang ada untuk robot_id tertentu atau seluruh robot.
        """
        with self.lock:
            target_ids = []
            if robot_id and robot_id in self._stores:
                target_ids = [robot_id]
            elif not robot_id:
                target_ids = list(self._stores.keys())

            for rid in target_ids:
                feat = self._stores[rid]["features"]
                feat["robot_trigger"] = trigger
                hw = feat.get("hardware")
                if isinstance(hw, dict):
                    if lcd_cmd:
                        hw["lcd_command"] = lcd_cmd
                    if speaker_cmd:
                        hw["speaker_command"] = speaker_cmd
                    if lcd_label:
                        hw["lcd_label"] = lcd_label
                    if speaker_label:
                        hw["speaker_label"] = speaker_label
                    hw["robot_trigger"] = trigger

            if self._features is not None:
                self._features["robot_trigger"] = trigger
                hw = self._features.get("hardware")
                if isinstance(hw, dict):
                    if lcd_cmd:
                        hw["lcd_command"] = lcd_cmd
                    if speaker_cmd:
                        hw["speaker_command"] = speaker_cmd
                    if lcd_label:
                        hw["lcd_label"] = lcd_label
                    if speaker_label:
                        hw["speaker_label"] = speaker_label
                    hw["robot_trigger"] = trigger

