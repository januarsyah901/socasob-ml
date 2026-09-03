# realtime/hardware_controller.py
"""
hardware_controller.py — Pengendali Perintah Aktuator Hardware (LCD & Speaker).

Menerjemahkan status kesehatan mata (fatigue, dry eye, myopia guard, startup, break)
menjadi 5 ekspresi LCD dan 5 efek suara Speaker sesuai spesifikasi rancangan robot:

========================================================================================
 KONDISI / EVENT                 | EKSPRESI LCD           | EFEK SUARA SPEAKER
========================================================================================
 1. Startup Pertama Kali         | normal                 | "cling!"
 2. Normal (Kedip Normal)        | normal                 | none (tidak)
 3. Mata Lelah (0 - 5 Menit)     | fatigue_5m (mata sayu) | none (tidak)
 4. Mata Lelah (>= 10 Menit)     | fatigue_10m (kesal)    | "bip-bip"
 5. Istirahat 20 Menit (20s)     | break_20m (senang)     | "ting-tong"
 6. Selesai Istirahat 20s        | normal                 | "ta-da"
 7. Terdeteksi Mata Kering       | dry_eye (sipit/kecewa) | "pop-pop"
========================================================================================
"""

import time
from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class HardwareActuatorController:
    """
    State machine pengendali ekspresi LCD & suara Speaker untuk Robot.
    """

    def __init__(self):
        # Tracking status fatigue
        self._fatigue_start_time: Optional[float] = None
        self._is_fatigued: bool = False

        # Tracking status break (aturan 20-20-20)
        self._was_on_break: bool = False

        # Flag startup robot pertama kali
        self._startup_sound_emitted: bool = False

        # Cache perintah terakhir agar tidak berulang kirim suara yang sama terus-menerus
        self._last_speaker_command: Optional[str] = None
        self._last_speaker_time: float = 0.0

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluasi hasil deteksi dari InferenceEngine dan hasilkan perintah hardware.

        Args:
            results: Dict dari InferenceEngine.run() yang berisi:
                     'fatigue', 'dry_eye', 'myopia_risk'.

        Returns:
            Dict payload perintah hardware:
                {
                    "target": "hardware",
                    "lcd_command": str,       # "normal" | "fatigue_5m" | "fatigue_10m" | "break_20m" | "dry_eye"
                    "speaker_command": str,   # "cling" | "bip-bip" | "ting-tong" | "pop-pop" | "ta-da" | "none"
                    "lcd_label": str,
                    "speaker_label": str,
                    "fatigue_duration_sec": float,
                    "break_remaining_sec": float,
                    "timestamp": float
                }
        """
        now = time.time()

        fatigue_data = results.get("fatigue", {})
        dry_eye_data = results.get("dry_eye", {})
        myopia_data = results.get("myopia_risk", {})

        fatigue_status = fatigue_data.get("status", "Aman")
        dry_eye_status = dry_eye_data.get("status", "Aman")
        break_state = myopia_data.get("break_state", "active")
        break_remaining = myopia_data.get("break_remaining_sec", 0.0)

        # ─────────────────────────────────────────────────────────────
        # 1. Event Startup Pertama Kali
        # ─────────────────────────────────────────────────────────────
        if not self._startup_sound_emitted:
            self._startup_sound_emitted = True
            logger.info("[Hardware] Startup event triggered -> Suara: 'cling!'")
            return self._build_payload(
                lcd="normal",
                speaker="cling",
                lcd_label="Muka Normal (Robot Startup)",
                speaker_label="Suara 'cling!' (Startup Robot)",
                fatigue_sec=0.0,
                break_rem=0.0,
            )

        # ─────────────────────────────────────────────────────────────
        # 2. Tracking Durasi Fatigue
        # ─────────────────────────────────────────────────────────────
        fatigue_duration_sec = 0.0
        if "Peringatan" in fatigue_status or "Berat" in fatigue_status:
            if self._fatigue_start_time is None:
                self._fatigue_start_time = now
            fatigue_duration_sec = now - self._fatigue_start_time
            self._is_fatigued = True
        else:
            self._fatigue_start_time = None
            self._is_fatigued = False

        # ─────────────────────────────────────────────────────────────
        # 3. Evaluasi Kondisi Prioritas
        # ─────────────────────────────────────────────────────────────

        lcd_cmd = "normal"
        speaker_cmd = "none"
        lcd_lbl = "Muka Normal (Kedip Normal)"
        speaker_lbl = "Tidak Bersuara"

        # (A) PRIORITAS 1: Selesai Sesi Istirahat 20s -> Trigger Suara "ta-da"
        if self._was_on_break and (break_state == "active" or break_remaining == 0.0):
            self._was_on_break = False
            lcd_cmd = "normal"
            speaker_cmd = "ta-da"
            lcd_lbl = "Muka Normal (Istirahat Selesai)"
            speaker_lbl = "Suara 'ta-da' (Sesi Istirahat Selesai)"
            logger.info("[Hardware] Istirahat 20s selesai -> Trigger: 'ta-da'")

        # (B) PRIORITAS 2: Sesi Istirahat 20s Aktif (Break 20 Menit)
        elif break_state in ("break_needed", "on_break") or break_remaining > 0:
            self._was_on_break = True
            lcd_cmd = "break_20m"
            lcd_lbl = "Muka Senang (Peringatan Istirahat 20 Detik)"
            
            # Sound "ting-tong" dipicu saat istirahat dimulai
            if self._last_speaker_command != "ting-tong" or (now - self._last_speaker_time) > 15.0:
                speaker_cmd = "ting-tong"
                speaker_lbl = "Suara 'ting-tong' (Pengingat Istirahat)"
            else:
                speaker_cmd = "none"
                speaker_lbl = "Tidak Bersuara"

        # (C) PRIORITAS 3: Terdeteksi Mata Kering
        elif "Peringatan" in dry_eye_status or "Berat" in dry_eye_status:
            lcd_cmd = "dry_eye"
            lcd_lbl = "Muka Kecewa/Sipit (Terdeteksi Mata Kering)"

            if self._last_speaker_command != "pop-pop" or (now - self._last_speaker_time) > 10.0:
                speaker_cmd = "pop-pop"
                speaker_lbl = "Suara 'pop-pop' (Deteksi Mata Kering)"
            else:
                speaker_cmd = "none"
                speaker_lbl = "Tidak Bersuara"

        # (D) PRIORITAS 4: Mata Lelah >= 10 Menit (600 Detik)
        elif self._is_fatigued and fatigue_duration_sec >= 600.0:
            lcd_cmd = "fatigue_10m"
            lcd_lbl = "Muka Kesal/Tajam (Mata Lelah >= 10 Menit)"

            if self._last_speaker_command != "bip-bip" or (now - self._last_speaker_time) > 15.0:
                speaker_cmd = "bip-bip"
                speaker_lbl = "Suara 'bip-bip' (Peringatan Mata Lelah 10m)"
            else:
                speaker_cmd = "none"
                speaker_lbl = "Tidak Bersuara"

        # (E) PRIORITAS 5: Mata Lelah 5 Menit Pertama (< 600 Detik)
        elif self._is_fatigued and fatigue_duration_sec > 0:
            lcd_cmd = "fatigue_5m"
            lcd_lbl = "Muka Sayu (Mata Lelah 5 Menit Pertama)"
            speaker_cmd = "none"
            speaker_lbl = "Tidak Bersuara"

        # (F) DEFAULT: Kondisi Normal
        else:
            lcd_cmd = "normal"
            speaker_cmd = "none"
            lcd_lbl = "Muka Normal (Kondisi Normal, Kedip Normal)"
            speaker_lbl = "Tidak Bersuara"

        # Catat histori speaker command jika bersuara
        if speaker_cmd != "none":
            self._last_speaker_command = speaker_cmd
            self._last_speaker_time = now

        return self._build_payload(
            lcd=lcd_cmd,
            speaker=speaker_cmd,
            lcd_label=lcd_lbl,
            speaker_label=speaker_lbl,
            fatigue_sec=fatigue_duration_sec,
            break_rem=break_remaining,
        )

    @staticmethod
    def to_robot_trigger(lcd_cmd: str) -> str:
        """
        Memetakan perintah LCD internal ke 4 trigger pesan teks untuk robot:
        - 'normal'       -> 'normal'
        - 'fatigue_5m'   -> '5'
        - 'fatigue_10m'  -> '10'
        - 'dry_eye'      -> 'dry'
        - default        -> 'normal'
        """
        mapping = {
            "normal": "normal",
            "fatigue_5m": "5",
            "fatigue_10m": "10",
            "dry_eye": "dry",
        }
        return mapping.get(lcd_cmd, "normal")

    def _build_payload(
        self,
        lcd: str,
        speaker: str,
        lcd_label: str,
        speaker_label: str,
        fatigue_sec: float,
        break_rem: float,
    ) -> Dict[str, Any]:
        return {
            "target": "hardware",
            "lcd_command": lcd,
            "robot_trigger": self.to_robot_trigger(lcd),
            "speaker_command": speaker,
            "lcd_label": lcd_label,
            "speaker_label": speaker_label,
            "fatigue_duration_sec": round(fatigue_sec, 1),
            "break_remaining_sec": round(break_rem, 1),
            "timestamp": time.time(),
        }

