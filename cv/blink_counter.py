# cv/blink_counter.py

import time
from utils.logger import get_logger

logger = get_logger(__name__)

class BlinkCounter:
    """
    Modul untuk mengakumulasi statistik kedipan mata dari waktu ke waktu.
    
    Kelas ini menyimpan state terkait waktu mulai (start time) dan menghitung
    metrik lanjutan seperti total kedipan (blink_count), laju kedipan per menit (blink_rate),
    serta durasi mata tertutup (eye_closure_duration).
    """

    def __init__(self):
        """
        Inisialisasi penghitung kedipan dengan waktu mulai saat ini.
        """
        self.blink_count: int = 0
        self.start_time: float = time.time()
        
        # State untuk melacak durasi mata tertutup
        self.closure_start_time: float = 0.0
        self.last_closure_duration: float = 0.0

    def process(self, eye_status: str, blink_event: bool, current_time: float) -> tuple[int, float, float]:
        """
        Memperbarui dan menghitung statistik kedipan berdasarkan input terbaru.

        Args:
            eye_status (str): Status mata saat ini ("Open" atau "Closed").
            blink_event (bool): True jika baru saja terjadi satu kedipan penuh.
            current_time (float): Timestamp saat frame diproses (dalam detik).

        Returns:
            tuple[int, float, float]: 
                - blink_count (int): Total kedipan yang telah terjadi.
                - blink_rate (float): Rata-rata kedipan per menit.
                - closure_duration (float): Durasi mata tertutup (detik).
        """
        # 1. Update total kedipan
        if blink_event:
            self.blink_count += 1
            logger.debug(f"Kedipan terdeteksi! Total: {self.blink_count}")

        # 2. Hitung laju kedipan (Blinks per Minute)
        elapsed_minutes = (current_time - self.start_time) / 60.0
        blink_rate = 0.0
        if elapsed_minutes > 0:
            blink_rate = self.blink_count / elapsed_minutes

        # 3. Hitung durasi mata tertutup (Closure Duration)
        if eye_status == "Closed":
            if self.closure_start_time == 0.0:
                self.closure_start_time = current_time
            # Durasi saat ini jika sedang tertutup
            closure_duration = current_time - self.closure_start_time
            self.last_closure_duration = closure_duration
        else:
            self.closure_start_time = 0.0
            # Kembalikan durasi dari penutupan terakhir jika mata sedang terbuka
            closure_duration = self.last_closure_duration

        return self.blink_count, round(blink_rate, 2), round(closure_duration, 2)