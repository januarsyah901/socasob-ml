# cv/blink_detector.py

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class BlinkDetector:
    """
    State machine untuk melacak status mata (Open/Closed) dan mendeteksi kedipan.
    
    Kelas ini membaca nilai EAR dari waktu ke waktu dan membandingkannya
    dengan EAR_THRESHOLD dan CONSEC_FRAMES dari file konfigurasi.
    """

    def __init__(self):
        """
        Inisialisasi BlinkDetector dengan parameter dari config.
        """
        self.ear_threshold = settings.EAR_THRESHOLD
        self.consec_frames = settings.CONSEC_FRAMES
        
        # State internal
        self.frame_counter = 0
        self.eye_status = "Open"

    def process(self, ear: float) -> tuple[str, bool]:
        """
        Memproses nilai EAR terbaru untuk mengupdate status mata dan mendeteksi kedipan.

        Args:
            ear (float): Nilai Eye Aspect Ratio dari frame saat ini.

        Returns:
            tuple[str, bool]: 
                - str: Status mata saat ini ("Open" atau "Closed").
                - bool: True jika satu event kedipan penuh baru saja selesai, False jika tidak.
        """
        blink_event = False

        # Jika EAR di bawah threshold, berarti mata sedang tertutup
        if ear < self.ear_threshold:
            self.frame_counter += 1
            self.eye_status = "Closed"
            
        # Jika EAR di atas threshold, berarti mata terbuka
        else:
            # Jika sebelumnya mata tertutup selama durasi yang cukup panjang (>= CONSEC_FRAMES),
            # maka catat ini sebagai satu kedipan (blink) yang valid.
            if self.frame_counter >= self.consec_frames:
                blink_event = True
                
            # Reset counter karena mata sudah terbuka kembali
            self.frame_counter = 0
            self.eye_status = "Open"

        return self.eye_status, blink_event