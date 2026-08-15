# utils/fps_counter.py

import time
from collections import deque

class FPSCounter:
    """
    Menghitung Frame Per Second (FPS) menggunakan metode rolling average.
    
    Kelas ini menyimpan riwayat timestamp dari sejumlah frame terakhir
    (maksimal `max_frames`) untuk menghasilkan estimasi FPS yang lebih stabil
    dibandingkan menghitung FPS hanya dari jarak antar dua frame secara instan.
    """

    def __init__(self, max_frames: int = 30):
        """
        Inisialisasi penghitung FPS.

        Args:
            max_frames (int): Jumlah frame yang disimpan di memori untuk 
                              perhitungan rata-rata (default: 30).
        """
        # Menggunakan deque dengan ukuran maksimal agar frame terlama 
        # otomatis terhapus saat frame baru masuk
        self.frame_times = deque(maxlen=max_frames)

    def update(self) -> float:
        """
        Mencatat waktu frame saat ini dan menghitung FPS rata-rata.

        Returns:
            float: Nilai FPS rata-rata, atau 0.0 jika frame belum cukup.
        """
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Jika baru ada 1 frame, kita belum bisa menghitung jarak waktu
        if len(self.frame_times) <= 1:
            return 0.0
            
        # Hitung selisih waktu antara frame paling baru dan frame paling lama di deque
        elapsed_time = current_time - self.frame_times[0]
        
        if elapsed_time <= 0.0:
            return 0.0
            
        # FPS = Jumlah jarak frame / Total waktu yang dihabiskan
        fps = (len(self.frame_times) - 1) / elapsed_time
        
        return round(fps, 1)