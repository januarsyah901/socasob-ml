# camera/base_camera.py

from abc import ABC, abstractmethod
import numpy as np

class BaseCamera(ABC):
    """
    Abstract Base Class (ABC) untuk semua sumber video.
    
    Kelas ini menerapkan pola Strategy untuk sumber video. Modul lain dalam
    aplikasi hanya perlu berinteraksi dengan interface ini (Dependency Inversion),
    sehingga sumber video dapat diganti (misal dari Webcam ke ESP32-CAM)
    tanpa mengubah kode yang mengonsumsinya.
    """

    @abstractmethod
    def read_frame(self) -> np.ndarray | None:
        """
        Membaca frame terbaru dari sumber video.

        Returns:
            np.ndarray | None: Matriks gambar BGR berupa array NumPy jika berhasil membaca frame.
                               Mengembalikan None jika gagal membaca atau kamera terputus.
        """
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """
        Mendapatkan nilai Frame Per Second (FPS) dari sumber video.

        Returns:
            float: Nilai FPS dari hardware kamera atau estimasi stream.
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Mengecek apakah koneksi ke sumber video masih aktif/terbuka.

        Returns:
            bool: True jika kamera siap dan terbuka, False sebaliknya.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Menutup koneksi ke sumber video dan membersihkan resource.
        """
        pass