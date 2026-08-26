# camera/base.py
"""
base.py — Interface abstrak untuk semua sumber video (CameraSource).

Menerapkan Strategy Pattern: modul pemrosesan frame hanya berinteraksi
dengan interface ini, sehingga sumber video (webcam, ESP32-CAM, dll.)
dapat diganti tanpa mengubah kode konsumen (Dependency Inversion Principle).
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class CameraSource(ABC):
    """
    Abstract Base Class untuk semua sumber video.

    Setiap implementasi konkret (WebcamSource, ESP32CamSource, dsb.)
    wajib mengimplementasikan seluruh method abstrak di bawah ini.
    """

    @abstractmethod
    def read_frame(self) -> np.ndarray | None:
        """
        Membaca frame terbaru dari sumber video.

        Returns:
            np.ndarray | None: Matriks gambar BGR (Height × Width × 3)
                               jika berhasil membaca frame.
                               None jika gagal atau kamera terputus.
        """
        ...

    @abstractmethod
    def get_fps(self) -> float:
        """
        Mendapatkan Frame Per Second (FPS) dari sumber video.

        Returns:
            float: FPS hardware atau estimasi stream. 0.0 jika tidak tersedia.
        """
        ...

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """
        Mendapatkan resolusi frame dari sumber video.

        Diperlukan oleh modul DistanceEstimator untuk menghitung
        focal length dalam model kamera pinhole.

        Returns:
            Tuple[int, int]: (width, height) dalam piksel.
        """
        ...

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Mengecek apakah koneksi ke sumber video masih aktif.

        Returns:
            bool: True jika kamera siap, False sebaliknya.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """
        Menutup koneksi ke sumber video dan membersihkan resource.
        Setelah dipanggil, instance ini tidak boleh digunakan lagi.
        """
        ...
