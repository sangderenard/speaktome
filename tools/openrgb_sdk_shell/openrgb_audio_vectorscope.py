"""ctypes binding for native WASAPI ingestion and fftfree analysis."""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_char_p, c_float, c_int, c_size_t, c_uint32, c_void_p
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DEFAULT_SCOPE_DLL = HERE / "native_audio_scope" / "build" / "Release" / "openrgb_audio_scope.dll"
DEFAULT_FFTFREE_DLL = WORKSPACE / "fftfree" / "build" / "Release" / "fft_cffi.dll"


class NativeAudioVectorscope:
    """Own the native speaker loopback and power-of-two fftfree plan."""

    def __init__(
        self,
        *,
        fft_size: int = 4096,
        oversample: int = 8,
        band_count: int = 128,
        min_frequency_hz: float = 27.5,
        max_frequency_hz: float = 18000.0,
        scope_dll: Path = DEFAULT_SCOPE_DLL,
        fftfree_dll: Path = DEFAULT_FFTFREE_DLL,
    ) -> None:
        if not scope_dll.is_file():
            raise RuntimeError(
                f"native audio scope is not built at {scope_dll}; run build_audio_scope.ps1"
            )
        if not fftfree_dll.is_file():
            raise RuntimeError(f"fftfree C ABI library is missing at {fftfree_dll}")
        self._library = ctypes.WinDLL(str(scope_dll))
        self._bind()
        self._handle = self._library.openrgb_audio_scope_create(
            str(fftfree_dll), int(fft_size), int(oversample), int(band_count),
            float(min_frequency_hz), float(max_frequency_hz),
        )
        if not self._handle:
            raise RuntimeError("native audio scope could not create its fftfree analysis context")
        self._configuration = None
        self.configure(
            fft_size=fft_size,
            oversample=oversample,
            band_count=band_count,
            min_frequency_hz=min_frequency_hz,
            max_frequency_hz=max_frequency_hz,
        )
        self.sample_rate = 0
        self.channel_count = 2
        self.last_rms = 0.0

    def _bind(self) -> None:
        library = self._library
        library.openrgb_audio_scope_create.argtypes = (
            ctypes.c_wchar_p, c_size_t, c_size_t, c_size_t, c_float, c_float,
        )
        library.openrgb_audio_scope_create.restype = c_void_p
        library.openrgb_audio_scope_start.argtypes = (c_void_p,)
        library.openrgb_audio_scope_start.restype = c_int
        library.openrgb_audio_scope_configure.argtypes = (
            c_void_p, c_size_t, c_size_t, c_size_t, c_float, c_float,
        )
        library.openrgb_audio_scope_configure.restype = c_int
        library.openrgb_audio_scope_snapshot.argtypes = (
            c_void_p, c_float, POINTER(c_float), POINTER(c_float), POINTER(c_float),
            c_size_t, POINTER(c_float), POINTER(c_uint32),
        )
        library.openrgb_audio_scope_snapshot.restype = c_size_t
        library.openrgb_audio_scope_running.argtypes = (c_void_p,)
        library.openrgb_audio_scope_running.restype = c_int
        library.openrgb_audio_scope_error.argtypes = (c_void_p,)
        library.openrgb_audio_scope_error.restype = c_char_p
        library.openrgb_audio_scope_free.argtypes = (c_void_p,)
        library.openrgb_audio_scope_free.restype = None

    @property
    def running(self) -> bool:
        return bool(self._handle and self._library.openrgb_audio_scope_running(self._handle))

    @property
    def error(self) -> str | None:
        if not self._handle:
            return "native audio scope is closed"
        raw = self._library.openrgb_audio_scope_error(self._handle)
        value = raw.decode("utf-8", errors="replace") if raw else ""
        return value or None

    def start(self) -> None:
        if not self._library.openrgb_audio_scope_start(self._handle):
            raise RuntimeError("native audio scope refused to start")
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and not self.running and self.error is None:
            time.sleep(0.01)
        if self.error:
            raise RuntimeError(self.error)
        if not self.running:
            raise RuntimeError("native audio scope did not start within five seconds")

    def configure(
        self,
        *,
        fft_size: int,
        oversample: int,
        band_count: int,
        min_frequency_hz: float,
        max_frequency_hz: float,
    ) -> None:
        configuration = (
            int(fft_size), int(oversample), int(band_count),
            float(min_frequency_hz), float(max_frequency_hz),
        )
        if configuration == self._configuration:
            return
        if not self._library.openrgb_audio_scope_configure(self._handle, *configuration):
            raise ValueError(
                "native vectorscope requires power-of-two FFT size/oversample, "
                "positive bands, and increasing frequency limits"
            )
        self._configuration = configuration

    def vectorscope_bands(self, *, gain: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        band_count = self._configuration[2]
        xy = np.zeros((band_count, 2), dtype=np.float32)
        power = np.zeros(band_count, dtype=np.float32)
        frequency = np.zeros(band_count, dtype=np.float32)
        rms = c_float(0.0)
        sample_rate = c_uint32(0)
        produced = self._library.openrgb_audio_scope_snapshot(
            self._handle,
            float(gain),
            xy.ctypes.data_as(POINTER(c_float)),
            power.ctypes.data_as(POINTER(c_float)),
            frequency.ctypes.data_as(POINTER(c_float)),
            band_count,
            byref(rms),
            byref(sample_rate),
        )
        if produced == 0:
            if self.error:
                raise RuntimeError(self.error)
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        self.sample_rate = int(sample_rate.value)
        self.last_rms = float(rms.value)
        return (
            xy[:produced].astype(np.float64),
            power[:produced].astype(np.float64),
            frequency[:produced].astype(np.float64),
        )

    def close(self) -> None:
        if self._handle:
            self._library.openrgb_audio_scope_free(self._handle)
            self._handle = None


__all__ = ["NativeAudioVectorscope"]
