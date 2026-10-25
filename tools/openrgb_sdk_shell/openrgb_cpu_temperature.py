"""Persistent LibreHardwareMonitor processor-temperature input."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
DEFAULT_ASSEMBLY = Path(
    r"C:\Users\alber\AppData\Local\Programs\NiceHash Miner"
    r"\app_3.0.9.2\LibreHardwareMonitorLib.dll"
)
DEFAULT_SENSOR_IDENTIFIER = "/amdcpu/0/temperature/2"


class CpuTemperatureError(RuntimeError):
    pass


class CpuTemperatureReader:
    """Keep one elevated Windows PowerShell/LHM sensor process alive."""

    def __init__(
        self,
        assembly_path: str | Path = DEFAULT_ASSEMBLY,
        sensor_identifier: str = DEFAULT_SENSOR_IDENTIFIER,
    ) -> None:
        assembly = Path(assembly_path).resolve()
        if not assembly.is_file():
            raise CpuTemperatureError(
                f"LibreHardwareMonitor assembly not found: {assembly}"
            )
        if not sensor_identifier:
            raise ValueError("processor temperature sensor identifier must not be empty")
        script = HERE / "stream_cpu_temperature.ps1"
        self.sensor_identifier = sensor_identifier
        self._process = subprocess.Popen(
            (
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-File", str(script),
                "-AssemblyPath", str(assembly),
                "-SensorIdentifier", sensor_identifier,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    def read_celsius(self) -> float:
        if self._process.poll() is not None:
            detail = self._process.stderr.read().strip() if self._process.stderr else ""
            raise CpuTemperatureError(
                f"processor temperature worker exited unexpectedly: {detail}"
            )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write("sample\n")
        self._process.stdin.flush()
        response = self._process.stdout.readline().strip()
        if not response:
            raise CpuTemperatureError("processor temperature worker returned no value")
        if response.startswith("ERROR|"):
            raise CpuTemperatureError(response.partition("|")[2])
        try:
            value = float(response)
        except ValueError as exc:
            raise CpuTemperatureError(
                f"invalid processor temperature response: {response!r}"
            ) from exc
        if not math.isfinite(value) or not 0.0 < value < 150.0:
            raise CpuTemperatureError(
                f"processor temperature is outside the valid Celsius range: {value}"
            )
        return value

    def close(self) -> None:
        if self._process.poll() is None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()


__all__ = [
    "CpuTemperatureError",
    "CpuTemperatureReader",
    "DEFAULT_ASSEMBLY",
    "DEFAULT_SENSOR_IDENTIFIER",
]
