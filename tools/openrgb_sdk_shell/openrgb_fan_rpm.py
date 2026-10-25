"""Persistent LibreHardwareMonitor fan-speed input."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess

from openrgb_cpu_temperature import DEFAULT_ASSEMBLY


HERE = Path(__file__).resolve().parent


class FanRpmError(RuntimeError):
    pass


class FanRpmReader:
    """Read instantaneous RPM from one persistent elevated LHM worker."""

    def __init__(
        self,
        assembly_path: str | Path = DEFAULT_ASSEMBLY,
        sensor_identifier: str = "",
    ) -> None:
        assembly = Path(assembly_path).resolve()
        if not assembly.is_file():
            raise FanRpmError(f"LibreHardwareMonitor assembly not found: {assembly}")
        script = HERE / "stream_fan_rpm.ps1"
        arguments = (
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(script), "-AssemblyPath", str(assembly),
        )
        if sensor_identifier:
            arguments += ("-SensorIdentifier", sensor_identifier)
        self.sensor_identifier = sensor_identifier
        self.sensor_name = ""
        self._process = subprocess.Popen(
            arguments,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    def read_rpm(self) -> float:
        if self._process.poll() is not None:
            detail = self._process.stderr.read().strip() if self._process.stderr else ""
            raise FanRpmError(f"fan RPM worker exited unexpectedly: {detail}")
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write("sample\n")
        self._process.stdin.flush()
        response = self._process.stdout.readline().strip()
        if not response:
            raise FanRpmError("fan RPM worker returned no value")
        if response.startswith("ERROR|"):
            raise FanRpmError(response.partition("|")[2])
        parts = response.split("|", 2)
        try:
            value = float(parts[0])
        except ValueError as exc:
            raise FanRpmError(f"invalid fan RPM response: {response!r}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise FanRpmError(f"fan RPM is not positive and finite: {value}")
        if len(parts) > 1:
            self.sensor_identifier = parts[1]
        if len(parts) > 2:
            self.sensor_name = parts[2]
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


__all__ = ["FanRpmError", "FanRpmReader"]
