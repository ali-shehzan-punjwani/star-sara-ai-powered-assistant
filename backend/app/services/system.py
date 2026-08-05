"""Host telemetry for the SYSTEM dashboard card."""

from __future__ import annotations

import shutil
import socket
import time
from typing import Any, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]

_STARTED = time.time()


def _battery() -> Optional[dict[str, Any]]:
    if psutil is None or not hasattr(psutil, "sensors_battery"):
        return None
    battery = psutil.sensors_battery()
    if battery is None:
        return None
    return {"percent": round(battery.percent), "plugged": battery.power_plugged}


def _network_online(host: str = "1.1.1.1", port: int = 53, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def snapshot() -> dict[str, Any]:
    disk = shutil.disk_usage("/")
    stats: dict[str, Any] = {
        "uptime_seconds": round(time.time() - _STARTED),
        "disk_percent": round(disk.used / disk.total * 100, 1),
        "battery": _battery(),
        "network": _network_online(),
    }
    if psutil is not None:
        memory = psutil.virtual_memory()
        stats.update(
            cpu_percent=psutil.cpu_percent(interval=None),
            cpu_cores=psutil.cpu_count(logical=True),
            ram_percent=memory.percent,
            ram_used_gb=round(memory.used / 1024**3, 1),
            ram_total_gb=round(memory.total / 1024**3, 1),
        )
    else:
        stats.update(cpu_percent=None, cpu_cores=None, ram_percent=None)
    return stats
