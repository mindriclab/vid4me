"""VRAM-Erkennung und Warnungen. Funktioniert auch ohne torch (nvidia-smi-Fallback)."""
from __future__ import annotations

import shutil
import subprocess

# Grobe Mindestanforderungen (GB) pro Modell-Familie bei aktivem CPU-Offload
VRAM_REQUIREMENTS_GB = {
    "wan22": 16.0,
    "chroma": 12.0,
}


def query_vram() -> dict:
    """Gibt {available: bool, total_gb, free_gb, used_gb, device_name} zurueck."""
    info = {"available": False, "total_gb": 0.0, "free_gb": 0.0,
            "used_gb": 0.0, "device_name": None, "source": None}

    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            info.update(
                available=True,
                total_gb=round(total / 1024**3, 2),
                free_gb=round(free / 1024**3, 2),
                used_gb=round((total - free) / 1024**3, 2),
                device_name=torch.cuda.get_device_name(0),
                source="torch",
            )
            return info
    except ImportError:
        pass

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=10,
            ).strip().splitlines()[0]
            name, total, free, used = [x.strip() for x in out.split(",")]
            info.update(
                available=True,
                total_gb=round(float(total) / 1024, 2),
                free_gb=round(float(free) / 1024, 2),
                used_gb=round(float(used) / 1024, 2),
                device_name=name,
                source="nvidia-smi",
            )
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass
    return info


def check_vram_for(plugin_name: str) -> dict:
    """VRAM-Check fuer ein Plugin: {ok, warning, vram}."""
    vram = query_vram()
    required = VRAM_REQUIREMENTS_GB.get(plugin_name, 8.0)
    result = {"ok": True, "warning": None, "required_gb": required, "vram": vram}
    if not vram["available"]:
        result["ok"] = False
        result["warning"] = "Keine CUDA-GPU erkannt. Generierung nicht moeglich."
    elif vram["total_gb"] < required:
        result["ok"] = False
        result["warning"] = (f"GPU hat {vram['total_gb']} GB VRAM, "
                             f"empfohlen sind mindestens {required} GB (mit CPU-Offload).")
    elif vram["free_gb"] < required * 0.75:
        result["warning"] = (f"Nur {vram['free_gb']} GB VRAM frei — andere GPU-Programme "
                             f"schliessen, sonst droht Out-of-Memory.")
    return result
