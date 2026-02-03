"""GPU scanning utilities using NVML."""

from __future__ import annotations

from typing import Any, TypedDict


class GPUInfo(TypedDict):
    """GPU information record."""

    gpu_index: int
    gpu_name: str
    total_vram_mb: float
    used_vram_mb: float
    free_vram_mb: float


def _load_nvml() -> Any:
    try:
        from py3nvml import py3nvml as nvml
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("py3nvml is not available") from exc
    return nvml


class GPUScanner:
    """Scan NVIDIA GPUs via NVML."""

    def __init__(self) -> None:
        self._nvml = _load_nvml()
        try:
            self._nvml.nvmlInit()
        except Exception as exc:  # pragma: no cover - depends on host
            raise RuntimeError("Failed to initialize NVML") from exc

    def close(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self._nvml.nvmlShutdown()

    def __del__(self) -> None:  # pragma: no cover - destructor best effort
        self.close()

    def get_gpu_info(self, gpu_index: int = 0) -> GPUInfo:
        try:
            handle = self._nvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception as exc:
            raise RuntimeError("GPU not found") from exc

        try:
            name = self._nvml.nvmlDeviceGetName(handle)
            mem_info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
        except Exception as exc:
            raise RuntimeError("Failed to query GPU information") from exc

        gpu_name = name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)

        total_mb = mem_info.total / (1024 * 1024)
        used_mb = mem_info.used / (1024 * 1024)
        free_mb = mem_info.free / (1024 * 1024)

        return {
            "gpu_index": gpu_index,
            "gpu_name": gpu_name,
            "total_vram_mb": float(total_mb),
            "used_vram_mb": float(used_mb),
            "free_vram_mb": float(free_mb),
        }

    def get_all_gpus(self) -> list[GPUInfo]:
        try:
            count = self._nvml.nvmlDeviceGetCount()
        except Exception as exc:
            raise RuntimeError("Failed to query GPU count") from exc

        return [self.get_gpu_info(i) for i in range(count)]
