import pytest

from llm_lander.gpu_scanner import GPUScanner


class _MemoryInfo:
    def __init__(self, total: int, used: int, free: int) -> None:
        self.total = total
        self.used = used
        self.free = free


class _FakeNVML:
    def __init__(self) -> None:
        self._handles = [object(), object()]

    def nvmlInit(self) -> None:
        return None

    def nvmlShutdown(self) -> None:
        return None

    def nvmlDeviceGetCount(self) -> int:
        return len(self._handles)

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
        return self._handles[index]

    def nvmlDeviceGetName(self, _handle: object) -> bytes:
        return b"Fake GPU"

    def nvmlDeviceGetMemoryInfo(self, _handle: object) -> _MemoryInfo:
        return _MemoryInfo(total=8 * 1024**3, used=2 * 1024**3, free=6 * 1024**3)


def test_get_gpu_info(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nvml = _FakeNVML()
    monkeypatch.setattr("llm_lander.gpu_scanner._load_nvml", lambda: fake_nvml)

    scanner = GPUScanner()
    info = scanner.get_gpu_info(0)

    assert info["gpu_name"] == "Fake GPU"
    assert info["total_vram_mb"] == pytest.approx(8192.0)
    assert info["used_vram_mb"] == pytest.approx(2048.0)
    assert info["free_vram_mb"] == pytest.approx(6144.0)


def test_get_all_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nvml = _FakeNVML()
    monkeypatch.setattr("llm_lander.gpu_scanner._load_nvml", lambda: fake_nvml)

    scanner = GPUScanner()
    infos = scanner.get_all_gpus()
    assert len(infos) == 2
