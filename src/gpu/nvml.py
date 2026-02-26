"""Thin pynvml wrapper for GPU monitoring."""

from dataclasses import dataclass

import pynvml

import log


@dataclass
class NvmlDeviceInfo:
    index: int
    handle: object  # pynvml handle
    uuid: str
    name: str
    total_memory: int  # bytes
    pci_bus_id: str = ""  # e.g. "00000000:01:00.0"


_initialized = False


def init() -> None:
    global _initialized
    if _initialized:
        return
    pynvml.nvmlInit()
    _initialized = True
    log.info(f"NVML initialized: driver {pynvml.nvmlSystemGetDriverVersion()}")


def shutdown() -> None:
    global _initialized
    if _initialized:
        pynvml.nvmlShutdown()
        _initialized = False


def get_device_count() -> int:
    return pynvml.nvmlDeviceGetCount()


def get_devices() -> list[NvmlDeviceInfo]:
    count = get_device_count()
    devices = []
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
        pci_bus_id = pci_info.busId
        if isinstance(pci_bus_id, bytes):
            pci_bus_id = pci_bus_id.decode("utf-8").rstrip("\x00")
        devices.append(NvmlDeviceInfo(
            index=i,
            handle=handle,
            uuid=uuid,
            name=name,
            total_memory=mem_info.total,
            pci_bus_id=pci_bus_id,
        ))
    return devices


def get_memory_info(handle: object) -> tuple[int, int, int]:
    """Returns (total, used, free) in bytes."""
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.total, info.used, info.free


def get_temperature(handle: object) -> int:
    """Returns GPU temperature in Celsius."""
    return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)


def get_utilization(handle: object) -> tuple[int, int]:
    """Returns (gpu_util%, memory_util%)."""
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return util.gpu, util.memory
