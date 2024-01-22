import enum
import uuid
from platform import uname

import psutil
import torch

from vllm import cuda_utils


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97  # pylint: disable=invalid-name
    max_shared_mem = cuda_utils.get_device_attribute(
        cudaDevAttrMaxSharedMemoryPerBlockOptin, gpu)
    return int(max_shared_mem)


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(uname()).lower()

import time
class TimeCounter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.st = time.time()
        self.ed = self.st
        self.total_time = 0.0
    
    def start(self):
        self.st = time.time()
        return self.st

    def end(self):
        self.ed = time.time()
        self.total_time += self.ed - self.st
        return self.ed
    
    def interval(self):
        return self.ed - self.st
    
    def __repr__(self) -> str:
        return f"TimeCounter Name: {self.name:>20}, interval = {self.interval():.4f}. start = {self.st:.4f}. end = {self.ed:.4f}. total = {self.total_time:.4f}"
    
from datetime import datetime

def _print_time(log: str):
    current_time = datetime.now()
    current_time_ft = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
    print(f"[LOG] {log}. Time NOW: {current_time_ft}")
    return current_time

def _print_time_interval(st: datetime, ed: datetime, log: str):
    time_diff = ed - st
    print(f"[LOG] Interval of :{log}, {time_diff.seconds} sec {time_diff.microseconds} us.")

from contextlib import contextmanager
ctx_layer = 0

@contextmanager
def ctx_get_inteval_datetime(name: str, sync: bool = False, device: int | None = None, record_list: list | None = None):
    global ctx_layer
    if sync:
        torch.cuda.synchronize(device)
    start = _print_time("  "*ctx_layer +"Enter: "+name)
    ctx_layer += 1
    try:
        yield
    finally:
        if sync:
            torch.cuda.synchronize(device)
        ctx_layer -= 1
        end = _print_time("  "*ctx_layer+"Exit: "+name)
        _print_time_interval(start, end, name)
        if record_list is not None:
            record_list.append(end - start)