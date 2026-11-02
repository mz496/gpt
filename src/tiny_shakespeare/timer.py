import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from .utils import get_device

class PhaseTimer:
    """
    Accumulates wall-clock time per named phase, split into CPU time and GPU tail.

    MPS/CUDA run asynchronously: Python queues kernels and returns, and the work finishes later. So for each phase we record:
    - cpu: time until the Python block returns, i.e. how long the CPU spent issuing work (or blocked inside it, e.g. on `.item()`).
    - total: time until the device has finished everything the phase queued (we sync at the end).
    - gpu tail = total - cpu: device work still running after the CPU was done.
    Big gpu tail => GPU-bound. cpu ~= total => CPU-bound (launch overhead, Python loops, or a blocking sync inside the phase).
    We also sync at the start so work queued by an earlier phase isn't billed to this one.

    The syncs slow training down, so only use sync=True for profiling (--profile). With sync=False, total == cpu.
    A disabled timer makes `phase` a no-op.
    """
    def __init__(self, enabled: bool = True, sync: bool = False):
        self.enabled = enabled
        self.sync = sync
        self.cpu: dict[str, float] = defaultdict(float)
        self.total: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def _sync(self) -> None:
        if not self.sync:
            return
        device = get_device()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            cpu_end = time.perf_counter()
            self._sync()
            end = time.perf_counter()
            self.cpu[name] += cpu_end - start
            self.total[name] += end - start
            self.counts[name] += 1

    def report(self, title: str) -> None:
        if not self.enabled or not self.total:
            return
        grand_total = sum(self.total.values())
        print(f"--- {title} (total {grand_total:.2f}s) ---")
        print(f"{'phase':>16}  {'total':>8} {'%':>6} {'calls':>6} {'ms/call':>9} {'cpu ms':>9} {'gpu tail ms':>12}")
        for name, t in sorted(self.total.items(), key=lambda kv: -kv[1]):
            n = self.counts[name]
            cpu = self.cpu[name]
            print(f"{name:>16}  {t:7.3f}s {100 * t / grand_total:5.1f}% {n:>6} {1000 * t / n:9.2f} {1000 * cpu / n:9.2f} {1000 * (t - cpu) / n:12.2f}")
