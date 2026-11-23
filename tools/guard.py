#!/usr/bin/env python3
"""Run a command under a resource guard (Linux / WSL2 / native Windows).

Samples GPU (nvidia-smi) and host (/proc, or the Win32 API on Windows) state every --interval
seconds, appends a CSV log, and kills the command's whole process tree if any limit is crossed, so a long benchmark or
training run on a laptop can never drive the machine into a thermal or out-of-memory reset.

Before starting, it waits for the GPU to cool below --start-max-temp and for the host load average
to settle; after the command exits, it
keeps sampling until the GPU has cooled below --cooldown-temp (bounded by --cooldown-max-s), so
back-to-back jobs never stack heat and every job ends with an idle, settled GPU.

    python tools/guard.py --log logs/run.csv -- python train.py ...

Exit code: the command's exit code, or 99 if the guard had to kill it.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

WINDOWS = os.name == "nt"


def gpu_sample() -> dict:
    q = "temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total,clocks.sm"
    """One sample over all visible GPUs, reporting the worst one for each limit: the hottest temperature,
    the highest power and utilization, the fullest memory (multi-GPU training)."""
    try:
        lines = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        f = lambda s: float(s) if s not in ("[N/A]", "N/A", "") else float("nan")  # noqa: E731
        gpus = [[f(x.strip()) for x in ln.split(",")] for ln in lines if ln.strip()]
        if not gpus:
            return {}
        fullest = max(gpus, key=lambda g: g[3] / g[4] if g[4] > 0 else 0.0)
        return {"gpu_temp_c": max(g[0] for g in gpus), "gpu_power_w": max(g[1] for g in gpus),
                "gpu_util": max(g[2] for g in gpus), "gpu_mem_used_mb": fullest[3],
                "gpu_mem_total_mb": fullest[4], "sm_clock_mhz": min(g[5] for g in gpus), "n_gpus": len(gpus)}
    except Exception:  # noqa: BLE001 - a failed sample must never crash the guard
        return {}


class _WinHost:
    """Windows has no /proc and no load average: memory comes from GlobalMemoryStatusEx and "load1"
    is busy CPUs (from GetSystemTimes), smoothed with the 60 s time constant of the Unix 1-min load
    average so the same --max-load-per-cpu thresholds apply. (Busy CPUs ignore the run queue, so this
    is never above the Unix value for the same work.)"""

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ct, self.wt = ctypes, wintypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        self.MEMSTAT = MEMSTAT
        self.k32 = ctypes.windll.kernel32
        self.prev = self._times()
        self.prev_t = time.time()
        self.load = None

    def _times(self):
        idle, kern, user = (self.wt.FILETIME() for _ in range(3))
        self.k32.GetSystemTimes(self.ct.byref(idle), self.ct.byref(kern), self.ct.byref(user))
        v = lambda f: (f.dwHighDateTime << 32) | f.dwLowDateTime  # noqa: E731  100 ns units
        return v(idle), v(kern) + v(user)  # kernel time includes idle time

    def sample(self) -> dict:
        m = self.MEMSTAT()
        m.dwLength = self.ct.sizeof(m)
        self.k32.GlobalMemoryStatusEx(self.ct.byref(m))
        ncpu = os.cpu_count() or 1
        if time.time() - self.prev_t < 0.5:  # need a measurable interval for the first sample
            time.sleep(0.5)
        (idle, total), t = self._times(), time.time()
        d_idle, d_total = idle - self.prev[0], total - self.prev[1]
        busy = ncpu * (1.0 - d_idle / d_total) if d_total > 0 else 0.0
        a = 1.0 - pow(2.718281828, -(t - self.prev_t) / 60.0)
        self.load = busy if self.load is None else self.load + a * (busy - self.load)
        self.prev, self.prev_t = (idle, total), t
        return {"host_mem_avail_mb": m.ullAvailPhys / 2**20, "host_mem_total_mb": m.ullTotalPhys / 2**20,
                "load1": self.load, "ncpu": ncpu}


_win_host = _WinHost() if WINDOWS else None


def host_sample() -> dict:
    if _win_host is not None:
        return _win_host.sample()
    mem = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, v = line.split(":", 1)
            mem[k] = int(v.split()[0])  # kB
    load1 = os.getloadavg()[0]
    return {"host_mem_avail_mb": mem["MemAvailable"] / 1024, "host_mem_total_mb": mem["MemTotal"] / 1024,
            "load1": load1, "ncpu": os.cpu_count()}


def start(cmd) -> subprocess.Popen:
    if WINDOWS:
        return subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return subprocess.Popen(cmd, start_new_session=True)


def kill_tree(proc: subprocess.Popen, hard: bool) -> None:
    if WINDOWS:  # taskkill /T takes the children (compiler workers, data loaders) with it
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T"] + (["/F"] if hard else []), capture_output=True)
    else:
        os.killpg(proc.pid, signal.SIGKILL if hard else signal.SIGTERM)


def wait_settled(max_load_per_cpu: float, max_s: float) -> float:
    """Wait until the 1-min load average is below the kill threshold (a previous job's spike decays
    slowly; starting into it would get the new job killed for load it did not cause)."""
    t0 = time.time()
    while time.time() - t0 < max_s:
        h = host_sample()
        if h["load1"] / h["ncpu"] < max_load_per_cpu:
            break
        time.sleep(2.0)
    return time.time() - t0


def wait_cool(below_c: float, max_s: float) -> float:
    """Sample every 2 s until the GPU is below `below_c` (or max_s passes); returns seconds waited."""
    t0 = time.time()
    if gpu_sample().get("gpu_temp_c", 0.0) < below_c:
        return 0.0  # (nvidia-smi itself can take seconds on Windows: don't report that as waiting)
    while time.time() - t0 < max_s:
        time.sleep(2.0)
        if gpu_sample().get("gpu_temp_c", 0.0) < below_c:
            break
    return time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--max-gpu-temp", type=float, default=84.0, help="kill at/above this GPU temp (C)")
    ap.add_argument("--max-gpu-mem-frac", type=float, default=0.90)
    ap.add_argument("--start-max-temp", type=float, default=65.0, help="wait until the GPU is below this to start")
    ap.add_argument("--cooldown-temp", type=float, default=65.0, help="after the command, wait until below this")
    ap.add_argument("--cooldown-max-s", type=float, default=120.0)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="kill after this many seconds (0: no limit)")
    ap.add_argument("--min-host-mem-mb", type=float, default=1500.0)
    ap.add_argument("--max-load-per-cpu", type=float, default=1.5, help="sustained 1-min load / ncpu")
    ap.add_argument("--hot-strikes", type=int, default=3, help="consecutive bad samples before kill")
    ap.add_argument("--log", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command given")

    log = None
    if a.log:
        os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
        log = open(a.log, "w", buffering=1)
        log.write("t_s,gpu_temp_c,gpu_power_w,gpu_util,gpu_mem_used_mb,sm_clock_mhz,host_mem_avail_mb,load1\n")

    waited = wait_cool(a.start_max_temp, 300.0)
    if waited > 1:
        print(f"[guard] waited {waited:.0f}s for the GPU to cool below {a.start_max_temp:.0f}C", file=sys.stderr)
    waited = wait_settled(0.5 * a.max_load_per_cpu, 180.0)
    if waited > 1:
        print(f"[guard] waited {waited:.0f}s for the host load to settle", file=sys.stderr)
    proc = start(cmd)
    t0, strikes, peak = time.time(), 0, {"temp": 0.0, "power": 0.0, "gpu_mem": 0.0, "min_host_mem": 1e12}
    killed_reason = None
    while proc.poll() is None:
        g, h = gpu_sample(), host_sample()
        temp = g.get("gpu_temp_c", 0.0)
        peak["temp"] = max(peak["temp"], temp)
        peak["power"] = max(peak["power"], g.get("gpu_power_w", 0.0))
        peak["gpu_mem"] = max(peak["gpu_mem"], g.get("gpu_mem_used_mb", 0.0))
        peak["min_host_mem"] = min(peak["min_host_mem"], h["host_mem_avail_mb"])
        if log:
            log.write(f"{time.time()-t0:.1f},{temp},{g.get('gpu_power_w','')},{g.get('gpu_util','')},"
                      f"{g.get('gpu_mem_used_mb','')},{g.get('sm_clock_mhz','')},{h['host_mem_avail_mb']:.0f},{h['load1']:.2f}\n")
        reasons = []
        if temp >= a.max_gpu_temp:
            reasons.append(f"GPU temp {temp:.0f}C >= {a.max_gpu_temp:.0f}C")
        if g.get("gpu_mem_total_mb") and g["gpu_mem_used_mb"] / g["gpu_mem_total_mb"] > a.max_gpu_mem_frac:
            reasons.append(f"GPU mem {g['gpu_mem_used_mb']:.0f}/{g['gpu_mem_total_mb']:.0f} MB")
        if h["host_mem_avail_mb"] < a.min_host_mem_mb:
            reasons.append(f"host mem available {h['host_mem_avail_mb']:.0f} MB < {a.min_host_mem_mb:.0f} MB")
        if h["load1"] / h["ncpu"] > a.max_load_per_cpu:
            reasons.append(f"load {h['load1']:.1f} on {h['ncpu']} CPUs")
        if a.max_seconds and time.time() - t0 > a.max_seconds:
            reasons.append(f"time limit {a.max_seconds:.0f}s")
        strikes = strikes + 1 if reasons else 0
        if strikes >= a.hot_strikes:
            killed_reason = "; ".join(reasons)
            kill_tree(proc, hard=WINDOWS)  # console programs ignore a soft taskkill
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                kill_tree(proc, hard=True)
            break
        time.sleep(a.interval)
    rc = proc.wait()
    cooled = wait_cool(a.cooldown_temp, a.cooldown_max_s)
    summary = (f"[guard] elapsed {time.time()-t0:.0f}s | peak GPU temp {peak['temp']:.0f}C, power {peak['power']:.0f}W, "
               f"GPU mem {peak['gpu_mem']:.0f}MB | min host mem avail {peak['min_host_mem']:.0f}MB"
               f" | cooldown {cooled:.0f}s")
    print(summary, file=sys.stderr, flush=True)
    if killed_reason:
        print(f"[guard] KILLED: {killed_reason}", file=sys.stderr, flush=True)
        return 99
    return rc


if __name__ == "__main__":
    sys.exit(main())
