#!/usr/bin/env python3
"""Run a command under a resource guard (Linux / WSL2).

Samples GPU (nvidia-smi) and host (/proc) state every --interval seconds, appends a CSV log, and
kills the whole process group of the command if any limit is crossed, so a long benchmark or
training run on a laptop can never drive the machine into a thermal or out-of-memory reset.

Before starting, it waits for the GPU to cool below --start-max-temp; after the command exits, it
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


def gpu_sample() -> dict:
    q = "temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total,clocks.sm"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        t, p, u, mu, mt, clk = [x.strip() for x in out.split(",")]
        f = lambda s: float(s) if s not in ("[N/A]", "N/A", "") else float("nan")  # noqa: E731
        return {"gpu_temp_c": f(t), "gpu_power_w": f(p), "gpu_util": f(u),
                "gpu_mem_used_mb": f(mu), "gpu_mem_total_mb": f(mt), "sm_clock_mhz": f(clk)}
    except Exception:  # noqa: BLE001 - a failed sample must never crash the guard
        return {}


def host_sample() -> dict:
    mem = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, v = line.split(":", 1)
            mem[k] = int(v.split()[0])  # kB
    load1 = os.getloadavg()[0]
    return {"host_mem_avail_mb": mem["MemAvailable"] / 1024, "host_mem_total_mb": mem["MemTotal"] / 1024,
            "load1": load1, "ncpu": os.cpu_count()}


def wait_cool(below_c: float, max_s: float) -> float:
    """Sample every 2 s until the GPU is below `below_c` (or max_s passes); returns seconds waited."""
    t0 = time.time()
    while time.time() - t0 < max_s:
        if gpu_sample().get("gpu_temp_c", 0.0) < below_c:
            break
        time.sleep(2.0)
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
    proc = subprocess.Popen(cmd, start_new_session=True)
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
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
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
