"""Experimental per-op kernel profiling (opt-in via OPERATORX_PROFILE=1).

After the runner's normal event-timed measurement, the op's kernel is
replayed a few times under torch.profiler (CUPTI on CUDA, rocprofiler on
ROCm -- both through kineto) and a compact per-kernel summary is attached
to the result metrics:

  metrics["profile"] = {
    "iters": N,                       # profiled replays
    "kernels": [                      # one entry per distinct kernel/memop,
      {"name": ..., "cat": "kernel",  #   sorted by time, per-CALL numbers
       "count_per_call": 2.0, "us_per_call": 31.2,
       "grid": [..], "block": [..], "regs": 255, "smem": 0,
       "blocks_per_sm": ..., "warps_per_sm": ..., "occupancy_pct": ...},
      ...],
    "gpu_us_per_call": ...,           # sum over kernels (compare to latency_us:
                                      #   the gap is launch/idle time)
    "trace": "traces/op000123.json",  # present when a full chrome trace was kept
  }

OPERATORX_PROFILE_TRACE_DIR   keep full chrome traces here (default: off)
OPERATORX_PROFILE_TRACE_EVERY keep every Nth op's trace (default 200)
OPERATORX_PROFILE_ITERS       replay count under the profiler (default 3)

Launch-config fields depend on what the platform's kineto backend reports;
missing fields are simply absent. This is measurement-side instrumentation
only -- it runs after timing and cannot affect recorded latencies.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import torch

PROFILE = os.environ.get("OPERATORX_PROFILE", "") == "1"
_TRACE_DIR = os.environ.get("OPERATORX_PROFILE_TRACE_DIR") or None
_TRACE_EVERY = int(os.environ.get("OPERATORX_PROFILE_TRACE_EVERY", "200"))
_ITERS = int(os.environ.get("OPERATORX_PROFILE_ITERS", "3"))

_ARG_FIELDS = (
    ("grid", "grid"),
    ("block", "block"),
    ("registers per thread", "regs"),
    ("shared memory", "smem"),
    ("blocks per SM", "blocks_per_sm"),
    ("warps per SM", "warps_per_sm"),
    ("est. achieved occupancy %", "occupancy_pct"),
)

_counter = 0


def profile_op(kernel_fn) -> dict | None:
    """Replay kernel_fn under torch.profiler; return the summary dict."""
    global _counter
    if not PROFILE:
        return None
    _counter += 1
    acts = [torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA]
    try:
        with torch.profiler.profile(activities=acts) as prof:
            for _ in range(_ITERS):
                kernel_fn()
            torch.cuda.synchronize()
    except Exception as e:  # profiling must never fail the measurement
        return {"error": f"{type(e).__name__}: {e}"[:200]}

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        events = json.load(open(path)).get("traceEvents", [])
    except Exception as e:
        os.unlink(path)
        return {"error": f"trace: {type(e).__name__}: {e}"[:200]}

    kernels: dict[str, dict] = {}
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in (
                "kernel", "gpu_memcpy", "gpu_memset"):
            continue
        name = e.get("name", "")[:200]
        a = e.get("args") or {}
        k = kernels.setdefault(name, {"name": name, "cat": e["cat"],
                                      "count": 0, "total_us": 0.0})
        k["count"] += 1
        k["total_us"] += float(e.get("dur", 0.0))
        for src, dst in _ARG_FIELDS:
            if src in a and dst not in k:
                k[dst] = a[src]

    out = []
    gpu_us = 0.0
    for k in kernels.values():
        k["count_per_call"] = round(k.pop("count") / _ITERS, 2)
        k["us_per_call"] = round(k.pop("total_us") / _ITERS, 3)
        gpu_us += k["us_per_call"]
        out.append(k)
    out.sort(key=lambda x: -x["us_per_call"])

    summary = {"iters": _ITERS, "kernels": out,
               "gpu_us_per_call": round(gpu_us, 3)}
    try:
        if _TRACE_DIR and (_counter % _TRACE_EVERY) == 1:
            os.makedirs(_TRACE_DIR, exist_ok=True)
            dest = os.path.join(_TRACE_DIR, f"op{_counter:06d}.json")
            # shutil.move, not os.replace: the temp file lives on a different
            # filesystem than the (typically bind-mounted) trace dir.
            shutil.move(path, dest)
            summary["trace"] = dest
        else:
            os.unlink(path)
    except OSError as e:
        summary["trace_error"] = str(e)[:120]
    return summary
