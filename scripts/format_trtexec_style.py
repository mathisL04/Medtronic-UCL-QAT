"""Render our CUDA-event latency results in trtexec's '=== Performance summary ==='
layout, for a supervisor who expects that format.

NOT trtexec. trtexec is not installed on this host. These are OUR numbers, from
scripts/benchmark_latency_trt.py: real val100 frames, batch=1, N=1000 timed
inference calls, idle-GPU-gated. The GPU-side rows (H2D / GPU Compute / D2H) are
measured with CUDA events exactly as trtexec measures them, so those quantities
are directly comparable. Host Walltime is wall-clock and, on a loaded shared
box, includes CPU scheduling latency -- flagged inline.

Usage:  python scripts/format_trtexec_style.py fp32 fp16
        (reads the pooled_summary.provenance.json for each engine tag)
"""
import sys
import json
from pathlib import Path

RUNS = Path("/home/zcemml1/medtronic_qat_data/runs_sanoscience")


def row(label, s):
    # trtexec order: min, max, mean, median, percentile(90,95,99)
    return (f"{label}: min = {s['min_ms']:.4f} ms, max = {s['max_ms']:.4f} ms, "
            f"mean = {s['mean_ms']:.4f} ms, median = {s['median_ms']:.4f} ms, "
            f"percentile(90%) = {s['p90_ms']:.4f} ms, "
            f"percentile(95%) = {s['p95_ms']:.4f} ms, "
            f"percentile(99%) = {s['p99_ms']:.4f} ms")


def render(tag):
    f = RUNS / f"benchmark_latency_trt_{tag}_seed42_pooled_summary.provenance.json"
    d = json.load(open(f))
    s = d["summary"]
    n = s["Total"]["n"]
    exclusive = d["exclusive_gpu"]

    # Throughput: queries per second from GPU-compute median (what trtexec's
    # throughput tracks under single-stream), and separately from GPU latency.
    comp = s["Kernel(GPU)"]
    gpu_lat = s["GPUlatency"]
    qps_compute = 1000.0 / comp["median_ms"]

    print("=" * 66)
    print(f"  {tag.upper()} engine  --  OUR CUDA-EVENT HARNESS (not trtexec)")
    print(f"  N = {n} timed inferences, batch=1, val100 frames, idle-gated")
    print(f"  exclusive_gpu = {exclusive}")
    print("=" * 66)
    print("=== Performance summary ===")
    print(f"Throughput: {qps_compute:.1f} qps   "
          f"(1000 / GPU-compute median; single stream, batch=1)")
    print(row("Latency (GPU: H2D+Compute+D2H)", gpu_lat))
    print("Enqueue Time: n/a  "
          "(synchronous single-shot wrapper; no enqueue pipelining to measure)")
    print(row("H2D Latency", s["H2D(GPU)"]))
    print(row("GPU Compute Time", comp))
    print(row("D2H Latency", s["D2H(GPU)"]))
    walltime_median = s["Total"]["median_ms"]
    tag_flag = "" if exclusive else "   [wall-clock; loaded host inflates this]"
    print(f"Total Host Walltime (pipeline A, wall-clock): "
          f"{walltime_median:.4f} ms median{tag_flag}")
    print(f"Total GPU Compute Time (median): {comp['median_ms']:.4f} ms")
    print()


if __name__ == "__main__":
    tags = sys.argv[1:] or ["fp32", "fp16"]
    for t in tags:
        render(t)
