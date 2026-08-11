#!/usr/bin/env python3
"""Plot CUMULATIVE per-kernel latency for a trtexec --exportProfile JSON.

Two views of how the total inference latency is built up from the engine's kernels:
  (1) execution order  -> running total climbing to the full latency
  (2) sorted (Pareto)  -> how many kernels make up X% of the total time

On a launch-bound model (many tiny kernels, no dominant one) the Pareto curve hugs
the "perfectly flat" diagonal -- i.e. you need a large fraction of the kernels to
reach 50%/80% of the time. That is the visual signature of "death by a thousand cuts".

Usage:
  python plot_cumulative_latency.py [path/to/profile.json]
Defaults to qat_batch32_per_kernel_profile.json next to this script.
Requires: matplotlib, numpy.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Settings ----
JSON = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("qat_batch32_per_kernel_profile.json")
# trtexec per-kernel times are INSTRUMENTED (inflated); scale their sum to the real
# measured median GPU latency so the y-axis reads in true ms. Set to None to plot raw averageMs.
ANCHOR_MS = 1.19971   # QAT batch32 measured median (GPU compute, batch=1, no CUDA graph)

# ---- Load ----
data = json.load(open(JSON))
entries = [e for e in data if isinstance(e, dict) and "name" in e]   # drop the {"count": N} header
ms = np.array([e["averageMs"] for e in entries], dtype=float)
names = [e["name"] for e in entries]
total_inst = ms.sum()
scale = (ANCHOR_MS / total_inst) if ANCHOR_MS else 1.0
ms = ms * scale
total = ms.sum()
n = len(ms)

# ---- (1) cumulative in execution order ----
cum = np.cumsum(ms)
x = np.arange(1, n + 1)

# ---- (2) Pareto: sorted largest-first ----
order = np.argsort(ms)[::-1]
cum_sorted_pct = 100 * np.cumsum(ms[order]) / total

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

# left: running total through the network
ax1.plot(x, cum, color="#c0392b", lw=1.8)
ax1.fill_between(x, cum, alpha=0.12, color="#c0392b")
ax1.axhline(total, ls="--", color="gray", lw=0.8)
ax1.text(2, total * 0.96, f"total {total:.3f} ms", fontsize=8, color="gray")
ax1.set_xlabel("kernel # (execution order)")
ax1.set_ylabel("cumulative latency (ms)")
ax1.set_title(f"Cumulative latency through the engine\n{n} kernels → {total:.3f} ms")
ax1.grid(alpha=0.3)

# right: Pareto — kernels vs % of total time
ax2.plot(x, cum_sorted_pct, color="#2980b9", lw=1.8, label="actual")
ax2.plot([1, n], [100 / n, 100], ls=":", color="gray", lw=1, label="perfectly flat (all equal)")
for p in (50, 80):
    k = int(np.searchsorted(cum_sorted_pct, p)) + 1
    ax2.annotate(f"{p}% of time\nneeds {k} kernels", xy=(k, p), fontsize=8,
                 xytext=(k + n * 0.06, p - 12), arrowprops=dict(arrowstyle="->", lw=0.7))
ax2.set_xlabel("number of kernels (largest first)")
ax2.set_ylabel("cumulative % of total latency")
ax2.set_title("Pareto: how spread out the time is\n(near-diagonal = launch-bound)")
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
out = JSON.with_name("qat_batch32_cumulative_latency.png")
plt.savefig(out, dpi=140)
print(f"wrote {out}")
print(f"kernels={n}  total(scaled)={total:.4f} ms  biggest={names[int(order[0])][:46]} "
      f"({ms[int(order[0])]*1000:.1f} us, {cum_sorted_pct[0]:.1f}% of total)")
for p in (50, 80, 90):
    k = int(np.searchsorted(cum_sorted_pct, p)) + 1
    print(f"  {p}% of latency comes from the top {k} kernels ({100*k/n:.0f}% of kernels)")
