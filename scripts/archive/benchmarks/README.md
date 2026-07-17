# Archived benchmark scripts (provenance only)

These scripts are kept as **provenance, not as working tools**. They produced the superseded
~16.1 ms FP32 latency figure that appears in the week's report; they are not to be run for
new measurements.

- `benchmark_20_images_latency.py` — 20-image smoke demo (device hardcoded, no warmup, no
  thread pinning).
- `benchmark_fp32_geneva_ram_stable.py` — 5×100 RAM-preloaded run behind the earlier
  ~16–17 ms figures.

Both were superseded by `scripts/benchmark_latency.py`, which adds a compute-only idle-GPU
gate, strict `DEVICE`, per-repeat contention snapshots, and pooled percentile stats. On a
verified-idle A100 it establishes the real baseline (~8.6 ms median total). The gap from the
old ~16 ms figure was environmental (a contended / pre-driver-fix box), not a code change.

Do not build on or re-run these. Use `scripts/benchmark_latency.py`.
