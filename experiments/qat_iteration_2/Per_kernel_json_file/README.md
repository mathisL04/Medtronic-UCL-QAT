# Per-kernel profile JSONs (trtexec --exportProfile)

Directly-comparable per-kernel latency profiles for the three production engines,
all generated with identical trtexec flags on an exclusive idle A100
(`--exportProfile --profilingVerbosity=detailed --separateProfileRun --useSpinWait
--warmUp=500 --iterations=3000 --duration=0 --avgRuns=200 --noDataTransfers`).

| file | engine | kernels | median GPU compute |
|---|---|--:|--:|
| `qat_batch32_per_kernel_profile.json` | QAT batch32 (INT8, explicit Q/DQ) | 245 | 1.200 ms |
| `ptq_int8_per_kernel_profile.json`    | PTQ INT8 (implicit quant)         | 189 | 1.072 ms |
| `fp16_per_kernel_profile.json`        | FP16 (same arch, NO Q/DQ)         | 231 | 1.127 ms |

**Identical schema** — one parser reads all three. Element `[0]` is a header
`{"count": N}` (iterations); skip it. Every following element is one kernel:

```json
{ "name": str, "timeMs": float, "averageMs": float, "medianMs": float, "percentage": float }
```
- `averageMs` = per-inference time (use this for summing/ranking)
- `timeMs` = total over all iterations · `percentage` = % of engine total
- Parse: `entries = [e for e in json.load(open(f)) if "name" in e]`

Layer-info / precision files for each engine are in `../profiling_exports/`.
