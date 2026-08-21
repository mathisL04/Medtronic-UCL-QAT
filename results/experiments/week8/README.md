# Week 8 — frozen baseline and the per-layer INT8 QAT sweep

Everything from week 8 lives here: the two experiments, the scripts that ran them, the
recaps, and the notebook that plots the result. Nothing in this folder is needed to
reproduce stages 1–7; it builds on them.

## The question

Stages 1–7 established a precision ladder on the fully-trained model (§3 of the root
`README.md`). Week 8 asks a different question: **which layers actually carry the
accuracy cost of INT8, and does the choice of layer move latency?**

To make that answerable, the model is *frozen* — backbone and neck fixed, only the head
trained — so that the one thing varying between runs is which layer gets quantised.

## Layout

| Path | What it is |
|---|---|
| `frozen_baseline/` | Step 1–2. `train_frozen_baseline.py` fine-tunes only the head (`freeze=23`) → **V0**, deployed FP16. `apply_qat_0iter.py` then applies calibrate-only INT8 QAT with **zero** training iterations → **V1**. V1 is the sweep's zero point: what INT8 costs before any fine-tuning recovers it. Start at `RESULTS.md`; `README.md` has the 3-step method. |
| `layer_sweep/` | Step 3. One layer quantised and fine-tuned at a time, all 24 candidates, everything else frozen FP16. Start at **`RECAP.md`**. `results_master_malmo_h100.csv` is the source of every number. `run_sweep_malmo.sh` / `run_layer_malmo.sh` / `queue_deploy_malmo.sh` / `make_metrics_malmo.py` are the runners; `layer_00/` … `layer_23/` hold per-layer provenance, metrics and logs. |
| `week8_layer_sweep_plots.ipynb` | The three figures, executed with outputs embedded so they render on GitHub without re-running. |
| `figures/` | Those figures as PNGs. |

## Results in one place

| | mAP50-95 | note |
|---|--:|---|
| V0 — frozen baseline, FP16 | 0.5313 | the bar a quantised layer has to clear |
| V1 — frozen + 0-iter QAT, INT8 | 0.4892 | zero point; whole-model quantisation costs 0.0421 |
| best sweep layer (`model.8`) | 0.6675 | recovers 62.4% of the V1 → FP32 gap |
| worst (`model.22`) | 0.4874 | *below* V1 — six epochs did not repay quantising it |

Four layers — `model.22`, `model.20`, `model.0`, `model.23`, i.e. the stem and the head —
land below the V0 FP16 line even with training. Mid/late backbone (`model.4`–`model.10`,
plus `model.13`) carries essentially all the recoverable accuracy.

**Latency did not move**: 0.7396–0.7656 ms across all 18, a 3.5% spread. That is not
evidence that layer choice is latency-neutral in general — it is a property of this
design. No single layer exceeds 9.1% of the model's conv MACs, so even a perfect 2×
speedup on the largest caps at ~4.5%, and a lone INT8 layer in an FP16 graph pays
quantize/dequantize reformats at its boundary that cost about what the INT8 kernel saves.
For scale, CUDA graphs alone are worth 224 µs (23%) on the same engines — launch overhead
dominates everything layer choice can do at batch 1.

## Caveats that carry forward

1. **Single-layer QAT is a ranking experiment, not a deployment candidate.** The best
   layer is still 0.107 mAP50-95 below the FP32 baseline.
2. **The ranking is measured on the frozen model** (mAP 0.49–0.67), not the fully-trained
   one where the PTQ/QAT ladder lives (0.75–0.77). Whether it transfers is untested.
3. **The sweep's latency and the ladder's latency are not on one axis** — the 18 engines
   were timed on an H100 with CUDA graphs, the ladder on an A100 with none. Figure 2 keeps
   them in separate panels for that reason.
4. **Engines have been cleared** to reclaim disk. Every sha256 and full build configuration
   survives in the committed `*.provenance.json` sidecars, and each engine is rebuildable
   from the retained `best_qat.onnx`.
