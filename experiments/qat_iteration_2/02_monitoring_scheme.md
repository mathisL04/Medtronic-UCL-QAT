# QAT Monitoring / Observability Scheme (design — for sign-off)

Goal: for every future sweep run, capture *what is happening inside the model* — cheaply, per-run,
reproducibly — so we can see how each hyperparameter moves weights, scales, quantization error, and
the deployed engine. **Lightweight**: per-epoch scalars are free; expensive captures (histograms,
per-layer error, engine profile) run at a low cadence or once.

Nothing here changes Phase-1 scripts. Proposed as **attachable callbacks + hooks** so a sweep run
adds them via `trainer.add_callback(...)` (same mechanism `train_qat.py` already uses for
`save_best_qat`) — no edit to the training core required.

---

## 1. What to capture

### 1a. Per-layer (the model internals)
| Signal | What it tells us | How captured | Cost |
|---|---|---|---|
| **Weight distribution** (per layer, histogram) | is a layer's weight range drifting / clipping under QAT | snapshot weight tensors on a fixed sample of layers | every K epochs |
| **Activation distribution** (per layer, histogram) | are activations well-matched to the frozen per-tensor scale (saturation/underuse) | `forward hook` on sampled layers during 1 fixed eval batch | every K epochs |
| **Learned amax / scale** (per quantizer) | the actual INT8 grid each quantizer uses; drift over epochs | iterate `TensorQuantizer._amax` buffers | **every epoch** (cheap) |
| **Weight shift** (‖ΔW‖ vs epoch 0, per layer) | which layers QAT actually moves (where the recovery happens) | L2 of (W_epoch − W_0), relative to ‖W_0‖ | every epoch |

### 1b. Training curves (per epoch — all cheap, from Ultralytics `results.csv` + callback)
- **loss** (box/cls/dfl + total), **mAP50 / mAP50-95**, **LR** schedule, **scale-drift summary**
  (mean/max |Δamax| across all 608 quantizers this epoch).

### 1c. Quantization health
| Signal | What it tells us | How captured |
|---|---|---|
| **INT8 vs float per layer** | coverage map (post-build) | engine inspector (reuse `repartition.py`) |
| **Per-layer quant error (SQNR)** | *which layers' quantization costs the most accuracy* — the key diagnostic | diagnostic pass: for a fixed batch, compare each layer's FP32 output vs its fake-quant output, `SQNR = 10·log10(‖x‖² / ‖x − x_fq‖²)` (dB) | every K epochs |

> **SQNR** (signal-to-quantization-noise ratio) is the standard per-layer quantization-quality number:
> high dB = quantization barely perturbs that layer; low dB = that layer is where precision is lost.
> This directly answers "which layers should stay float" and validates the attention/head findings.

### 1d. Engine-side (post-build, per candidate engine)
| Signal | How captured |
|---|---|
| **Per-layer precision** (INT8/FP16/FP32) | `repartition.py` on the engine (already built) |
| **Per-layer latency contribution** | `trtexec --loadEngine=… --dumpProfile --exportProfile=prof.json` (or `IEngineInspector`) |
| **Reformat locations** | filter profile/inspector for "Reformatting CopyNode" → where INT8↔float seams cost time |

---

## 2. How to capture — the mechanisms

```text
QATMonitor (new: experiments/qat_iteration_2/monitoring/qat_monitor.py)
  ├─ on_fit_epoch_end callback   → per-epoch: dump amax of all quantizers, ‖ΔW‖ per layer,
  │                                 append loss/mAP/LR row  (reads trainer.metrics + results.csv)
  ├─ every-K-epochs block        → register forward hooks on sampled layers, run 1 fixed eval
  │                                 batch → weight & activation histograms + per-layer SQNR
  └─ writes everything under the run folder (below)

Engine profiler (new: monitoring/profile_engine.py)
  └─ post-build: repartition.py (precision) + trtexec --dumpProfile (per-layer latency + reformats)
```

- **Quantizer iteration** (amax): `for n,m in model.named_modules(): if isinstance(m, TensorQuantizer): m._amax`.
- **Weight shift**: cache `W_0` state_dict at epoch 0; each epoch compute per-layer `‖W−W_0‖/‖W_0‖`.
- **Hooks**: `module.register_forward_hook` on a fixed sampled layer set (e.g. one per stage +
  both attention blocks + head) — capture input/output tensors on ONE batch, histogram, remove hooks.
- **SQNR**: temporarily read each quantizer's pre- and post-quant tensor on that same batch.
- All heavy captures gated by `MONITOR_EVERY_K` (default e.g. 5) so overhead stays ~1 eval batch / K epochs.

---

## 3. Where to store — per-run folder (reproducible)

```text
experiments/qat_iteration_2/sweeps/<run_name>/
├── monitoring/
│   ├── scales.csv            epoch, quantizer_name, amax                     (per epoch)
│   ├── weight_shift.csv      epoch, layer, rel_l2_shift                      (per epoch)
│   ├── curves.csv            epoch, loss_total, box, cls, dfl, mAP50, mAP50-95, lr, amax_drift_mean/max
│   ├── sqnr.csv              epoch, layer, sqnr_db                           (every K)
│   ├── hist/                 weight_<layer>_ep<K>.npz, act_<layer>_ep<K>.npz (every K)
│   └── engine/
│       ├── precision.json    per-layer INT8/FP16/FP32  (repartition.py)
│       └── profile.json      per-layer latency + reformats (trtexec)
├── qat_modelopt_state_best.pt   (the artifact — as today)
└── run_config.json          all hyperparameters for this run (the sweep axis + fixed set)
```
CSV + npz = git-friendly, diffable, re-plottable. Raw tensors are NOT stored (only histograms/scalars).

---

## 4. How to visualize — `monitoring/plot_run.py` (post-run, matplotlib)

| Plot | From | Reads |
|---|---|---|
| **Training curves** (loss, mAP50-95, LR — 3-panel) | curves.csv | per-run |
| **Scale drift** (|Δamax| mean/max vs epoch; + heatmap quantizer×epoch) | scales.csv | per-run |
| **Weight-shift bar** (rel ‖ΔW‖ per layer, final) — *where QAT worked* | weight_shift.csv | per-run |
| **Per-layer SQNR bar** (dB, colored by INT8/float) — *where precision is lost* | sqnr.csv + precision.json | per-run |
| **Weight/activation histograms** (small multiples, sampled layers) | hist/ | per-run |
| **Engine precision map** (backbone/neck/head INT8/FP16/FP32 — reuse the repartition view) | precision.json | per-run |
| **Engine latency profile** (per-layer ms; reformats flagged) | profile.json | per-run |
| **Cross-run overlay** (mAP50-95 vs epoch for all runs in a sweep; final mAP vs swept value) | all curves.csv | per-sweep |

Output: `sweeps/<run_name>/monitoring/plots/*.png` + a one-page `report.md` per run, and a
`sweeps/<sweep_name>/summary.md` overlaying the runs.

---

## 5. Lightweight guarantees
- Per-epoch cost = iterate 608 buffers + one weight-norm pass ≈ **negligible** (<1 s/epoch).
- Heavy cost = **1 eval batch every K epochs** for hists + SQNR (K=5 → ~7 extra batches over a 35-epoch run).
- Engine profile = **once per built engine** (seconds).
- Zero new dependencies (torch, numpy, matplotlib, trtexec — all present).
- Fully reproducible: everything keyed to `run_config.json` (hyperparameters + seeds) in the run folder.

## 6. Open choices for your sign-off
1. `MONITOR_EVERY_K` cadence (proposed **5**).
2. Sampled-layer set for hists/SQNR: **all 253 quantized layers** (heavier) vs a **representative ~12**
   (one per stage + 2 attention + head) — proposed the representative set, with full-SQNR on demand.
3. Whether engine profiling runs for **every** run or only the **best** run per sweep (proposed: best
   only, to keep it light).
