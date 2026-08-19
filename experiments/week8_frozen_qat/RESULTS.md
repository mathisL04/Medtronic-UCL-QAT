# Week 8 — frozen baseline + 0-iteration QAT

## The experiment
1. **Frozen baseline (V0)**: COCO yolo26n.pt, freeze backbone+neck (`freeze=23`), train ONLY the
   head on the surgical data (1 class), fixed 50 epochs, no early stop. Same data/paths as usual.
2. **0-iteration QAT**: apply the standard modelopt QAT quantization (mtq.quantize + max-calibration
   on 128 episode-diverse frames, seed 42) to that frozen baseline with **ZERO training iterations**
   — i.e. explicit-Q/DQ quantization, no fine-tune ("PTQ via the QAT path").
3. Deploy: Q/DQ ONNX -> TensorRT INT8 -> accuracy (full val, pycocotools conf 0.001) + latency
   (idle A100, kernel-median).

## Results
```
model                          mAP50    mAP50-95   kernel latency
frozen baseline (float)        0.756    0.546      —
frozen + 0-iter QAT (INT8)     0.6906   0.4892     1.503 ms
Δ (INT8 quantization cost)    -0.065   -0.057
```

## Reading
- Frozen baseline lands at mAP50-95 **0.546** (vs full-trained baseline 0.782): freezing backbone+neck
  costs ~0.24 mAP50-95 — the COCO backbone isn't adapted to surgical imagery.
- 0-iteration QAT (INT8, no training) drops another **-0.057** (0.546 -> 0.4892): pure quantization
  loss with no fine-tuning to recover it, exactly like PTQ.
- Latency 1.503 ms kernel (standard QAT INT8 build; ~same as our other QAT engines).

## Artifacts (this folder)
- `apply_qat_0iter.py`  — the calibrate-only step (Ultralytics can't do epochs=0)
- `qat0/qat0_provenance.json`, `qat0/best_qat_int8.engine.map_full.json`, latency log
- (ONNX / engine gitignored — rebuildable)

Reused unchanged: export_qat_onnx.py, build_tensorrt_int8_qdq.py, benchmark_latency_trt.py,
evaluate_engine_map.py (+ MODEL_PATH/BASE_MODEL env overrides for the frozen source).

## UPDATE — both models DEPLOYED (ONNX -> TensorRT -> engine mAP + latency)
Each model deployed and benchmarked on an idle A100 (full val, pycocotools conf 0.001; kernel-median latency).

```
model                    precision  engine mAP50  engine mAP50-95  kernel latency  size
frozen baseline           FP16        0.7419        0.5313          1.253 ms        7.0 MB
frozen + 0-iter QAT       INT8        0.6906        0.4892          1.503 ms        4.2 MB
Δ (QAT INT8 vs FP16)                 -0.051        -0.042          +0.250 ms       -2.8 MB
```

Verdict: 0-iteration QAT (INT8) on the frozen model is WORSE on both axes vs FP16 — it loses
0.042 mAP50-95 (no fine-tune to recover quant loss) AND is 0.25 ms slower (explicit-Q/DQ INT8 has
more kernels; launch-bound A100 favours FP16). INT8's payoff needs edge/DLA HW or compute-bound
models, not this launch-bound case. Frozen-baseline FP16 engine mAP 0.531 ≈ PyTorch 0.546 (normal
export drop). Deploy path reuses export_onnx.py + build_tensorrt_engine.py (MODEL_PATH/ONNX_PATH
env overrides) + the shared eval/benchmark scripts.
