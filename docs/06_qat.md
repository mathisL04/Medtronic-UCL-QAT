# Quantisation-Aware Training

QAT fine-tunes the converged V1 baseline with fake-quant nodes so the INT8 scales
are *learned*, not just calibrated (PTQ). It is the V5 stage.

## Where QAT runs, where it is measured

```text
train:    PyTorch  -- fake-quant / Q-DQ nodes, optimiser + labelled data
convert:  PyTorch -> ONNX (Q/DQ) -> TensorRT INT8 engine
measure:  TensorRT -- accuracy (mAP) and, above all, latency
```

QAT is a training-time technique: it runs in PyTorch, never on TensorRT. TensorRT
is inference-only -- its job is to fold the learned scales into a real INT8 engine
that we then benchmark. We do **not** "train on TensorRT".

## Fine-tune parameters

Fine-tuning is training with a small learning rate: the weights are already
converged, so a high LR would retrain rather than adapt. Use a low LR that decays
to near-zero -- "small" is the start point, "decay" is the shape (not alternatives).

```text
lr0     ~1e-3 or lower   ~1% of the full-training lr0 (0.01); NOT yet set in the
                         script -- it currently inherits Ultralytics' 0.01
lrf     keep default     decay tail (cosine/linear to near-zero) -- wanted
epochs  by convergence   short fine-tune; choose when mAP stops improving, NOT 2^n
batch   16               power-of-two is a mild GPU-efficiency nicety, not a rule
imgsz   640              LOCKED: multiple of 32 (YOLO stride) and the engine's
                         fixed input -- do not change
amp     False            fake-quant scales are FP32; FP16 autocast corrupts them
                         (deviation from V1, which trained with amp)
workers 0 for the smoke  avoids the fork/overcommit OSError-12 on the shared box
layers  unchanged        QAT wraps existing layers with fake-quant; it never
                         adds/removes any. Architecture is fixed.
```

Power-of-two note: it matters *mildly* for batch (memory/tensor-core alignment)
and not at all for epochs. The only hard sizing rule is imgsz = multiple of 32,
which 640 already satisfies.

## Planned content (remaining)

- Multi-epoch fine-tune (needs an mto-based best-epoch checkpoint callback; the
  smoke script keeps only the final EMA because save_model is disabled)
- Export path: fake-quant model -> ONNX Q/DQ -> TensorRT INT8 (unwritten, untested)
- TensorRT INT8 accuracy + latency comparison vs V2/V3/PTQ
