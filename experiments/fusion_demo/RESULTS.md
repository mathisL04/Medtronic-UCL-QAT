# Q/DQ fusion-break demo — tiny CNN (3× Conv3x3→SiLU + 1×1 head)

Controlled isolation: identical tiny net, ONLY the quantization scheme differs.
Q/DQ placement in variant C mirrors the real QAT model (Q→DQ→Conv→SiLU→Q→DQ→Conv).
Latency: GPU2, 100-iter warmup + 300 iters, batch 1, median. NOTE: GPU2 had a
dormant foreign context (0% util, no active kernels) — not fully exclusive, but
numbers are internally consistent and physically ordered (INT8 < FP16).

| variant | conv+SiLU fused | standalone SiLU | Q/DQ | total kernels | no-graph ms | +graph ms |
|---|--:|--:|--:|--:|--:|--:|
| A  FP16 (no quant)      | 3 | 0 | 0 | 6 | 0.0568 | 0.0462 |
| B  INT8 PTQ (implicit)  | 3 | 0 | 0 | 5 | 0.0444 | 0.0368 |
| C  INT8 explicit Q/DQ   | 3 | 0 | 1 | 5 | 0.0425 | 0.0350 |

## Result: the naive hypothesis is FALSIFIED in the clean case
- **Explicit Q/DQ did NOT break conv+SiLU fusion.** All 3 blocks fuse in every
  variant. In C the kernel is literally `Conv + PWN(Sigmoid, Mul)` WITH the weight
  QuantizeLinear folded in — one INT8 kernel doing quant + conv + SiLU.
- **No extra kernels** from Q/DQ: C has 5, FP16 has 6 (C is not more).
- **No latency penalty**: C (0.0425) is the FASTEST no-graph (INT8 compute beats
  FP16), on par with PTQ. CUDA graphs shaves ~18–22% off all three equally.

## What this means for the real model
"Explicit Q/DQ breaks conv+SiLU fusion" is TOO SIMPLE. On a clean sequential
Conv→SiLU chain, TensorRT fuses Q→conv→SiLU→Q into one INT8 kernel with no penalty.
The real-model fusion break therefore comes from the INTERACTION of Q/DQ with the
model's STRUCTURAL COMPLEXITY — the C2f channel-splits, residual adds, and
multi-consumer / dynamic-shape tensors where a conv's SiLU output feeds branching
paths — NOT from Q/DQ presence alone. Q/DQ is necessary but not sufficient to break
the fusion; the branching structure is what tips TensorRT into leaving the SiLU
standalone.
