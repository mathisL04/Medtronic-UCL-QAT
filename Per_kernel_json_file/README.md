# Kernel-Level Latency Profiling — QAT vs PTQ

Per-kernel trtexec profiles for comparative latency analysis of the QAT (batch32)
and PTQ INT8 TensorRT engines.

## Files
- `qat_batch32_per_kernel_profile.json` — QAT engine per-kernel timings (245 kernels)
- `ptq_int8_per_kernel_profile.json` — PTQ INT8 engine per-kernel timings (189 kernels)

## Schema (per kernel)
`name`, `timeMs`, `averageMs` (per-inference, use for summing), `medianMs`, `percentage`.
Element [0] is a `{"count": N}` header (profiled iteration count) — skip it.

## Key finding
QAT's latency overhead vs PTQ is the conv+SiLU fusion break: QAT's explicit Q/DQ nodes
prevent TensorRT from fusing conv+SiLU (which PTQ does), so SiLU activations spin out into
standalone kernels. Analysis in the profiling notebook.

Measured with trtexec (TensorRT 10.16.1.11), A100-80GB, exclusive idle GPU.
