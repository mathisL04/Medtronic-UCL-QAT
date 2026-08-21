# Environment and Machine Access

Where the work runs, how to get on the machines, and which virtual environment
each stage needs. This is the first thing to read when picking the project up:
almost every reproduction step below assumes one of these environments.

---

## The three machines

Work moved between three UCL boxes as the project went on. Which machine a number
came from matters: **latency is not comparable across them**, and several results
in this repo carry that caveat explicitly.

```text
cork.ee.ucl.ac.uk     Tesla V100-PCIE-16GB     dataset build + baseline training (stage 1)
geneva.ee.ucl.ac.uk   A100-SXM4-80GB  x4       ONNX -> TensorRT ladder, QAT, iteration 2 (stages 2-7)
malmo.ee.ucl.ac.uk    H100 NVL        x4       week-8 frozen baseline + per-layer sweep (stage 8)
```

The ~4 GB generated YOLO dataset lives on Cork storage and is not in git:

```text
/home/zcemml1/medtronic_qat_data/datasets/sanoscience_yolo_full_nonexpert_stereo
```

## Getting on

All three sit behind the UCL EEE gateway, so it is two hops. The flags matter on a
UCL-managed machine: without them ssh tries GSSAPI and public-key first and can fail
before it ever offers to ask for a password.

```bash
ssh -4 \
  -o GSSAPIAuthentication=no \
  -o PubkeyAuthentication=no \
  -o PreferredAuthentications=password \
  zcemml1@ssh.ee.ucl.ac.uk
```

Then the target box, same flags:

```bash
ssh -4 -o GSSAPIAuthentication=no -o PubkeyAuthentication=no \
       -o PreferredAuthentications=password zcemml1@cork.ee.ucl.ac.uk
# or geneva.ee.ucl.ac.uk / malmo.ee.ucl.ac.uk
```

## Virtual environments

There are four, and they are **not** interchangeable. The split is forced by a real
dependency conflict, not by preference: `nvidia-modelopt` 0.29 cannot export this
model to Q/DQ ONNX under any torch version, and the fix (>= 0.31) needs Python >= 3.10,
while 0.29 was the last Python 3.9 build. So QAT training and export had to move to a
3.11 environment while the TensorRT tooling stayed on 3.9.

| venv | Python | Holds | Used for |
|---|---|---|---|
| `~/venvs/medtronic-qat-p311` | 3.11.13 | `nvidia-modelopt 0.33.1`, `torch 2.7.0+cu128`, `onnx 1.17.0`, `numpy 1.26.4`, `ultralytics 8.4.90`, `pycocotools 2.0.11` | QAT fine-tune and Q/DQ ONNX export |
| `~/venvs/medtronic-trt` | 3.9.25 | `TensorRT 10.16.1.11` | every engine build, every mAP evaluation, every latency measurement |
| `~/venvs/medtronic-qats` | 3.9.25 | `nvidia-modelopt 0.29.0`, `torch 2.8.0` | stage-1 training, dataset build, notebooks |
| `~/venvs/medtronic-qat` | 3.9.25 | `nvidia-modelopt 0.29.0` | superseded, kept only to reproduce pre-0.33 runs |

Rule of thumb: **anything that touches a TensorRT engine runs under `medtronic-trt`.**
Anything that trains or exports QAT runs under `medtronic-qat-p311`.

`medtronic-trt` also carries a root-less `trtexec` wrapper at
`~/venvs/medtronic-trt/bin/trtexec`, which points the RPM binary at the venv's
TensorRT libraries. Every `trtexec` figure in this repo came from it.

```bash
source ~/venvs/medtronic-trt/bin/activate      # or the p311 / qats venv
cd ~/medtronic_qat/Medtronics-UCL-QAT
```

## The idle-GPU rule

**Every latency number in this project was measured on a verified-idle GPU, and any
that was not says so.** This is not a formality. The 14 July FP32 baseline measured
16.928 ms median; re-run on a verified-idle GPU it measured 8.642 ms, a 49% correction
with identical timing code. The machine changed, not the benchmark.

Idle means idle **for compute**. These boxes always have Xorg holding a graphics
context, so the check uses `nvmlDeviceGetComputeRunningProcesses` and ignores
graphics contexts, otherwise no GPU ever looks free. The benchmark scripts refuse to
start on a contended device and snapshot utilisation, memory and process count around
every repeat, so a contended run is distinguishable from a clean one after the fact
rather than silently averaged in.

These are shared machines and contention is routine, not exceptional: on 2026-07-17 a
re-run was refused because another user held ~47 GB and 100% utilisation across all
four A100s.

`DEVICE` is strict everywhere and is recorded in the provenance sidecar. A missing
`DEVICE` is an error, never a silent fallback to GPU 0.

## Disk

Home is on NFS and the quota is small relative to what training churns. Training and
engine builds write to local scratch (`/tmp`) and only durable artifacts are copied
back. Engines, ONNX files and checkpoints are gitignored throughout; their
`*.provenance.json` sidecars are committed instead and carry the sha256 plus the full
build configuration, so any cleared artifact stays identifiable and rebuildable.

## The devcontainer

`.devcontainer/` defines a `nvcr.io/nvidia/tensorrt:26.05-py3` image with `--gpus=all`
and the workspace bind-mounted at `/workspace`, paired with
`configs/sanoscience_yolo_local.yaml` for the container-side dataset path.

**It is the intended environment, not the one the results came from.** Every number in
this repository was produced by the venvs above, running directly on Cork, Geneva or
malmo. The container has not been used to reproduce them, so treat it as a starting
point that still needs validating rather than a known-good path.
