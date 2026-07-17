# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Medtronic x UCL summer research on optimising a surgical-tool detector for embedded GPU deployment. The pipeline is linear and staged; each stage is documented in `docs/`:

```
Open-H Sanoscience videos → segmentation-based auto-labelling → YOLO26n training
→ ONNX export → TensorRT FP16 → TensorRT INT8/PTQ → QAT → accuracy/latency comparison
```

Stages 1–2 are done (baseline `models/yolo26n_sanoscience_full_left/best.pt`, mAP50 0.934 / mAP50-95 0.782). Everything from ONNX export onward is unwritten — `docs/02`–`docs/05` are placeholders listing planned content, not records of work done. Do not describe those stages as complete.

## Scripts are configured by editing constants, with env-var overrides for run-time knobs

Every script in `scripts/` is a flat top-to-bottom program with a `# Settings` block of hardcoded constants and no argparse. **Structural** configuration — absolute paths, model/dataset locations, epochs, batch size — is hardcoded; to change it you edit the constants.

`benchmark_latency.py` is the exception that matters: **`DEVICE` has no default and the script aborts if it is unset.** This is deliberate. The earlier benchmark `benchmark_fp32_geneva_ram_stable.py` defaults `DEVICE` to `0`, so any run outside the exact runbook silently used a different physical GPU than the one reported — which is how a headline latency number became unattributable. Making `DEVICE` strict removes that failure mode: no GPU specified, no run.

Beyond that, the benchmark, evaluation, dataset-build, and subset scripts read a handful of **run-time knobs from environment variables**, each with a hardcoded default so a bare `python scripts/<name>.py` still runs. Override inline, e.g. `DEVICE=3 BENCHMARK_REPEATS=10 python scripts/benchmark_latency.py`. These set the device, run parameters, and workload partitioning — never the dataset or model paths, which stay hardcoded:

- `build_sanoscience_yolo_full_cork.py` — `START_EPISODE_INDEX`, `END_EPISODE_INDEX`, `SHARD_NAME`: partition the labelling workload across shards (`SHARD_NAME` also names the shard's temp-frames directory).
- `benchmark_latency.py` — `DEVICE`, `SEED`, `IMG_SIZE`, `CONF`, `BENCHMARK_REPEATS`, `WARMUP_IMAGES`, plus the `GATE_UTIL_THRESHOLD` / `GATE_SAMPLES` / `GATE_INTERVAL_S` gating thresholds.
- `evaluate_100_random_accuracy.py` — `DEVICE`.
- `make_val100_subset.py` — `N_IMAGES`, `SEED`, `OVERWRITE`.

There are no tests, no linter config, and no packaging files (`requirements.txt`, `pyproject.toml`). Dependencies are installed ad hoc into a venv: `ultralytics`, `torch`, `opencv-python`, `huggingface_hub`, `numpy`, `pyyaml`, `pandas`, `imageio_ffmpeg`.

## Two execution environments, forked scripts

Scripts exist in near-duplicate variants because paths are hardcoded per environment. Before editing, confirm which variant is in play — a fix in one usually needs mirroring into the others.

- **Cork** (UCL EEE GPU server, Tesla V100): paths under `/home/zcemml1/medtronic_qat_data/...`, `*_cork.py` variants, `imageio_ffmpeg.get_ffmpeg_exe()` for the ffmpeg binary, `configs/sanoscience_yolo_cork.yaml`. This is where the dataset and baseline were actually produced.
- **Local devcontainer** (`.devcontainer/`, `nvcr.io/nvidia/tensorrt:26.05-py3`, `--gpus=all`): paths under `/workspace`, non-cork variants, system `ffmpeg`, `configs/sanoscience_yolo_local.yaml`. This container is the intended home for the TensorRT stages.

`build_sanoscience_yolo_full_resume.py` hardcodes `START_EPISODE_INDEX = 606` — a one-off resume after a crash, not a general entry point.

## Cork workflow

```bash
# gateway, then cork
ssh -4 -o GSSAPIAuthentication=no -o PubkeyAuthentication=no -o PreferredAuthentications=password zcemml1@ssh.ee.ucl.ac.uk
ssh -4 -o GSSAPIAuthentication=no -o PubkeyAuthentication=no -o PreferredAuthentications=password zcemml1@cork.ee.ucl.ac.uk

cd ~/medtronic_qat/Medtronics-UCL-QAT
source ~/venvs/medtronic-qats/bin/activate
```

Cork is a shared server, so long jobs run detached under `nohup` and `nice`:

```bash
# dataset build — one worker per non-overlapping episode range
nohup nice -n 10 env START_EPISODE_INDEX=0 END_EPISODE_INDEX=400 SHARD_NAME=s0 \
  python -u scripts/build_sanoscience_yolo_full_cork.py \
  > /home/zcemml1/medtronic_qat_data/runs_sanoscience/logs/build_cork_s0.log 2>&1 &

# training
nohup python -u scripts/train_sanoscience_yolo_full_cork.py \
  > /home/zcemml1/medtronic_qat_data/runs_sanoscience/logs/train_yolo26n_full_cork.log 2>&1 &

# monitoring
pgrep -af build_sanoscience_yolo_full_cork.py
tail -f /home/zcemml1/medtronic_qat_data/runs_sanoscience/logs/train_yolo26n_full_cork.log
watch -n 1 nvidia-smi
```

The 1602 episodes were sharded across workers s0 (0–399), s1 (400–799), s2 (800–999), s2b (1000–1199), s3b (1200–1601). Sharding works because labelling is CPU/ffmpeg/network-bound, not GPU-bound; each shard gets its own `_tmp_frames_<SHARD_NAME>` directory so workers don't collide. `WORKERS = 8` in the training script is CPU dataloader workers, not parallel GPU jobs.

## Labelling approach

The dataset ships paired colour and segmentation MP4s. `segmentation_to_boxes()` derives boxes without any human annotation: quantise segmentation RGB to a 16-step grid (absorbs MP4 compression noise), match the two tool greens `(96, 240, 0)` and `(96, 240, 112)`, morphological open, connected components, filter by `MIN_AREA`/`MIN_BOX_SIZE`, emit normalised YOLO boxes. Everything is class 0 `surgical_tool`.

Frames come out of ffmpeg one at a time via `select=eq(n\,IDX)` into a temp PNG that is unlinked immediately after `cv2.imread` — this keeps disk usage flat over a multi-GB build. Videos are stereo side-by-side; `split_stereo()` takes the left half only (`USE_VIEW = "left"`), so the baseline is left-view-only. Frames with no boxes are dropped (`SAVE_EMPTY_IMAGES = False`), meaning the dataset has no true negatives.

Train/val split is by episode (first 80% of episodes train), not by frame — this is deliberate, since frames within an episode are highly correlated. Preserve this if you rebuild.

## What is and isn't in git

Generated datasets, `runs*/`, HF caches, and all weight/export formats (`*.pt`, `*.onnx`, `*.engine`, `*.trt`) are gitignored. The ~4 GB YOLO dataset lives only on Cork at `/home/zcemml1/medtronic_qat_data/datasets/sanoscience_yolo_full_nonexpert_stereo`.

Two files are deliberate exceptions, force-added past the ignore rules: `models/yolo26n_sanoscience_full_left/best.pt` (the baseline checkpoint, hand-copied from the Cork run) and `yolo26n.pt` (pretrained init). Committing any other checkpoint needs `git add -f` and probably shouldn't happen.

`reports/` holds curated metrics copied out of a run; `runs_sanoscience/` and `runs_utenn/` are raw untracked Ultralytics output. `scripts/archive/` holds superseded work — `utenn/` is an earlier abandoned dataset, `prototypes/` and `exploration/` are dead ends kept for reference. Don't build on archived scripts.

## Documentation convention

`docs/01`–`docs/05` are the project record, one file per pipeline stage, and README.md summarises status across all of them. When a stage is completed, fill in its doc and update the "Current Status" / "Next Steps" sections of README.md — results are reported as fenced plain-text blocks of `key: value` lines, not tables or prose.
