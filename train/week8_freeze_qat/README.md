# New QAT test framework 2 (Week 8) — frozen-baseline + QAT

New experiment: does a **frozen-backbone/neck** baseline (only the detection head
trained) behave differently under QAT than the fully-trained baseline?

## The 3 steps
1. **Frozen baseline** — COCO-pretrained `yolo26n.pt`, freeze backbone+neck
   (`model.0-22`), fine-tune ONLY the head (`model.23`) on the surgical dataset
   (1 class). → export ONNX → TensorRT → latency + accuracy.
   `train_frozen_baseline.py`  (Ultralytics `freeze=23`).
2. **QAT on the frozen baseline** — standard QAT (patience early-stop, same
   framework as `scripts/train/train_qat.py`) starting from step-1's `best.pt`.
   → TensorRT → latency + accuracy.
3. **Compare** QAT vs the frozen baseline (accuracy + latency).

## Data (same as usual)
`/home/zcemml1/medtronic_qat_data/datasets/sanoscience_yolo_full_nonexpert_stereo`
20,756 train / 6,449 val, split by episode, 1 class `surgical_tool`.

## Run (Geneva, idle A100)
```bash
cd ~/medtronic_qat/Medtronics-UCL-QAT
source ~/venvs/medtronic-qat-p311/bin/activate     # training venv
# Step 1: frozen baseline (DEVICE required, WORKERS=0 fork-guard)
nohup env DEVICE=2 python -u train/week8_freeze_qat/train_frozen_baseline.py \
  > runs_week8/train_frozen_head.log 2>&1 &
# monitor:  tail -f runs_week8/train_frozen_head.log   |   watch -n1 nvidia-smi
```
Output: `runs_week8/week8_frozen_head/weights/best.pt`.

Step 2 (QAT) then points `train_qat.py` at that `best.pt` and runs with patience.
