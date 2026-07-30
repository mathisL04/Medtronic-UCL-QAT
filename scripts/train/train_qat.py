import os
import random
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import torch


# -----------------------------
# Settings
# -----------------------------
# QAT fine-tune of the V1 baseline checkpoint (docs/06). This is the SMOKE-RUN
# form: 1 epoch by default, to prove plumbing, not to produce accuracy.
#
# Invoked through scripts/train/qat_run.sh, which supplies PROJECT (local scratch, so
# per-epoch checkpoint churn stays off the 50 GB NFS quota) and copies the final
# artifacts back to NFS on exit.
MODEL_PATH = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "models/yolo26n_sanoscience_full_left/baseline/best.pt"
)
DATA_YAML = Path(
    "/home/zcemml1/medtronic_qat_data/datasets/"
    "sanoscience_yolo_full_nonexpert_stereo/sanoscience_yolo.yaml"
)

# Warm-start frames for the INITIAL quantiser scales.
#
# WHY THIS EXISTS AT ALL. Calibration is usually a PTQ concept, and QAT does
# learn its scales from labelled data during fine-tuning. But mtq.quantize()
# inserts fake-quant nodes that need SOME initial range, and starting the
# fine-tune from observed activations rather than uninitialised ones is step 1
# of NVIDIA's documented QAT recipe. It is a warm start, not PTQ calibration:
# training then adapts these scales over all 20,756 train frames.
#
# WHY NOT THE TRAINING DATALOADER. It does not exist yet. Ultralytics builds it
# in _build_train_pipeline() at trainer.py:369, and quantisation has to happen
# in get_model() at :292 to precede ModelEMA at :373. So the warm start needs an
# independent image source.
#
# WHY NOT THE PTQ 500-FRAME SET. An earlier draft of this script reused it,
# arguing it removed calibration data as a variable between V4 and V5. That
# reasoning does not transfer: QAT adapts these scales across the whole train
# split, so a 500-frame warm start barely constrains the result, and reusing
# that set would add a dependency on data that is deleted and regenerable.
# Sampling the train split directly is simpler and equally valid.
TRAIN_IMAGES = Path(
    "/home/zcemml1/medtronic_qat_data/datasets/"
    "sanoscience_yolo_full_nonexpert_stereo/images/train"
)
N_CALIB = int(os.environ.get("N_CALIB", 128))
CALIB_SEED = int(os.environ.get("CALIB_SEED", 42))

DEVICE = os.environ.get("DEVICE")
if DEVICE is None:
    sys.exit("DEVICE is required (e.g. DEVICE=1). No GPU specified, no run.")

EPOCHS = int(os.environ.get("EPOCHS", 1))
# Early stopping (opt-in). 0 = OFF (default: run the full fixed EPOCHS -- unchanged,
# backward-compatible). >0 = stop after PATIENCE epochs with no validation improvement.
# In Ultralytics 8.4.90 the detection `fitness` weights are [0,0,0,1.0] = PURE mAP50-95
# (DetMetrics.fitness -> Metric.fitness, verified in ultralytics/utils/metrics.py), so
# BOTH this early-stopping and the save_best_qat checkpoint track mAP50-95 -- not the old
# 0.1*mAP50+0.9*mAP50-95 blend, and not mAP50.
PATIENCE = int(os.environ.get("PATIENCE", 0))
RUN_NAME = os.environ.get("RUN_NAME", "qat_smoke")
PROJECT = os.environ.get("PROJECT", "/tmp/zcemml1_qat")
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
BATCH = int(os.environ.get("BATCH", 16))

# Dataloader worker processes. Ultralytics defaults to 8, spawned via os.fork().
# Under this host's strict overcommit (vm.overcommit_memory=2), forking a
# multi-GB training process 8 times demands 8x that much committed address space
# at once, which fails as OSError [Errno 12] on a busy shared box -- independent
# of how much RAM is actually free. Default low; raise only when the box is
# quiet. workers=0 loads data in-process (slowest, but forks nothing).
WORKERS = int(os.environ.get("WORKERS", 4))

# Single-GPU, to match V1's training conditions. DDP would add a second variable
# and would also wrap the model in a way the quantised modules have not been
# checked against.
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from ultralytics import YOLO                            # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer   # noqa: E402
import modelopt.torch.quantization as mtq               # noqa: E402
import modelopt.torch.opt as mto                        # noqa: E402
from modelopt.torch.quantization.nn import TensorQuantizer   # noqa: E402


def count_quantizers(model):
    """Count the unit modelopt itself reports at insertion: TensorQuantizer
    instances. NOT quantised modules -- a QuantConv2d holds several quantizers
    (input, weight, ...), so a module count (e.g. 241) does not match the
    quantizer count (608) and silently looks like a partial reload."""
    total = sum(1 for m in model.modules() if isinstance(m, TensorQuantizer))
    with_amax = sum(1 for m in model.modules()
                    if isinstance(m, TensorQuantizer) and getattr(m, "_amax", None) is not None)
    return total, with_amax


# Number of TensorQuantizers inserted at quantise time, captured live so the
# reload gate asserts against what THIS run actually inserted, never a literal.
_inserted_quantizers = None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------
# Calibration forward loop
# -----------------------------
# Preprocessing copied verbatim from validate_engine_parity.py / int8_calibrator.py.
# Duplicated rather than imported because those are flat scripts. If this drifts
# from the inference path, the initial scales are computed on the wrong input
# distribution -- the same silent-corruption risk as the PTQ calibrator.
def letterbox(img, new_shape):
    h0, w0 = img.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    top, left = (new_shape - nh) // 2, (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess(img_bgr, imgsz):
    lb = letterbox(img_bgr, imgsz)
    im = lb[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(im, dtype=np.float32)[None] / 255.0


def make_forward_loop(frames, device):
    """Returns the callable mtq.quantize uses to observe activation ranges."""
    def forward_loop(model):
        model.eval()
        with torch.no_grad():
            for i, fp in enumerate(frames):
                img = cv2.imread(str(fp))
                if img is None:
                    raise RuntimeError(f"Calibration frame unreadable: {fp}")
                x = torch.from_numpy(preprocess(img, IMG_SIZE)).to(device)
                model(x)
                if (i + 1) % 32 == 0:
                    print(f"    calibrated {i + 1}/{len(frames)}", flush=True)
        model.train()
    return forward_loop


# -----------------------------
# HAZARD 1 -- quantise BEFORE ModelEMA snapshots the model
# -----------------------------
# Ultralytics' BaseTrainer._setup_train runs:
#
#     line 292:  ckpt = self.setup_model()      -> calls self.get_model()  (line 782)
#     ...
#     line 373:  self.ema = ModelEMA(self.model)
#
# get_model() is therefore 81 lines AHEAD of the EMA snapshot. Overriding it is
# what makes quantisation-before-EMA structural rather than incidental. Quantising
# from a callback such as on_pretrain_routine_end (line 384) would run AFTER line
# 373, leaving EMA holding an un-quantised copy whose state_dict keys no longer
# match the live model -- EMA.update() would then either KeyError or silently
# average the wrong tensors.
class QATTrainer(DetectionTrainer):
    """DetectionTrainer that returns an already-quantised model.

    Everything else in Ultralytics -- augmentation, loss, LR schedule, logging --
    is inherited untouched. mtq.quantize() swaps modules in place and preserves
    the module tree, so DetectionModel.loss() and .forward() still work.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)

        # Seeded, episode-diverse sample of the train split. One frame per
        # episode where possible: frames within an episode are highly correlated
        # (see CLAUDE.md), so N random frames would span far fewer episodes and
        # observe a narrower activation range than the count suggests.
        all_train = sorted(TRAIN_IMAGES.glob("*.jpg"))
        if not all_train:
            sys.exit(f"No training images found at {TRAIN_IMAGES}")
        by_episode = {}
        for p in all_train:
            by_episode.setdefault(p.name.split("_left_")[0], []).append(p)
        rng = random.Random(CALIB_SEED)
        episodes = sorted(by_episode)
        chosen = rng.sample(episodes, min(N_CALIB, len(episodes)))
        frames = [rng.choice(by_episode[e]) for e in sorted(chosen)]

        device = next(model.parameters()).device
        print(f"\n[QAT] warm-starting quantiser ranges from {len(frames)} train "
              f"frames across {len(chosen)} episodes (seed {CALIB_SEED})", flush=True)
        print("[QAT] quantising with INT8_DEFAULT_CFG BEFORE ModelEMA "
              "(get_model :292 vs ModelEMA :373)", flush=True)
        model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG,
                             make_forward_loop(frames, device))
        mtq.print_quant_summary(model)

        global _inserted_quantizers
        _inserted_quantizers, with_amax = count_quantizers(model)
        print(f"[QAT] inserted {_inserted_quantizers} TensorQuantizers "
              f"({with_amax} with scales) -- this is the reload gate target",
              flush=True)
        return model

    def save_model(self):
        # Ultralytics' save_model() torch.save's the live module, which pickles
        # modelopt's dynamically-generated QuantConv2d by class reference and
        # fails. The `save=False` override is NOT enough on its own: the trainer
        # calls save_model() unconditionally on the final epoch
        # (`if (self.args.save or final_epoch) and self.save_model()`, :563), so
        # the last epoch always tried to save and always crashed.
        #
        # We skip Ultralytics' checkpoint entirely and write the durable artifact
        # with mto.save() in the post-training block. NOTE: this also removes
        # Ultralytics' best-epoch tracking, so a MULTI-epoch fine-tune keeps only
        # the FINAL EMA, not the best. A real fine-tune needs an mto-based
        # checkpoint callback for best-tracking -- flagged in docs/06.
        return None


# -----------------------------
# HAZARD 3 -- prove the quantised structure can be rebuilt on load
# -----------------------------
# Ultralytics pickles the EMA MODULE OBJECT into the checkpoint ("ema": ema,
# with "model": None), so YOLO(path) reconstructs a DetectionModel its own way
# and does not know how to re-create modelopt's quantised modules. mto.save
# records the modelopt state alongside the weights; mto.restore replays it onto
# a freshly-built model, rebuilding the same structure before weights load.
def save_qat_state(model, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "qat_modelopt_state.pt"
    mto.save(model, p)
    print(f"[QAT] modelopt state saved: {p} ({p.stat().st_size / 1e6:.1f} MB)")
    return p


def verify_reload(state_path):
    """Rebuild from the ORIGINAL baseline, replay modelopt state, ASSERT parity.

    The gate is an assertion, not a print: the restored TensorQuantizer count
    must equal what quantise() inserted THIS run, or the script exits non-zero.
    That is what makes save->restore a real check -- a wrong count fails the run
    rather than scrolling past as a smaller-but-plausible number.
    """
    print("\n[QAT] verifying reload symmetry")
    fresh = YOLO(str(MODEL_PATH)).model
    restored = mto.restore(fresh, state_path)
    n_quant, with_amax = count_quantizers(restored)
    print(f"  restored TensorQuantizers: {n_quant} ({with_amax} with scales)")
    print(f"  inserted at quantise time: {_inserted_quantizers}")

    if _inserted_quantizers is None:
        sys.exit("[QAT] reload gate: inserted count unknown -- cannot verify.")
    if n_quant != _inserted_quantizers:
        sys.exit(
            f"[QAT] RELOAD GATE FAILED: restored {n_quant} quantizers but "
            f"{_inserted_quantizers} were inserted. Structure did not round-trip."
        )
    if with_amax == 0:
        sys.exit("[QAT] RELOAD GATE FAILED: no restored quantizer has a scale "
                 "(amax) -- the trained scales did not survive save/restore.")
    print(f"  [QAT] reload gate PASSED: {n_quant} quantizers, scales intact")
    return n_quant, with_amax


# -----------------------------
# Best-epoch checkpoint (multi-epoch fine-tune)
# -----------------------------
# save_model() is disabled (it cannot pickle modelopt's dynamic QuantConv2d), so
# Ultralytics keeps NO best.pt -- without this a multi-epoch run would retain only
# the FINAL EMA. This callback snapshots the modelopt state whenever validation
# fitness improves, giving a best-epoch artifact alongside the final one. For a
# 1-epoch smoke the best IS the final, so nothing changes there.
_best_fitness = float("-inf")


def save_best_qat(trainer):
    global _best_fitness
    fit = getattr(trainer, "fitness", None)
    if fit is None or fit <= _best_fitness:
        return
    _best_fitness = fit
    model = trainer.ema.ema if getattr(trainer, "ema", None) else trainer.model
    out = Path(PROJECT) / RUN_NAME / "qat_modelopt_state_best.pt"
    mto.save(model, out)
    print(f"[QAT] best fitness {fit:.5f} @ epoch {getattr(trainer, 'epoch', '?')} "
          f"-> saved {out.name}", flush=True)


# -----------------------------
# Run
# -----------------------------
print("=" * 60)
print(f"QAT fine-tune (smoke form)  run={RUN_NAME}  epochs={EPOCHS}")
print("=" * 60)
print("Model:   ", MODEL_PATH)
print("Data:    ", DATA_YAML)
print("Device:  ", DEVICE, "(single-GPU, matching V1)")
print("Project: ", PROJECT, "(local scratch -- churn stays off the NFS quota)")
print("Warm-start:", TRAIN_IMAGES, f"({N_CALIB} train frames, seed {CALIB_SEED})")

if not MODEL_PATH.exists():
    sys.exit(f"Baseline checkpoint not found: {MODEL_PATH}")
if not DATA_YAML.exists():
    sys.exit(f"Dataset YAML not found: {DATA_YAML}")

# -----------------------------
# HAZARD 2 -- amp=False, a DOCUMENTED deviation from V1's recipe
# -----------------------------
# Ultralytics enables AMP by default (trainer.py:453, `with autocast(self.amp)`).
# Fake-quantisation inserts explicit quantise/dequantise ops whose scales are
# FP32; running them inside an FP16 autocast region mixes dtypes at the quantiser
# boundary and is a known source of wrong scales and dtype errors.
#
# V1 trained WITH amp. So QAT differs from the baseline recipe in this one
# respect, and it must be stated in docs/06 rather than buried: any V5-vs-V1
# comparison carries "amp=False" as a known, deliberate difference.
overrides = dict(
    model=str(MODEL_PATH),
    data=str(DATA_YAML),
    epochs=EPOCHS,
    patience=PATIENCE,    # 0 -> Ultralytics disables (patience or inf); >0 -> early-stop on mAP50-95 plateau
    imgsz=IMG_SIZE,
    batch=BATCH,
    workers=WORKERS,
    device=int(DEVICE) if DEVICE.isdigit() else DEVICE,
    project=PROJECT,
    name=RUN_NAME,
    exist_ok=True,
    amp=False,            # <-- hazard 2
    val=True,
    plots=False,
    # HAZARD 3 -- Ultralytics' save_model() (trainer.py:682) does
    # torch.save({"ema": ema, ...}), pickling the live module. modelopt generates
    # QuantConv2d as a DYNAMIC class at runtime, which pickle cannot resolve by
    # attribute lookup -> PicklingError at end of every epoch. So Ultralytics'
    # native checkpointing is disabled; the durable checkpoint is written by
    # mto.save() in the post-training block instead, which stores the modelopt
    # recipe + state_dict and is reloadable via mto.restore (NOT YOLO(path)).
    save=False,
    # QAT fine-tune LR: the baseline is already converged, so start LOW (~1% of
    # V1's lr0=0.01) and let Ultralytics' scheduler decay it (lrf = decay tail).
    lr0=float(os.environ.get("LR0", 1e-3)),
    lrf=float(os.environ.get("LRF", 0.01)),
)

print("\n[QAT] amp=False (deviation from V1 -- see docs/06)")
trainer = QATTrainer(overrides=overrides)
trainer.add_callback("on_fit_epoch_end", save_best_qat)
trainer.train()

# -----------------------------
# Post-run: save + verify
# -----------------------------
run_dir = Path(PROJECT) / RUN_NAME
model = trainer.ema.ema if getattr(trainer, "ema", None) else trainer.model
state_path = save_qat_state(model, run_dir)
n_quant, n_scales = verify_reload(state_path)

prov = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "run_name": RUN_NAME,
    "epochs": EPOCHS,
    "device": DEVICE,
    "amp": False,
    "source_checkpoint": str(MODEL_PATH),
    "source_sha256": sha256(MODEL_PATH),
    "quant_config": "INT8_DEFAULT_CFG",
    "n_calib_frames": N_CALIB,
    "warm_start_source": str(TRAIN_IMAGES),
    "warm_start_seed": CALIB_SEED,
    "quantizers_inserted": _inserted_quantizers,
    "quantizers_after_restore": n_quant,
    "quantizers_with_scales_after_restore": n_scales,
    "reload_gate": "PASSED",       # verify_reload exits non-zero otherwise
    "torch": torch.__version__,
    "single_gpu": True,
}
(run_dir / "qat_provenance.json").write_text(json.dumps(prov, indent=2))
print(f"\n[QAT] provenance: {run_dir / 'qat_provenance.json'}")
print("[QAT] DONE -- plumbing run. Not an accuracy result.")
