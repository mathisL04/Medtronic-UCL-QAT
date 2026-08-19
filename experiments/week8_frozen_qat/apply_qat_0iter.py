"""Week-8 Step 2: apply the QAT quantization framework to the frozen baseline (0.546)
with ZERO training iterations. Same modelopt path as train_qat.py (mtq.quantize +
max-calibration on 128 episode-diverse train frames, seed 42) -- just no fine-tune.
Output: a calibrated modelopt state, reloadable by export_qat_onnx.py -> Q/DQ ONNX -> TRT.

This is "PTQ via the explicit-Q/DQ QAT path": the exact QAT quantization, 0 iterations.
"""
import os, sys, json, random, hashlib
from pathlib import Path
from datetime import datetime, timezone
REPO = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT")
FROZEN = Path(os.environ.get("MODEL_PATH", str(REPO/"runs_week8/week8_frozen_head/weights/best.pt")))
TRAIN_IMAGES = Path("/home/zcemml1/medtronic_qat_data/datasets/sanoscience_yolo_full_nonexpert_stereo/images/train")
OUT = Path(os.environ.get("OUT", str(REPO/"experiments/week8_frozen_qat/qat0")))
N_CALIB = int(os.environ.get("N_CALIB", 128)); CALIB_SEED = int(os.environ.get("CALIB_SEED", 42))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
DEVICE = os.environ.get("DEVICE")
if DEVICE is None: sys.exit("DEVICE required (e.g. DEVICE=3)")
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np, cv2, torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from modelopt.torch.quantization.nn import TensorQuantizer

_lb = LetterBox((IMG_SIZE, IMG_SIZE), auto=False)
def preprocess(img_bgr):
    im = _lb(image=img_bgr)[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(im, dtype=np.float32)[None] / 255.0

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

print("="*60); print("Week-8: 0-iteration QAT (quantize+calibrate, NO training)"); print("="*60)
print("frozen model:", FROZEN); print("device:", DEVICE, "N_CALIB:", N_CALIB)
if not FROZEN.exists(): sys.exit(f"frozen checkpoint not found: {FROZEN}")

model = YOLO(str(FROZEN)).model.cuda().eval()

# same episode-diverse calibration sample as train_qat.py (one frame per episode, seed 42)
all_train = sorted(TRAIN_IMAGES.glob("*.jpg"))
by_ep = {}
for p in all_train: by_ep.setdefault(p.name.split("_left_")[0], []).append(p)
rng = random.Random(CALIB_SEED)
chosen = rng.sample(sorted(by_ep), min(N_CALIB, len(by_ep)))
frames = [rng.choice(by_ep[e]) for e in sorted(chosen)]
print(f"calibration: {len(frames)} frames across {len(chosen)} episodes (seed {CALIB_SEED})")

def forward_loop(m):
    m.eval()
    with torch.no_grad():
        for i, fp in enumerate(frames):
            img = cv2.imread(str(fp))
            if img is None: raise RuntimeError(f"unreadable {fp}")
            x = torch.from_numpy(preprocess(img)).to(next(m.parameters()).device)
            m(x)
            if (i+1) % 32 == 0: print(f"  calibrated {i+1}/{len(frames)}", flush=True)

cfg = mtq.INT8_DEFAULT_CFG          # standard config (byte-identical to V6 default)
print("quantizing (mtq.quantize + max calibration) ...")
model = mtq.quantize(model, cfg, forward_loop)      # <-- insert Q/DQ + calibrate, NO training

nq = sum(1 for m in model.modules() if isinstance(m, TensorQuantizer))
nscale = sum(1 for m in model.modules() if isinstance(m, TensorQuantizer) and getattr(m,"_amax",None) is not None)
print(f"inserted {nq} TensorQuantizers ({nscale} with scales)")

state = OUT/"qat_modelopt_state_best.pt"
mto.save(model, str(state))
print("saved state:", state)

# reload gate (same as train_qat.py verify_reload)
fresh = YOLO(str(FROZEN)).model
mto.restore(fresh, str(state))
rq = sum(1 for m in fresh.modules() if isinstance(m, TensorQuantizer))
rs = sum(1 for m in fresh.modules() if isinstance(m, TensorQuantizer) and getattr(m,"_amax",None) is not None)
gate = "PASSED" if rs>0 else "FAILED"
print(f"reload gate {gate}: {rq} quantizers, {rs} with scales")

prov = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "stage":"week8_qat_0iter",
        "source_frozen": str(FROZEN), "source_sha256": sha256(FROZEN),
        "epochs": 0, "training":"NONE (calibration only)", "quant_config":"INT8_DEFAULT_CFG",
        "n_calib_frames": N_CALIB, "calib_seed": CALIB_SEED, "calib_method":"max",
        "quantizers_inserted": nq, "quantizers_with_scales": nscale,
        "reload_gate": gate, "modelopt": __import__("modelopt").__version__, "torch": torch.__version__}
(OUT/"qat0_provenance.json").write_text(json.dumps(prov, indent=2))
print("provenance:", OUT/"qat0_provenance.json")
if gate!="PASSED": sys.exit("reload gate failed")
print("DONE -- 0-iteration QAT state ready for export.")
