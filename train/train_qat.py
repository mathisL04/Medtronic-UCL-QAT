import os
import random
import sys
import json
import copy
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import torch


# ==========================================================
# USER CONFIGURATION
# Modify this section when changing the baseline model,
# dataset, QAT calibration, training setup, or sweep settings.
# ==========================================================

# Baseline checkpoint used as the starting point for QAT.
# Replace when quantizing a different model/checkpoint.
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
        "models/yolo26n_sanoscience_full_left/baseline/best.pt",
    )
)

# Dataset YAML used by the Ultralytics trainer.
# Replace when changing training/validation data.
DATA_YAML = Path(
    "/home/zcemml1/medtronic_qat_data/datasets/"
    "sanoscience_yolo_full_nonexpert_stereo/"
    "sanoscience_yolo.yaml"
)

# Training images used to initialise QAT quantizer ranges.
# Replace when using another dataset or training split.
TRAIN_IMAGES = Path(
    "/home/zcemml1/medtronic_qat_data/datasets/"
    "sanoscience_yolo_full_nonexpert_stereo/"
    "images/train"
)

# Number of representative frames used to warm-start quantizer ranges.
N_CALIB = int(
    os.environ.get("N_CALIB", 128)
)

# Seed controlling reproducible warm-start sampling.
CALIB_SEED = int(
    os.environ.get("CALIB_SEED", 42)
)

# GPU selection.
# DEVICE is deliberately required rather than guessed.
DEVICE = os.environ.get("DEVICE")
if DEVICE is None:
    sys.exit(
        "DEVICE is required "
        "(e.g. DEVICE=1)."
    )

# Main training controls.
EPOCHS = int(
    os.environ.get("EPOCHS", 1)
)

# 0 disables early stopping.
# >0 stops after this many epochs without validation improvement.
PATIENCE = int(
    os.environ.get("PATIENCE", 0)
)

RUN_NAME = os.environ.get(
    "RUN_NAME",
    "qat_smoke"
)

PROJECT = os.environ.get(
    "PROJECT",
    "/tmp/zcemml1_qat"
)

# Input resolution must remain compatible with the model
# and later TensorRT engine input.
IMG_SIZE = int(
    os.environ.get("IMG_SIZE", 640)
)

# Adapt batch size to the model and available GPU memory.
BATCH = int(
    os.environ.get("BATCH", 16)
)

# Adapt worker count to the host/memory environment.
WORKERS = int(
    os.environ.get("WORKERS", 4)
)

# Fine-tuning learning-rate parameters.
LR0 = float(
    os.environ.get("LR0", 1e-3)
)

LRF = float(
    os.environ.get("LRF", 0.01)
)

# Quantization configuration knobs.
# WEIGHT_AXIS:
#   0    -> per-channel weight quantization
#   None -> per-tensor weight quantization
WEIGHT_AXIS = os.environ.get(
    "WEIGHT_AXIS",
    "0"
)

# Optional layer patterns to exclude from quantization.
DISABLE_LAYERS = os.environ.get(
    "DISABLE_LAYERS",
    ""
)

# Initial range calibration method.
# Supported:
# max | entropy | mse | percentile
CALIB_METHOD = os.environ.get(
    "CALIB_METHOD",
    "max"
).lower()

CALIB_PERCENTILE = float(
    os.environ.get(
        "CALIB_PERCENTILE",
        "99.99"
    )
)

CALIB_NUM_BINS = int(
    os.environ.get(
        "CALIB_NUM_BINS",
        "2048"
    )
)

os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE


import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from modelopt.torch.quantization.nn import TensorQuantizer


def count_quantizers(model):
    total = sum(
        1
        for m in model.modules()
        if isinstance(m, TensorQuantizer)
    )

    with_amax = sum(
        1
        for m in model.modules()
        if isinstance(m, TensorQuantizer)
        and getattr(m, "_amax", None) is not None
    )

    return total, with_amax


_inserted_quantizers = None
_quant_cfg = None
_quant_knobs = {}


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def letterbox(img, new_shape):
    h0, w0 = img.shape[:2]

    r = min(
        new_shape / h0,
        new_shape / w0,
    )

    nw = int(round(w0 * r))
    nh = int(round(h0 * r))

    resized = cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (new_shape, new_shape, 3),
        114,
        dtype=np.uint8,
    )

    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2

    canvas[
        top:top + nh,
        left:left + nw
    ] = resized

    return canvas


def preprocess(img_bgr, imgsz):

    # Modify if the replacement model uses a different
    # input preprocessing convention.
    lb = letterbox(
        img_bgr,
        imgsz,
    )

    im = lb[
        :, :, ::-1
    ].transpose(
        2, 0, 1
    )

    return (
        np.ascontiguousarray(
            im,
            dtype=np.float32,
        )[None]
        / 255.0
    )


def make_forward_loop(
    frames,
    device,
):
    def forward_loop(model):
        model.eval()

        with torch.no_grad():
            for i, fp in enumerate(frames):

                img = cv2.imread(
                    str(fp)
                )

                if img is None:
                    raise RuntimeError(
                        f"Calibration frame unreadable: {fp}"
                    )

                x = torch.from_numpy(
                    preprocess(
                        img,
                        IMG_SIZE,
                    )
                ).to(
                    next(
                        model.parameters()
                    ).device
                )

                model(x)

                if (
                    i + 1
                ) % 32 == 0:
                    print(
                        f"calibrated "
                        f"{i + 1}/"
                        f"{len(frames)}",
                        flush=True,
                    )

        model.train()

    return forward_loop


def compose_quant_config():

    # Modify this function when changing:
    # - quantization bit-width/configuration,
    # - weight granularity,
    # - selectively quantized layers,
    # - custom ModelOpt quantization rules.

    cfg = copy.deepcopy(
        mtq.INT8_DEFAULT_CFG
    )

    knobs = {}

    waxis = (
        WEIGHT_AXIS or "0"
    )

    if str(waxis).lower() == "none":
        cfg[
            "quant_cfg"
        ][
            "*weight_quantizer"
        ][
            "axis"
        ] = None

        knobs["WEIGHT_AXIS"] = None

    elif str(waxis) != "0":
        sys.exit(
            "WEIGHT_AXIS must be "
            "0 or None."
        )

    dis = DISABLE_LAYERS.strip()

    if dis:
        patterns = [
            p.strip()
            for p in dis.split(",")
            if p.strip()
        ]

        for pattern in patterns:
            cfg[
                "quant_cfg"
            ][pattern] = {
                "enable": False
            }

        knobs[
            "DISABLE_LAYERS"
        ] = patterns

    return cfg, knobs


def calibrate_qat(
    model,
    cfg,
    forward_loop,
):

    # Modify CALIB_METHOD and related parameters to compare
    # different quantizer range-initialization strategies.

    method = CALIB_METHOD

    if method == "max":
        return mtq.quantize(
            model,
            cfg,
            forward_loop,
        )

    if method not in (
        "entropy",
        "mse",
        "percentile",
    ):
        sys.exit(
            "CALIB_METHOD must be "
            "max, entropy, mse or percentile."
        )

    from modelopt.torch.quantization.model_calib import (
        enable_stats_collection,
    )
    from modelopt.torch.quantization.calib.histogram import (
        HistogramCalibrator,
    )

    if torch.cuda.is_available():
        model.cuda()

    cfg = copy.deepcopy(cfg)

    cfg[
        "quant_cfg"
    ][
        "*input_quantizer"
    ][
        "calibrator"
    ] = "histogram"

    cfg[
        "quant_cfg"
    ][
        "*input_quantizer"
    ][
        "num_bins"
    ] = CALIB_NUM_BINS

    cfg["algorithm"] = None

    mtq.quantize(
        model,
        cfg,
        forward_loop=None,
    )

    enable_stats_collection(
        model
    )

    forward_loop(
        model
    )

    for _, q in model.named_modules():

        if (
            not isinstance(
                q,
                TensorQuantizer,
            )
            or q._disabled
        ):
            continue

        cal = getattr(
            q,
            "_calibrator",
            None,
        )

        if (
            cal is not None
            and not getattr(
                q,
                "_dynamic",
                False,
            )
        ):

            if isinstance(
                cal,
                HistogramCalibrator,
            ):
                kwargs = (
                    {
                        "percentile":
                        CALIB_PERCENTILE
                    }
                    if method == "percentile"
                    else {}
                )

                if (
                    cal.compute_amax(
                        method,
                        **kwargs,
                    )
                    is not None
                ):
                    q.load_calib_amax(
                        method=method,
                        **kwargs,
                    )

            elif (
                cal.compute_amax()
                is not None
            ):
                q.load_calib_amax()

        q.enable_quant()
        q.disable_calib()

    return model


class QATTrainer(
    DetectionTrainer
):

    def get_model(
        self,
        cfg=None,
        weights=None,
        verbose=True,
    ):
        model = super().get_model(
            cfg=cfg,
            weights=weights,
            verbose=verbose,
        )

        # Dataset-specific grouping logic.
        # Replace if filenames do not encode correlated
        # sequences using "_left_".
        all_train = sorted(
            TRAIN_IMAGES.glob("*.jpg")
        )

        if not all_train:
            sys.exit(
                f"No training images found at "
                f"{TRAIN_IMAGES}"
            )

        by_group = {}

        for p in all_train:

            group = p.name.split(
                "_left_"
            )[0]

            by_group.setdefault(
                group,
                [],
            ).append(p)

        rng = random.Random(
            CALIB_SEED
        )

        groups = sorted(
            by_group
        )

        chosen = rng.sample(
            groups,
            min(
                N_CALIB,
                len(groups),
            ),
        )

        frames = [
            rng.choice(
                by_group[g]
            )
            for g in sorted(chosen)
        ]

        device = next(
            model.parameters()
        ).device

        global _quant_cfg
        global _quant_knobs

        (
            _quant_cfg,
            _quant_knobs,
        ) = compose_quant_config()

        model = calibrate_qat(
            model,
            _quant_cfg,
            make_forward_loop(
                frames,
                device,
            ),
        )

        mtq.print_quant_summary(
            model
        )

        global _inserted_quantizers

        (
            _inserted_quantizers,
            with_amax,
        ) = count_quantizers(
            model
        )

        print(
            f"[QAT] inserted "
            f"{_inserted_quantizers} "
            f"TensorQuantizers "
            f"({with_amax} with scales)",
            flush=True,
        )

        return model

    def save_model(self):
        return None


def save_qat_state(
    model,
    out_dir,
):
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    p = (
        out_dir
        / "qat_modelopt_state.pt"
    )

    mto.save(
        model,
        p,
    )

    print(
        f"[QAT] state saved: {p}"
    )

    return p


def verify_reload(
    state_path,
):
    fresh = YOLO(
        str(MODEL_PATH)
    ).model

    restored = mto.restore(
        fresh,
        state_path,
    )

    (
        n_quant,
        with_amax,
    ) = count_quantizers(
        restored
    )

    if _inserted_quantizers is None:
        sys.exit(
            "Inserted quantizer count unavailable."
        )

    if (
        n_quant
        != _inserted_quantizers
    ):
        sys.exit(
            f"Reload failed: "
            f"{n_quant} restored vs "
            f"{_inserted_quantizers} inserted."
        )

    if with_amax == 0:
        sys.exit(
            "Reload failed: "
            "no quantization scales restored."
        )

    print(
        f"[QAT] reload gate PASSED: "
        f"{n_quant} quantizers"
    )

    return (
        n_quant,
        with_amax,
    )


_best_fitness = float(
    "-inf"
)


def save_best_qat(
    trainer,
):
    global _best_fitness

    fit = getattr(
        trainer,
        "fitness",
        None,
    )

    if (
        fit is None
        or fit <= _best_fitness
    ):
        return

    _best_fitness = fit

    model = (
        trainer.ema.ema
        if getattr(
            trainer,
            "ema",
            None,
        )
        else trainer.model
    )

    out = (
        Path(PROJECT)
        / RUN_NAME
        / "qat_modelopt_state_best.pt"
    )

    mto.save(
        model,
        out,
    )

    print(
        f"[QAT] best fitness "
        f"{fit:.5f} "
        f"@ epoch "
        f"{getattr(trainer, 'epoch', '?')} "
        f"-> {out.name}",
        flush=True,
    )


print("=" * 60)
print(
    f"QAT fine-tune: "
    f"{RUN_NAME}"
)
print("=" * 60)

print(
    "Model:",
    MODEL_PATH,
)

print(
    "Dataset:",
    DATA_YAML,
)

print(
    "Device:",
    DEVICE,
)

print(
    "Epochs:",
    EPOCHS,
)

print(
    "Patience:",
    PATIENCE,
)

print(
    "Batch:",
    BATCH,
)

print(
    "Calibration:",
    N_CALIB,
    CALIB_METHOD,
)


if not MODEL_PATH.exists():
    sys.exit(
        f"Baseline checkpoint not found: "
        f"{MODEL_PATH}"
    )

if not DATA_YAML.exists():
    sys.exit(
        f"Dataset YAML not found: "
        f"{DATA_YAML}"
    )


overrides = dict(

    model=str(
        MODEL_PATH
    ),

    data=str(
        DATA_YAML
    ),

    epochs=EPOCHS,

    # Modify for fixed-duration vs early-stopping experiments.
    patience=PATIENCE,

    imgsz=IMG_SIZE,

    batch=BATCH,

    workers=WORKERS,

    device=(
        int(DEVICE)
        if DEVICE.isdigit()
        else DEVICE
    ),

    project=PROJECT,

    name=RUN_NAME,

    exist_ok=True,

    # Keep disabled for the reference ModelOpt QAT configuration.
    # Change only after validating compatibility with the chosen QAT method.
    amp=False,

    val=True,

    plots=False,

    # ModelOpt states are saved through mto.save() instead.
    save=False,

    # Main QAT fine-tuning hyperparameters.
    lr0=LR0,
    lrf=LRF,
)


trainer = QATTrainer(
    overrides=overrides
)

trainer.add_callback(
    "on_fit_epoch_end",
    save_best_qat,
)

trainer.train()


run_dir = (
    Path(PROJECT)
    / RUN_NAME
)

model = (
    trainer.ema.ema
    if getattr(
        trainer,
        "ema",
        None,
    )
    else trainer.model
)

state_path = save_qat_state(
    model,
    run_dir,
)

n_quant, n_scales = (
    verify_reload(
        state_path
    )
)


prov = {
    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "run_name":
        RUN_NAME,

    "epochs":
        EPOCHS,

    "patience":
        PATIENCE,

    "device":
        DEVICE,

    "amp":
        False,

    "source_checkpoint":
        str(MODEL_PATH),

    "source_sha256":
        sha256(MODEL_PATH),

    "quant_config":
        "INT8_DEFAULT_CFG(composed)",

    "quant_config_knobs":
        _quant_knobs,

    "calib_method":
        CALIB_METHOD,

    "calib_percentile":
        CALIB_PERCENTILE,

    "calib_num_bins":
        CALIB_NUM_BINS,

    "quant_config_resolved":
        json.loads(
            json.dumps(
                _quant_cfg,
                default=str,
            )
        ),

    "n_calib_frames":
        N_CALIB,

    "warm_start_source":
        str(TRAIN_IMAGES),

    "warm_start_seed":
        CALIB_SEED,

    "quantizers_inserted":
        _inserted_quantizers,

    "quantizers_after_restore":
        n_quant,

    "quantizers_with_scales_after_restore":
        n_scales,

    "reload_gate":
        "PASSED",

    "torch":
        torch.__version__,

    "single_gpu":
        True,
}


(
    run_dir
    / "qat_provenance.json"
).write_text(
    json.dumps(
        prov,
        indent=2,
    )
)

print(
    f"\n[QAT] provenance: "
    f"{run_dir / 'qat_provenance.json'}"
)

print(
    "[QAT] DONE"
)
