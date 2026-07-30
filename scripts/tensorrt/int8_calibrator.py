"""INT8 PTQ calibrators for build_tensorrt_engine.py (docs/05).

Unlike the rest of scripts/, this is an importable module rather than a flat
top-to-bottom program: a TensorRT calibrator has to be a class the builder calls
back into. It has no side effects on import and is not meant to be run directly.

NOTE ON DEPRECATION. IInt8EntropyCalibrator2 / IInt8MinMaxCalibrator are marked
"[DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization"
in TensorRT 10.16.1.11. They still function, and implicit calibration is what
"post-training quantisation" classically means, so this is the right mechanism
for the PTQ stage -- but the API may be removed in a future TensorRT, and the
QAT stage (docs/06) will use explicit Q/DQ instead. Version is pinned in the
engine provenance so this build stays reproducible.
"""
from pathlib import Path

import numpy as np
import cv2
import tensorrt as trt
from cuda.bindings import runtime as cudart


# -----------------------------------------------------------------------------
# Preprocessing -- MUST match inference exactly
# -----------------------------------------------------------------------------
# Copied verbatim from validate_engine_parity.py / evaluate_engine_map.py. It is
# duplicated rather than imported because those are flat scripts: importing one
# would execute an entire parity or mAP run as a side effect.
#
# If the preprocessing here ever drifts from the preprocessing used at inference,
# every activation range observed during calibration is measured on the wrong
# input distribution, and TensorRT computes scales for a network that will never
# see that data. The engine still builds and still validates as "successful" --
# it is just quietly wrong. Keep these two functions byte-identical to the
# inference path.
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
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    return im[None]


def _chk(ret):
    err, rest = (ret[0], ret[1:]) if isinstance(ret, (tuple, list)) else (ret, ())
    if int(err) != 0:
        raise RuntimeError(f"CUDA error code {int(err)}")
    return rest[0] if len(rest) == 1 else (rest or None)


# -----------------------------------------------------------------------------
# Shared batch-feeding logic
# -----------------------------------------------------------------------------
class _CalibratorMixin:
    """Feeds calibration frames to the builder one at a time.

    Batch size is 1 because the engine is static [1, 3, 640, 640] -- the
    calibration batch shape must match the network's input shape.
    """

    def _setup(self, image_paths, cache_path, imgsz, log=print):
        self.image_paths = list(image_paths)
        self.cache_path = Path(cache_path)
        self.imgsz = imgsz
        self.log = log
        self.index = 0
        self.nbytes = 1 * 3 * imgsz * imgsz * 4          # float32 NCHW
        self.d_input = _chk(cudart.cudaMalloc(self.nbytes))
        self.log(f"  calibrator: {len(self.image_paths)} frames, batch=1, "
                 f"imgsz={imgsz}")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        """Return device pointers for the next batch, or None when exhausted."""
        if self.index >= len(self.image_paths):
            return None

        path = self.image_paths[self.index]
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Calibration frame unreadable: {path}")

        batch = np.ascontiguousarray(preprocess(img, self.imgsz), dtype=np.float32)
        if batch.nbytes != self.nbytes:
            raise RuntimeError(
                f"Calibration batch is {batch.nbytes} bytes, network expects "
                f"{self.nbytes}. Preprocessing does not match the engine input."
            )
        _chk(cudart.cudaMemcpy(
            self.d_input, batch.ctypes.data, self.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

        self.index += 1
        if self.index % 100 == 0 or self.index == len(self.image_paths):
            self.log(f"    calibrated {self.index}/{len(self.image_paths)} frames")
        return [int(self.d_input)]

    def read_calibration_cache(self):
        """Reuse a previous calibration if one exists.

        TensorRT skips the whole calibration pass when this returns bytes, which
        makes rebuilds cheap. The cache is tied to the calibration set by the
        sha256 recorded in provenance -- change the set, delete the cache.
        """
        if self.cache_path.exists():
            self.log(f"  reusing calibration cache: {self.cache_path}")
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(cache)
        self.log(f"  wrote calibration cache: {self.cache_path}")


class EntropyCalibrator(_CalibratorMixin, trt.IInt8EntropyCalibrator2):
    """KL-divergence calibration. Default for this project.

    Histograms each activation and picks the saturation threshold that minimises
    information loss, clipping the tail. Suits the conv/SiLU backbone, whose
    activations have long right tails that would otherwise let a single outlier
    set the scale and compress the informative range into a few of the 256 levels.
    """

    def __init__(self, image_paths, cache_path, imgsz, log=print):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self._setup(image_paths, cache_path, imgsz, log)


class MinMaxCalibrator(_CalibratorMixin, trt.IInt8MinMaxCalibrator):
    """Absolute min/max calibration. Kept as the A/B against entropy.

    Clips nothing, so it preserves extreme values -- which matters for box
    regression outputs, where large coordinates are semantically real rather than
    outliers. The trade is resolution: one extreme activation stretches the scale
    for everything else. If entropy shows a disproportionate mAP50-95 drop
    relative to mAP50, that is the localisation-clipping signature and this is
    the better choice.
    """

    def __init__(self, image_paths, cache_path, imgsz, log=print):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self._setup(image_paths, cache_path, imgsz, log)


CALIBRATORS = {"entropy": EntropyCalibrator, "minmax": MinMaxCalibrator}
