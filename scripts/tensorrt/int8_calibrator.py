from pathlib import Path

import numpy as np
import cv2
import tensorrt as trt
from cuda.bindings import runtime as cudart


# ==========================================================
# PREPROCESSING
#
# IMPORTANT WHEN CHANGING MODEL / DATA:
# This preprocessing MUST match the exact inference input path.
#
# MODIFY if the new model uses:
# - different image resizing
# - different padding
# - different input normalization
# - different channel ordering
# - different input dimensions
# ==========================================================

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

    lb = letterbox(
        img_bgr,
        imgsz,
    )

    # MODIFY if the new model does not use RGB / CHW / [0,1].
    im = (
        lb[:, :, ::-1]
        .transpose(2, 0, 1)
    )

    im = (
        np.ascontiguousarray(
            im,
            dtype=np.float32,
        )
        / 255.0
    )

    # Current model input:
    # [1, 3, imgsz, imgsz]
    return im[None]


def _chk(ret):

    err, rest = (
        (ret[0], ret[1:])
        if isinstance(
            ret,
            (tuple, list),
        )
        else (ret, ())
    )

    if int(err) != 0:
        raise RuntimeError(
            f"CUDA error code {int(err)}"
        )

    return (
        rest[0]
        if len(rest) == 1
        else (rest or None)
    )


# ==========================================================
# CALIBRATOR INPUT HANDLING
#
# MODIFY if the new TensorRT model changes:
# - batch size
# - number of channels
# - image dimensions
# - input datatype
#
# Current contract:
# batch = 1
# channels = 3
# dtype = float32
# shape = [1, 3, imgsz, imgsz]
# ==========================================================

class _CalibratorMixin:

    def _setup(
        self,
        image_paths,
        cache_path,
        imgsz,
        log=print,
    ):

        self.image_paths = list(
            image_paths
        )

        self.cache_path = Path(
            cache_path
        )

        self.imgsz = imgsz
        self.log = log
        self.index = 0

        # MODIFY if input shape or datatype changes.
        self.nbytes = (
            1
            * 3
            * imgsz
            * imgsz
            * 4
        )

        self.d_input = _chk(
            cudart.cudaMalloc(
                self.nbytes
            )
        )

        self.log(
            f"  calibrator: "
            f"{len(self.image_paths)} frames, "
            f"batch=1, "
            f"imgsz={imgsz}"
        )


    # MODIFY if the TensorRT engine uses batch > 1.
    def get_batch_size(self):
        return 1


    def get_batch(
        self,
        names,
    ):

        if self.index >= len(
            self.image_paths
        ):
            return None


        path = self.image_paths[
            self.index
        ]

        img = cv2.imread(
            str(path)
        )

        if img is None:
            raise RuntimeError(
                f"Calibration frame "
                f"unreadable: {path}"
            )


        batch = np.ascontiguousarray(
            preprocess(
                img,
                self.imgsz,
            ),
            dtype=np.float32,
        )


        if batch.nbytes != self.nbytes:
            raise RuntimeError(
                f"Calibration batch is "
                f"{batch.nbytes} bytes, "
                f"network expects "
                f"{self.nbytes}."
            )


        _chk(
            cudart.cudaMemcpy(
                self.d_input,
                batch.ctypes.data,
                self.nbytes,
                cudart.cudaMemcpyKind
                .cudaMemcpyHostToDevice,
            )
        )


        self.index += 1


        if (
            self.index % 100 == 0
            or self.index
            == len(self.image_paths)
        ):
            self.log(
                f"    calibrated "
                f"{self.index}/"
                f"{len(self.image_paths)} "
                f"frames"
            )


        return [
            int(self.d_input)
        ]


    # Calibration cache is reusable only when:
    # - model is unchanged
    # - preprocessing is unchanged
    # - calibration dataset is unchanged
    #
    # DELETE the cache when any of these change.
    def read_calibration_cache(
        self
    ):

        if self.cache_path.exists():

            self.log(
                f"  reusing calibration "
                f"cache: {self.cache_path}"
            )

            return (
                self.cache_path
                .read_bytes()
            )

        return None


    def write_calibration_cache(
        self,
        cache,
    ):

        self.cache_path.write_bytes(
            cache
        )

        self.log(
            f"  wrote calibration "
            f"cache: {self.cache_path}"
        )


# ==========================================================
# CALIBRATION METHOD
#
# Select from build_tensorrt_engine.py using:
#
# CALIBRATOR=entropy
# or
# CALIBRATOR=minmax
#
# Additional calibration methods can be added here.
# ==========================================================

class EntropyCalibrator(
    _CalibratorMixin,
    trt.IInt8EntropyCalibrator2,
):

    def __init__(
        self,
        image_paths,
        cache_path,
        imgsz,
        log=print,
    ):

        trt.IInt8EntropyCalibrator2.__init__(
            self
        )

        self._setup(
            image_paths,
            cache_path,
            imgsz,
            log,
        )


class MinMaxCalibrator(
    _CalibratorMixin,
    trt.IInt8MinMaxCalibrator,
):

    def __init__(
        self,
        image_paths,
        cache_path,
        imgsz,
        log=print,
    ):

        trt.IInt8MinMaxCalibrator.__init__(
            self
        )

        self._setup(
            image_paths,
            cache_path,
            imgsz,
            log,
        )


# MODIFY this dictionary if another calibration
# strategy is implemented.
CALIBRATORS = {
    "entropy": EntropyCalibrator,
    "minmax": MinMaxCalibrator,
}
