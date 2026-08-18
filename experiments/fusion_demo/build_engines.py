import numpy as np, tensorrt as trt, json
from pathlib import Path
from cuda.bindings import runtime as cudart
OUT=Path("experiments/fusion_demo"); LOG=trt.Logger(trt.Logger.ERROR)
def CHK(r):
    e=r[0] if isinstance(r,(tuple,list)) else r
    if int(e)!=0: raise RuntimeError(f"cuda {e}")
    return r[1] if isinstance(r,(tuple,list)) and len(r)==2 else (r[1:] if isinstance(r,(tuple,list)) else None)

class RandCalib(trt.IInt8EntropyCalibrator2):
    """Proper implicit-PTQ calibrator: feeds random batches from a device buffer."""
    def __init__(self, shape=(1,32,160,160), n=16):
        super().__init__(); self.shape=shape; self.n=n; self.i=0
        self.nbytes=int(np.prod(shape))*4
        self.d=CHK(cudart.cudaMalloc(self.nbytes))
    def get_batch_size(self): return 1
    def get_batch(self, names):
        if self.i>=self.n: return None
        self.i+=1
        buf=np.random.randn(*self.shape).astype(np.float32)
        CHK(cudart.cudaMemcpy(self.d, buf.ctypes.data, self.nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        return [int(self.d)]
    def read_calibration_cache(self): return None
    def write_calibration_cache(self, c): pass

def build(onnx, engine, int8=False, qdq=False):
    b=trt.Builder(LOG); net=b.create_network(0); p=trt.OnnxParser(net,LOG)
    if not p.parse(Path(onnx).read_bytes()): raise SystemExit("parse fail "+str(onnx))
    cfg=b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4<<30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    if int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
        if not qdq:
            cfg.int8_calibrator=RandCalib()   # implicit PTQ via calibration
    cfg.builder_optimization_level=5
    cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    ser=b.build_serialized_network(net,cfg)
    if ser is None: raise SystemExit("build fail "+str(onnx))
    Path(engine).write_bytes(bytes(ser))
    R=trt.Runtime(LOG); e=R.deserialize_cuda_engine(bytes(ser))
    L=json.loads(e.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    layers=L["Layers"] if isinstance(L,dict) else L
    print(f"  built {Path(engine).name:20} layers={len(layers)} ({len(bytes(ser))/1024:.0f} KB)")

CHK(cudart.cudaSetDevice(0))
print("A) FP16 (no quant):");      build(OUT/"tiny_plain.onnx", OUT/"tiny_A_fp16.engine", int8=False)
print("B) INT8 PTQ (implicit):");  build(OUT/"tiny_plain.onnx", OUT/"tiny_B_ptq.engine", int8=True, qdq=False)
print("C) INT8 explicit Q/DQ:");   build(OUT/"tiny_qdq.onnx",   OUT/"tiny_C_qdq.engine", int8=True, qdq=True)
