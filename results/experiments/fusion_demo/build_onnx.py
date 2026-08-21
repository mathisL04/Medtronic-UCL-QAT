"""Tiny CNN + 3 ONNX variants for the Q/DQ-fusion-break demo.
Conv3x3->SiLU x3, input 1x32x160x160. Variant C uses modelopt INT8 so its Q/DQ
sit exactly where the real QAT model has them (weight+input quant on conv, SiLU
in float, requantizer after) — the faithful analog of the real conv+SiLU blocks.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""   # export is CPU-fine
import torch, torch.nn as nn
from pathlib import Path
OUT=Path("results/experiments/fusion_demo"); OUT.mkdir(exist_ok=True)

class TinyCNN(nn.Module):
    def __init__(self, c=32, blocks=3):
        super().__init__()
        self.blocks=nn.ModuleList([nn.Conv2d(c,c,3,padding=1,bias=False) for _ in range(blocks)])
        self.act=nn.SiLU()
    def forward(self,x):
        for conv in self.blocks:
            x=self.act(conv(x))
        return x

torch.manual_seed(0)
m=TinyCNN().eval()
dummy=torch.zeros(1,32,160,160)

# --- Variant A/B share the SAME plain FP32 ONNX (no Q/DQ). A=FP16 build, B=PTQ implicit build ---
torch.onnx.export(m, dummy, str(OUT/"tiny_plain.onnx"), opset_version=17,
                  input_names=["x"], output_names=["y"], dynamic_axes=None)
print("wrote tiny_plain.onnx (for FP16 + PTQ-implicit builds)")

# --- Variant C: explicit Q/DQ via modelopt (mirrors real QAT placement) ---
import modelopt.torch.quantization as mtq
from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes_and_metadata, OnnxBytes
def forward_loop(model):
    for _ in range(8): model(torch.randn(1,32,160,160))  # calibrate on random data
mq=TinyCNN().eval()
mq.load_state_dict(m.state_dict())   # same random weights
mq=mtq.quantize(mq, mtq.INT8_DEFAULT_CFG, forward_loop)
from modelopt.torch.quantization.nn import TensorQuantizer
nq=sum(1 for x in mq.modules() if isinstance(x,TensorQuantizer))
print(f"modelopt inserted {nq} TensorQuantizers")
payload,_=get_onnx_bytes_and_metadata(mq, dummy, onnx_opset=17)
ob=OnnxBytes.from_bytes(payload) if isinstance(payload,(bytes,bytearray)) else payload
tmp=OUT/"_qdq_tmp"; ob.write_to_disk(str(tmp))
import glob
src=sorted(glob.glob(str(tmp/"**"/"*.onnx"),recursive=True))[0]
Path(src).replace(OUT/"tiny_qdq.onnx")
print("wrote tiny_qdq.onnx (explicit Q/DQ, modelopt)")

# --- report structure of both ONNX ---
import onnx
from collections import Counter
for f in ["tiny_plain.onnx","tiny_qdq.onnx"]:
    g=onnx.load(str(OUT/f)).graph
    c=Counter(n.op_type for n in g.node)
    print(f"  {f:16} nodes={len(g.node)} Conv={c['Conv']} Sigmoid={c.get('Sigmoid',0)} Mul={c.get('Mul',0)} "
          f"Q={c.get('QuantizeLinear',0)} DQ={c.get('DequantizeLinear',0)}")
