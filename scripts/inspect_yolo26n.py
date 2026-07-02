from ultralytics import YOLO
import torch.nn as nn

model = YOLO("yolo26n.pt")
pytorch_model = model.model

print("\n====================================")
print("YOLO26n CLEAN ARCHITECTURE SUMMARY")
print("====================================\n")

total_params = sum(p.numel() for p in pytorch_model.parameters())
trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

print("1. MODEL SIZE")
print(f"   Total parameters:        {total_params:,}")
print(f"   Trainable parameters:    {trainable_params:,}")
print(f"   Non-trainable params:    {non_trainable_params:,}")

num_top_layers = len(pytorch_model.model)
num_modules = sum(1 for _ in pytorch_model.modules())
num_conv2d = sum(1 for m in pytorch_model.modules() if isinstance(m, nn.Conv2d))
num_bn = sum(1 for m in pytorch_model.modules() if isinstance(m, nn.BatchNorm2d))
num_silu = sum(1 for m in pytorch_model.modules() if isinstance(m, nn.SiLU))
num_upsample = sum(1 for m in pytorch_model.modules() if isinstance(m, nn.Upsample))
num_pool = sum(1 for m in pytorch_model.modules() if isinstance(m, nn.MaxPool2d))

print("\n2. LAYER / MODULE COUNTS")
print(f"   Top-level YOLO layers:   {num_top_layers}")
print(f"   Total PyTorch modules:   {num_modules}")
print(f"   Conv2d layers:           {num_conv2d}")
print(f"   BatchNorm2d layers:      {num_bn}")
print(f"   SiLU activations:        {num_silu}")
print(f"   Upsample layers:         {num_upsample}")
print(f"   MaxPool2d layers:        {num_pool}")

print("\n3. HIGH-LEVEL ARCHITECTURE")
print("   Layers 0-10:   Backbone        -> extracts image features")
print("   Layers 11-22:  Neck            -> fuses multi-scale features")
print("   Layer 23:      Detection Head  -> predicts boxes and classes")

print("\n4. TOP-LEVEL LAYER BREAKDOWN")
print("   Index | Section      | Layer Type      | Parameters")
print("   ---------------------------------------------------")

for i, layer in enumerate(pytorch_model.model):
    params = sum(p.numel() for p in layer.parameters())

    if i <= 10:
        section = "Backbone"
    elif i <= 22:
        section = "Neck"
    else:
        section = "Head"

    print(f"   {i:02d}    | {section:11s} | {layer.__class__.__name__:15s} | {params:,}")

print("\n5. DETECTION HEAD")
head = pytorch_model.model[-1]
print(f"   Detection head layer: model.{num_top_layers - 1}")
print(f"   Detection head type:  {head.__class__.__name__}")

if hasattr(head, "nc"):
    print(f"   Number of classes:    {head.nc}")

if hasattr(head, "stride"):
    print(f"   Detection strides:    {head.stride}")

print("\n6. SIMPLE INTERPRETATION")
print("   YOLO26n is a lightweight object detection CNN.")
print("   The backbone extracts image features.")
print("   The neck combines features at different scales.")
print("   The detection head outputs bounding boxes and class predictions.")
print("   This summary helps before medical fine-tuning and TensorRT / INT8 work.")

print("\nDone.")    