from ultralytics import YOLO
import torch.nn as nn
from pathlib import Path
import argparse


def inspect_model(model_path: str):
    model_path = Path(model_path)

    if not model_path.exists() and not str(model_path).endswith(".pt"):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    pytorch_model = model.model

    print("\n====================================")
    print("YOLO MODEL CLEAN ARCHITECTURE SUMMARY")
    print("====================================\n")

    print(f"Model path: {model_path}")

    total_params = sum(p.numel() for p in pytorch_model.parameters())
    trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("\n1. MODEL SIZE")
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

    if hasattr(model, "names"):
        print(f"   Class names:          {model.names}")

    print("\n6. SIMPLE INTERPRETATION")
    print("   The backbone extracts general visual features.")
    print("   The neck combines multi-scale features.")
    print("   The detection head predicts boxes and class scores.")
    print("   After fine-tuning, the detection head changes to match the dataset classes.")
    print("   This is useful before TensorRT export, INT8 quantization, and QAT work.")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Path to YOLO model, e.g. yolo26n.pt or runs_utenn/.../best.pt",
    )
    args = parser.parse_args()

    inspect_model(args.model)