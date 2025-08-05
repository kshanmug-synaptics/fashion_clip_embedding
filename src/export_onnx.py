import os
import torch
from transformers import CLIPModel, CLIPProcessor

# ── 1) Prepare the ONNX output directory ─────────────────────────────────────────
onnx_dir = os.path.join(os.getcwd(), "onnx")
os.makedirs(onnx_dir, exist_ok=True)

# ── 2) Load the full-precision FashionCLIP vision tower on CPU ───────────────────
model_id = "patrickjohncyh/fashion-clip"
clip = CLIPModel.from_pretrained(model_id)
vision_encoder = clip.vision_model.eval().to("cpu")

# ── 3) Build a dummy input tensor ────────────────────────────────────────────────
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

# ── 4) Export the vision encoder to ONNX ─────────────────────────────────────────
onnx_fp32_path = os.path.join(onnx_dir, "fashion_clip_vision_fp32.onnx")
torch.onnx.export(
    vision_encoder,
    (dummy_input,),
    onnx_fp32_path,
    input_names=["pixel_values"],
    output_names=["image_features"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "image_features": {0: "batch"}
    },
    opset_version=14
)
print(f"Exported FP32 ONNX model to:\n  {onnx_fp32_path}")

# ── 5) (Optional) Quantize the ONNX model to INT8 ────────────────────────────────
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    onnx_int8_path = os.path.join(onnx_dir, "fashion_clip_vision_int8.onnx")
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized INT8 ONNX model saved to:\n  {onnx_int8_path}")
except ImportError:
    print("onnxruntime-tools not installed; skipping ONNX quantization.")
