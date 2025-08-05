import os
import torch
from transformers import CLIPModel, CLIPProcessor
import numpy as np

def load_onnx_or_hf_model(int8_onnx_path, bitsandbytes_available):
    if os.path.exists(int8_onnx_path):
        import onnxruntime as ort
        print("[INFO] Loading INT8 ONNX vision encoder from disk...")
        ort_session = ort.InferenceSession(int8_onnx_path)
        processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        def onnx_infer(images):
            inputs = processor(images=images, return_tensors="np")
            ort_inputs = {k: v for k, v in inputs.items() if k == "pixel_values"}
            ort_outs = ort_session.run(None, ort_inputs)
            return ort_outs[0]
        return onnx_infer, None, processor
    else:
        print("[INFO] Loading HuggingFace FashionCLIP model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if bitsandbytes_available and device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = CLIPModel.from_pretrained(
                "patrickjohncyh/fashion-clip",
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
        processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        def hf_infer(images):
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                embeds = model.get_image_features(**inputs)
                return embeds.cpu().numpy()
        return hf_infer, model, processor
