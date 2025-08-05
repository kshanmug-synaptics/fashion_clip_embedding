import os
import numpy as np
from src.data_utils import load_deepfashion_images, save_images
from src.model_utils import load_onnx_or_hf_model
from src.retrieval_utils import l2_normalize, save_embeddings, compute_distance_matrix, save_retrieval_results
from src.visualization_utils import save_distance_heatmap

# Setup
print("[INFO] Setting up cache directories...")
root_cache = os.path.join(os.getcwd(), "cache")
datasets_cache = os.path.join(root_cache, "datasets")

# Load images
data_split = "train[:1000]"
print(f"[INFO] Loading DeepFashion dataset split: {data_split}")
images = load_deepfashion_images(split=data_split, cache_dir=datasets_cache)
print(f"[INFO] Loaded {len(images)} images.")

# Save images
images_dir = os.path.join(os.getcwd(), "images")
print("[INFO] Saving images...")
save_images(images, images_dir, max_workers=8)
print("[INFO] Images saved.")

# Model selection and embedding
onnx_dir = os.path.join(os.getcwd(), "onnx")
int8_onnx_path = os.path.join(onnx_dir, "fashion_clip_image_int8_matmul.onnx")
try:
    import bitsandbytes
    bitsandbytes_available = True
except ImportError:
    bitsandbytes_available = False

infer_fn, model, processor = load_onnx_or_hf_model(int8_onnx_path, bitsandbytes_available)
print("[INFO] Computing image embeddings...")
embeds = infer_fn(images)
print("[INFO] Embeddings computed.")

# Normalize and save embeddings
print("[INFO] L2-normalizing embeddings...")
embeds = l2_normalize(embeds)
print("[INFO] Embeddings normalized.")
embeddings_dir = os.path.join(os.getcwd(), "embeddings")
print("[INFO] Saving embeddings...")
save_embeddings(embeds, embeddings_dir)
print("[INFO] Embeddings saved.")

# Compute distance matrix
print("[INFO] Computing pairwise cosine distance matrix...")
dist_matrix = compute_distance_matrix(embeds)
print("[INFO] Distance matrix computed.")

# Retrieval results
retrieval_dir = os.path.join(os.getcwd(), "retrieval_results")
print("[INFO] Saving retrieval results...")
save_retrieval_results(images, dist_matrix, retrieval_dir, max_workers=8)
print("[INFO] Retrieval results saved.")

# Visualization
outputs_dir = os.path.join(os.getcwd(), "outputs")
print("[INFO] Visualizing distance matrix as heatmap...")
save_distance_heatmap(dist_matrix, outputs_dir)
