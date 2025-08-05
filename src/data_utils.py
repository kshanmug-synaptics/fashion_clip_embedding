import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def load_deepfashion_images(split="train[:1000]", cache_dir="./cache/datasets"):
    ds = load_dataset(
        "lirus18/deepfashion",
        split=split,
        cache_dir=cache_dir
    )
    images = [example["image"].convert("RGB") for example in ds]
    return images

def save_images(images, images_dir, max_workers=8):
    os.makedirs(images_dir, exist_ok=True)
    def save_one(idx_img):
        idx, img = idx_img
        img.save(os.path.join(images_dir, f"img_{idx:03d}.png"))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(save_one, enumerate(images)), total=len(images), desc="Saving images (threaded)"))
