import os
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont

def l2_normalize(embeds):
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    return embeds / norms

def save_embeddings(embeds, embeddings_dir):
    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(os.path.join(embeddings_dir, "embeddings.npy"), embeds)

def compute_distance_matrix(embeds):
    return cosine_distances(embeds)

def save_retrieval_results(images, dist_matrix, retrieval_dir, max_workers=8):
    os.makedirs(retrieval_dir, exist_ok=True)
    def save_one(idx):
        dists = dist_matrix[idx]
        top4 = np.argsort(dists)[1:5]  # skip self (distance 0)
        imgs = [images[idx]] + [images[j] for j in top4]
        width, height = imgs[0].size
        font = None
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 54)
        except:
            font = ImageFont.load_default()
        # Dynamically compute title height based on font
        title_texts = ["Query"] + [f"Retrieval#{i+1}={dists[j]:.3f}" for i, j in enumerate(top4)]
        text_heights = [font.getbbox(text)[3] - font.getbbox(text)[1] for text in title_texts]
        title_height = max(text_heights) + 24  # add some padding
        concat_img = Image.new("RGB", (width * 5, height + title_height), (255, 255, 255))
        draw = ImageDraw.Draw(concat_img)
        # Draw titles centered above each image
        for i, text in enumerate(title_texts):
            text_width = font.getbbox(text)[2] - font.getbbox(text)[0]
            x = i * width + (width - text_width) // 2
            y = (title_height - text_heights[i]) // 2
            draw.text((x, y), text, fill=(0,0,0), font=font)
        # Paste images
        for i, img in enumerate(imgs):
            concat_img.paste(img, (i*width, title_height))
        out_name = f"img_{idx:03d}_retrieval.png"
        concat_img.save(os.path.join(retrieval_dir, out_name))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(save_one, range(len(images))), total=len(images), desc="Retrieval results (threaded)"))
