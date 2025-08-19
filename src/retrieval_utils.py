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
        try:
            dists = dist_matrix[idx]
            top4 = np.argsort(dists)[1:5]  # skip self (distance 0)
            
            # Safely load images with error handling
            imgs = []
            try:
                query_img = images[idx]
                if hasattr(query_img, 'load'):
                    query_img.load()  # Ensure image is fully loaded
                imgs.append(query_img.copy())  # Make a copy to avoid threading issues
            except Exception as e:
                print(f"[WARNING] Failed to load query image {idx}: {e}")
                return
            
            for j in top4:
                try:
                    retrieval_img = images[j]
                    if hasattr(retrieval_img, 'load'):
                        retrieval_img.load()  # Ensure image is fully loaded
                    imgs.append(retrieval_img.copy())  # Make a copy to avoid threading issues
                except Exception as e:
                    print(f"[WARNING] Failed to load retrieval image {j}: {e}")
                    # Create a placeholder image
                    placeholder = Image.new("RGB", (224, 224), (128, 128, 128))
                    imgs.append(placeholder)
            
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
        except Exception as e:
            print(f"[ERROR] Failed to create retrieval result for image {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Reduce max_workers to avoid threading issues with PIL
    with ThreadPoolExecutor(max_workers=min(max_workers, 4)) as executor:
        list(tqdm(executor.map(save_one, range(len(images))), total=len(images), desc="Retrieval results (threaded)"))
