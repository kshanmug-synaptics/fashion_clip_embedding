import os
import matplotlib.pyplot as plt

def save_distance_heatmap(dist_matrix, outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(dist_matrix, interpolation="nearest")
    plt.title("DeepFashion Lite (1000 images): Pairwise Cosine Distance")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "distance_matrix.png"))
    print("[INFO] Distance matrix heatmap saved.")
    plt.show()
