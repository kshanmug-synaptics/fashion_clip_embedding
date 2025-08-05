
# Fashion Embedding Pipeline

This project provides a pipeline for extracting image embeddings from the DeepFashion dataset using an ONNX or HuggingFace model, saving the embeddings, computing pairwise distances, and visualizing the results.

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Usage

Run the main pipeline with:

```bash
python main.py
```

## What Happens When You Run It

- **Images are loaded** from the DeepFashion dataset (first 1000 images by default).
- **Images are saved** to the `images/` directory.
- **Embeddings are computed** using the specified Quantized ONNX (if present) or HuggingFace model.
- **Embeddings are normalized** and saved to the `embeddings/` directory.
- **Pairwise cosine distance matrix** is computed and saved.
- **Retrieval results** are saved to the `retrieval_results/` directory.
- **A heatmap visualization** of the distance matrix is saved to the `outputs/` directory.

## Output Directories

- `images/` — Saved images from the dataset
- `embeddings/` — Numpy arrays of image embeddings
- `retrieval_results/` — Retrieval results for each image
- `outputs/` — Heatmap visualizations of the distance matrix

## Notes
- Make sure the ONNX model file (`fashion_clip_image_int8_matmul.onnx`) is present in the `onnx/` directory.
- The script will automatically create necessary directories if they do not exist.
