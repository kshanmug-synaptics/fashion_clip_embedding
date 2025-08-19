
# Fashion Embedding Pipeline

What Happens When You Run It

### Output Directories

- `images/` — Saved images from the DeepFashion dataset
- `images_segment/` — Individual person crops with white background (one file per detected person)
- `embeddings/` — Numpy arrays of image embeddings
- `embeddings_segment/` — Numpy arrays of individual person embeddings
- `retrieval_results/` — Retrieval results for each image
- `outputs/` — Heatmap visualizations of the distance matrix
- `video_results/` — Video processing results with person detection and similar fashion images
- `video_results/clip_1/` — Results for specific video clip with frame-by-frame analysisndard Pipeline (`python main.py`)

1. **Set up cache directories** for datasets and outputs if they do not exist.
2. **Load the first 10000 images** from the **DeepFashion** dataset.
3. **Save the loaded images** to the `images/` directory.
4. **Check for the `bitsandbytes` library** to determine model loading options.
5. **Load the ONNX or HuggingFace model** for extracting image embeddings.
6. **Compute image embeddings** for all loaded DeepFashion images using the selected model.
7. **L2-normalize the embeddings** to prepare them for distance computation.
8. **Save the normalized embeddings** to the `embeddings/` directory.
9. **Compute the pairwise cosine distance matrix** between all DeepFashion image embeddings.
10. **Save retrieval results** (e.g., nearest neighbors) to the `retrieval_results/` directory.
11. **Visualize the distance matrix** as a heatmap and save it to the `outputs/` directory.

### Segmentation Pipeline (`python main.py --use-segmentation`)

1. **Load the DeepFashion dataset** (steps 1-2 from standard pipeline).
2. **Load YOLOv8 large segmentation model** for person segmentation.
3. **Segment each image** to detect and isolate individual persons.
4. **Create separate crops** for each detected person with white background for non-person areas.
5. **Save individual person crops** to the `images_segment/` directory (Note: this creates more images than the original dataset).
6. **Compute embeddings** on each individual person crop using the CLIP model.
7. **Save person embeddings** to the `embeddings_segment/` directory.
8. **Continue with standard distance matrix and visualization** (steps 9-11).

**Important**: The segmentation process creates **one image per detected person**, so if the original dataset has multiple people per image, the segmented dataset will contain more images than the original.

### Video Processing with Segmentation (`python main.py --video_clip clip_1 --video_dir videos --use-segmentation`)

1. **Create individual person crops** from the dataset (steps 1-6 from segmentation pipeline).
2. **Load YOLOv8 detection and segmentation models**.
3. **Process video frames** with person detection and segmentation.
4. **For each detected person**: Apply segmentation to create white background within the detection crop.
5. **Compute embeddings** for segmented person crops from video.
6. **Find similar individual person crops** from the segmented dataset.
7. **Create visualizations** showing detection results and top similar person matches.vides a pipeline for extracting image embeddings from the DeepFashion dataset using an ONNX or HuggingFace model, saving the embeddings, computing pairwise distances, and visualizing the results.

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies:**

```bash
git clone git@gitlab.synaptics.com:kshanmug/fashion_embedding.git
cd fashion_embedding
virtualvenv .venv
source .venv/bin/activate (in Linux)
pip install -r requirements.txt
```

## Usage

### Standard Pipeline
Run the main pipeline with:

```bash
python main.py
```

### Using Pre-computed Embeddings
To use pre-computed embeddings and images (faster execution):

```bash
python main.py --use-embeddings
```

### Creating Segmented Images and Embeddings
To create segmented versions of images with person segmentation and white background:

```bash
python main.py --use-segmentation
```

### Using Pre-computed Segmented Embeddings
To use pre-computed segmented embeddings and images:

```bash
python main.py --use-embedded-segment
```

### Video Processing with Person Detection
To process a video with person detection and find similar fashion images:

```bash
python main.py --video_clip clip_1 --video_dir videos
```

### Video Processing with Pre-computed Embeddings
To process a video using pre-computed embeddings (faster):

```bash
python main.py --video_clip clip_1 --video_dir videos --use-embeddings
```

### Video Processing with Segmentation
To process a video with person segmentation for improved accuracy:

```bash
python main.py --video_clip clip_1 --video_dir videos --use-segmentation
```

### Video Processing with Pre-computed Segmented Embeddings
To process a video using pre-computed segmented embeddings (fastest and most accurate):

```bash
python main.py --video_clip clip_1 --video_dir videos --use-embedded-segment
```

This will:
- Process the video `videos/clip_1.mp4`
- Detect persons in video frames (every 30 frames)
- For each detected person, find the top 5 most similar images from the DeepFashion dataset
- Save visualization results to `video_results/clip_1/`

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
- For video processing, place your video file (e.g., `clip_1.mp4`) in the `videos/` directory.
- The YOLOv8 model will be automatically downloaded on first use.
- Video processing analyzes every 30th frame for efficiency.
- The script will automatically create necessary directories if they do not exist.
