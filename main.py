import os
import numpy as np
import argparse
from src.data_utils import load_deepfashion_images, save_images
from src.model_utils import load_onnx_or_hf_model
from src.retrieval_utils import l2_normalize, save_embeddings, compute_distance_matrix, save_retrieval_results
from src.visualization_utils import save_distance_heatmap
from src.video_utils import process_video_with_person_detection
from src.segmentation_utils import create_segmented_images


def main():
    parser = argparse.ArgumentParser(description="Fashion Embedding Pipeline")
    parser.add_argument("--video_clip", type=str, help="Name of video clip to process (e.g., clip_1)")
    parser.add_argument("--video_dir", type=str, default="videos", help="Directory containing video files")
    parser.add_argument("--use-embeddings", action="store_true", help="Use pre-computed embeddings and images from embeddings/ and images/ directories")
    parser.add_argument("--use-segmentation", action="store_true", help="Use person segmentation to create segmented images with white background")
    parser.add_argument("--use-embedded-segment", action="store_true", help="Use pre-computed segmented embeddings and images from embeddings_segment/ and images_segment/ directories")
    args = parser.parse_args()

    # Setup
    print("[INFO] Setting up cache directories...")
    root_cache = os.path.join(os.getcwd(), "cache")
    datasets_cache = os.path.join(root_cache, "datasets")

    if args.use_embedded_segment:
        # Load pre-computed segmented embeddings and images
        print("[INFO] Using pre-computed segmented embeddings and images...")
        print("[INFO] Note: Segmented dataset contains individual person crops")
        
        # Load embeddings
        embeddings_dir = os.path.join(os.getcwd(), "embeddings_segment")
        embeddings_path = os.path.join(embeddings_dir, "embeddings.npy")
        if not os.path.exists(embeddings_path):
            print(f"[ERROR] Segmented embeddings file not found: {embeddings_path}")
            return
        
        embeds = np.load(embeddings_path)
        print(f"[INFO] Loaded {embeds.shape[0]} segmented person embeddings from {embeddings_path}")
        
        # Load images list (we need to reconstruct the images list)
        images_dir = os.path.join(os.getcwd(), "images_segment")
        if not os.path.exists(images_dir):
            print(f"[ERROR] Segmented images directory not found: {images_dir}")
            return
        
        # Get list of image files and load them
        from PIL import Image
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        images = []
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            img = Image.open(img_path)
            images.append(img)
        
        print(f"[INFO] Loaded {len(images)} segmented person images from {images_dir}")
        
        # We still need to load the model for video processing if needed
        if args.video_clip and args.video_dir:
            onnx_dir = os.path.join(os.getcwd(), "onnx")
            int8_onnx_path = os.path.join(onnx_dir, "fashion_clip_image_int8_matmul.onnx")
            try:
                import bitsandbytes
                bitsandbytes_available = True
            except ImportError:
                bitsandbytes_available = False

            infer_fn, model, processor = load_onnx_or_hf_model(int8_onnx_path, bitsandbytes_available)
        else:
            infer_fn = None
    elif args.use_embeddings:
        # Load pre-computed embeddings and images
        print("[INFO] Using pre-computed embeddings and images...")
        
        # Load embeddings
        if args.use_segmentation:
            embeddings_dir = os.path.join(os.getcwd(), "embeddings_segment")
        else:
            embeddings_dir = os.path.join(os.getcwd(), "embeddings")
            
        embeddings_path = os.path.join(embeddings_dir, "embeddings.npy")
        if not os.path.exists(embeddings_path):
            print(f"[ERROR] Embeddings file not found: {embeddings_path}")
            return
        
        embeds = np.load(embeddings_path)
        print(f"[INFO] Loaded {embeds.shape[0]} embeddings from {embeddings_path}")
        
        # Load images list (we need to reconstruct the images list)
        if args.use_segmentation:
            images_dir = os.path.join(os.getcwd(), "images_segment")
        else:
            images_dir = os.path.join(os.getcwd(), "images")
            
        if not os.path.exists(images_dir):
            print(f"[ERROR] Images directory not found: {images_dir}")
            return
        
        # Get list of image files and load them
        from PIL import Image
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        images = []
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            img = Image.open(img_path)
            images.append(img)
        
        print(f"[INFO] Loaded {len(images)} images from {images_dir}")
        
        # We still need to load the model for video processing if needed
        if args.video_clip and args.video_dir:
            onnx_dir = os.path.join(os.getcwd(), "onnx")
            int8_onnx_path = os.path.join(onnx_dir, "fashion_clip_image_int8_matmul.onnx")
            try:
                import bitsandbytes
                bitsandbytes_available = True
            except ImportError:
                bitsandbytes_available = False

            infer_fn, model, processor = load_onnx_or_hf_model(int8_onnx_path, bitsandbytes_available)
        else:
            infer_fn = None
    else:
        # Original pipeline: compute embeddings from scratch
        # Load images and compute embeddings for the dataset
        data_split = "train[:10000]"
        print(f"[INFO] Loading DeepFashion dataset split: {data_split}")
        original_images = load_deepfashion_images(split=data_split, cache_dir=datasets_cache)
        print(f"[INFO] Loaded {len(original_images)} original images.")

        # Save images
        if args.use_segmentation:
            # Create segmented images with white background (one image per person)
            images_dir = os.path.join(os.getcwd(), "images_segment")
            print("[INFO] Creating segmented person crops with white background...")
            print("[INFO] Note: Each original image may produce multiple person crops")
            segmented_images = create_segmented_images(original_images, images_dir, max_workers=8)
            images = segmented_images  # Use segmented images for embedding computation
            print(f"[INFO] Total person crops created: {len(images)} from {len(original_images)} original images")
        else:
            images_dir = os.path.join(os.getcwd(), "images")
            print("[INFO] Saving original images...")
            save_images(original_images, images_dir, max_workers=8)
            images = original_images
            print("[INFO] Original images saved.")

        # Model selection and embedding
        onnx_dir = os.path.join(os.getcwd(), "onnx")
        int8_onnx_path = os.path.join(onnx_dir, "fashion_clip_image_int8_matmul.onnx")
        try:
            import bitsandbytes
            bitsandbytes_available = True
        except ImportError:
            bitsandbytes_available = False

        infer_fn, model, processor = load_onnx_or_hf_model(int8_onnx_path, bitsandbytes_available)
        print("[INFO] Computing embeddings for person crops...")
        embeds = infer_fn(images)
        print(f"[INFO] Embeddings computed for {len(images)} person crops.")

        # Normalize and save embeddings
        print("[INFO] L2-normalizing embeddings...")
        embeds = l2_normalize(embeds)
        print("[INFO] Embeddings normalized.")
        
        if args.use_segmentation:
            embeddings_dir = os.path.join(os.getcwd(), "embeddings_segment")
            print(f"[INFO] Saving {len(embeds)} person embeddings to segmented directory...")
        else:
            embeddings_dir = os.path.join(os.getcwd(), "embeddings")
            print(f"[INFO] Saving {len(embeds)} image embeddings...")
        
        save_embeddings(embeds, embeddings_dir)
        print("[INFO] Embeddings saved.")

    # Video processing if video arguments are provided
    if args.video_clip and args.video_dir:
        if infer_fn is None:
            print("[ERROR] Cannot process video: model not loaded. Use without --use-embeddings flag or ensure model loading works.")
            return
            
        video_path = os.path.join(args.video_dir, f"{args.video_clip}.mp4")
        output_dir = os.path.join("video_results", args.video_clip)
        
        # Determine if we should use segmentation for video processing
        use_segmentation_for_video = args.use_segmentation or args.use_embedded_segment
        
        print(f"[INFO] Processing video: {video_path}")
        if use_segmentation_for_video:
            print("[INFO] Using segmentation for video processing...")
            
        process_video_with_person_detection(
            video_path=video_path,
            output_dir=output_dir,
            embeddings=embeds,
            images=images,
            infer_fn=infer_fn,
            top_k=5,
            use_segmentation=use_segmentation_for_video
        )
        print(f"[INFO] Video processing completed. Results saved to {output_dir}")
    else:
        # Original pipeline without video processing
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

if __name__ == "__main__":
    main()
