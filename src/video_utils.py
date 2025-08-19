import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .segmentation_utils import load_segmentation_model, segment_person_crop

def load_person_detection_model():
    """Load YOLOv8 person detection model."""
    model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt, yolov8l.pt for larger models
    return model

def extract_person_crops(frame, detections, min_confidence=0.5, use_segmentation=False, segmentation_model=None):
    """Extract person crops from frame based on detections."""
    person_crops = []
    person_boxes = []
    segmented_crops = []
    
    for detection in detections:
        # YOLOv8 detection format: [x1, y1, x2, y2, confidence, class]
        if len(detection) >= 6:
            x1, y1, x2, y2, conf, cls = detection[:6]
            
            # Class 0 is 'person' in COCO dataset
            if int(cls) == 0 and conf >= min_confidence:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Extract crop
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:  # Valid crop
                    person_crops.append(crop)
                    person_boxes.append((x1, y1, x2, y2, conf))
                    
                    # Apply segmentation if requested
                    if use_segmentation and segmentation_model is not None:
                        segmented_crop = segment_person_crop(crop, segmentation_model)
                        segmented_crops.append(segmented_crop)
                    else:
                        segmented_crops.append(crop)  # Use original crop if no segmentation
    
    return person_crops, person_boxes, segmented_crops

def compute_embedding_similarity(person_crop, embeddings, infer_fn):
    """Compute cosine distance between person crop and dataset embeddings (lower is better)."""
    
    # person_crop is already processed (segmented if needed)
    # Convert crop to PIL Image and preprocess
    person_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
    
    # Compute embedding for person crop
    person_embedding = infer_fn([person_image])
    
    # Normalize embedding (dataset embeddings are already normalized)
    person_embedding = person_embedding / np.linalg.norm(person_embedding, axis=1, keepdims=True)
    
    # Compute cosine similarity first
    cosine_similarities = np.dot(person_embedding, embeddings.T).flatten()
    
    # Convert to cosine distances (same metric as single image comparison)
    cosine_distances = 1.0 - cosine_similarities
    
    return cosine_distances

def get_top_k_similar_images(distances, images, k=5):
    """Get top K most similar images (lowest cosine distances are best)."""
    top_k_indices = np.argsort(distances)[:k]  # Get k lowest distances
    top_k_images = [images[i] for i in top_k_indices]
    top_k_distances = distances[top_k_indices]
    
    return top_k_images, top_k_distances, top_k_indices

def visualize_frame_results(frame, person_boxes, person_crops, top_k_results, frame_idx, output_dir, use_segmentation=False):
    """Visualize detection results and top similar images for a frame."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    n_persons = len(person_boxes)
    if n_persons == 0:
        return
    
    # Create wider figure to accommodate video crop display
    fig, axes = plt.subplots(n_persons, 7, figsize=(24, 4 * n_persons))
    if n_persons == 1:
        axes = axes.reshape(1, -1)
    
    for person_idx, (box, person_crop, (top_k_images, top_k_distances, top_k_indices)) in enumerate(zip(person_boxes, person_crops, top_k_results)):
        x1, y1, x2, y2, conf = box
        
        # Plot original frame with bounding box
        ax = axes[person_idx, 0]
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        title = f'Frame {frame_idx}\nPerson {person_idx+1} (conf: {conf:.2f})'
        ax.set_title(title)
        ax.axis('off')
        
        # Plot the person crop from video (segmented if use_segmentation=True)
        ax = axes[person_idx, 1]
        ax.imshow(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        crop_title = 'Video Person'
        if use_segmentation:
            crop_title += '\n(Segmented)'
        ax.set_title(crop_title)
        ax.axis('off')
        
        # Plot top 5 similar images from dataset
        for k in range(5):
            ax = axes[person_idx, k+2]
            if k < len(top_k_images):
                ax.imshow(top_k_images[k])
                title = f'Rank {k+1}\nDist: {top_k_distances[k]:.3f}'
                if use_segmentation:
                    title += '\n(Segmented)'
                ax.set_title(title)
            ax.axis('off')
    
    plt.tight_layout()
    suffix = '_segmented' if use_segmentation else ''
    plt.savefig(os.path.join(output_dir, f'frame_{frame_idx:04d}_results{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def process_video_with_person_detection(video_path, output_dir, embeddings, images, infer_fn, top_k=5, use_segmentation=False):
    """Process video with person detection and find similar fashion images."""
    
    # Load person detection model
    print("[INFO] Loading YOLOv8 person detection model...")
    detection_model = load_person_detection_model()
    
    # Load segmentation model if needed
    segmentation_model = None
    if use_segmentation:
        print("[INFO] Loading YOLOv8 large segmentation model for video processing...")
        segmentation_model = load_segmentation_model()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    frame_idx = 0
    processed_frames = 0
    
    print(f"[INFO] Processing video frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run person detection every 30 frames (approximately 1 second at 30fps)
        if frame_idx % 30 == 0:
            # Run YOLO detection
            results = detection_model(frame)
            
            # Extract detections
            if len(results) > 0 and results[0].boxes is not None:
                detections = results[0].boxes.data.cpu().numpy()
                
                # Extract person crops
                person_crops, person_boxes, segmented_crops = extract_person_crops(
                    frame, detections, use_segmentation=use_segmentation, 
                    segmentation_model=segmentation_model
                )
                
                if person_crops:
                    top_k_results = []
                    
                    # For each detected person, find similar images
                    for segmented_crop in segmented_crops:
                        # Compute distances (using cosine distance like single image comparison)
                        distances = compute_embedding_similarity(
                            segmented_crop, embeddings, infer_fn
                        )
                        
                        # Get top K similar images (lowest distances)
                        top_k_images, top_k_distances, top_k_indices = get_top_k_similar_images(
                            distances, images, k=top_k
                        )
                        
                        top_k_results.append((top_k_images, top_k_distances, top_k_indices))
                    
                    # Visualize results
                    visualize_frame_results(
                        frame, person_boxes, segmented_crops, top_k_results, frame_idx, output_dir,
                        use_segmentation=use_segmentation
                    )
                    
                    seg_info = " with segmentation" if use_segmentation else ""
                    print(f"[INFO] Processed frame {frame_idx}, found {len(person_crops)} persons{seg_info}")
                    processed_frames += 1
        
        frame_idx += 1
    
    cap.release()
    seg_info = " with segmentation" if use_segmentation else ""
    print(f"[INFO] Video processing complete{seg_info}. Processed {processed_frames} frames with detections.")
