import os
import sys

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_dir)

from datasets import load_dataset
from src.data_pipeline.download_videos import download_all_videos
from src.data_pipeline.preprocess import process_video_segment
from src.feature_extraction.extract_visual import extract_visual_features
from src.feature_extraction.extract_text import extract_text_features
from src.data_pipeline.expand_dataset import run_dataset_expansion
from src.data_pipeline.integrate_dada2000 import run_dada_integration
from src.model.train import train_pipeline

def train_on_full_dataset():
    """
    Orchestrates full dataset pipeline: ViDAS + UCF-Crime + Dashcam + DADA-2000,
    extracts all multimodal embeddings, and trains the neural network.
    
    Parameters: None
    Returns: None
    """
    print("=" * 65)
    print("STARTING FULL DATASET PIPELINE (ViDAS + UCF-Crime + DADA-2000)")
    print("=" * 65)
    
    raw_vids_dir = os.path.join(base_dir, "data", "raw_videos")
    embeddings_dir = os.path.join(base_dir, "data", "embeddings")
    os.makedirs(raw_vids_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # 1. ViDAS Dataset
    print("\n[Phase 1/4] Preparing ViDAS dataset...")
    download_all_videos(raw_vids_dir, limit=0)
    
    dataset = load_dataset("pranked03/ViDAS", split="train")
    for row in dataset:
        vid_id = row['video_id']
        start_t = row['start_coord']
        end_t = row['end_coord']
        summary = row['video_summary']
        vid_path = os.path.join(raw_vids_dir, vid_id)
        
        if not os.path.exists(vid_path):
            continue
            
        frames_dir = os.path.join(base_dir, "data", "processed_frames", vid_id.replace('.mp4', ''))
        visual_out = os.path.join(embeddings_dir, f"{vid_id}_visual.npy")
        text_out = os.path.join(embeddings_dir, f"{vid_id}_text.npy")
        
        if not os.path.exists(visual_out) or not os.path.exists(text_out):
            process_video_segment(vid_path, frames_dir, start_t, end_t, num_frames=50)
            if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0:
                extract_visual_features(frames_dir, visual_out)
                extract_text_features(summary, text_out)

    # 2. UCF-Crime & Dashcam-VRU Datasets
    print("\n[Phase 2/4] Integrating UCF-Crime and Dashcam-VRU datasets...")
    run_dataset_expansion()

    # 3. DADA-2000 Benchmark (Pedestrian, Bike, Truck, Animal)
    print("\n[Phase 3/4] Integrating DADA-2000 accident benchmarks...")
    run_dada_integration()

    # 4. Neural Network Training on Whole Dataset
    print("\n[Phase 4/4] Training model on whole unified dataset...")
    train_pipeline()
    print("\nWhole dataset training completed successfully!")

if __name__ == "__main__":
    train_on_full_dataset()
