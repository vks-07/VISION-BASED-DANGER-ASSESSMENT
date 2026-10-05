import os
import sys

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_dir)

from datasets import load_dataset
from src.data_pipeline.download_videos import download_all_videos
from src.data_pipeline.preprocess import process_video_segment
from src.feature_extraction.extract_visual import extract_visual_features
from src.feature_extraction.extract_text import extract_text_features
from src.model.train import train_pipeline

def train_on_full_dataset():
    """
    Downloads the dataset, processes videos, and trains the model.
    """
    print("Starting full dataset training pipeline...")
    
    print("Step 1: Downloading videos")
    raw_vids_dir = os.path.join(base_dir, "data", "raw_videos")
    download_all_videos(raw_vids_dir, limit=0)
    
    print("Step 2: Preprocessing and feature extraction")
    dataset = load_dataset("pranked03/ViDAS", split="train")
    
    valid_samples = 0
    for row in dataset:
        vid_id = row['video_id']
        start_t = row['start_coord']
        end_t = row['end_coord']
        summary = row['video_summary']
        
        vid_path = os.path.join(raw_vids_dir, vid_id)
        
        # Skip if yt-dlp failed to download
        if not os.path.exists(vid_path):
            continue 
            
        frames_dir = os.path.join(base_dir, "data", "processed_frames", vid_id.replace('.mp4', ''))
        visual_out = os.path.join(base_dir, "data", "embeddings", f"{vid_id}_visual.npy")
        text_out = os.path.join(base_dir, "data", "embeddings", f"{vid_id}_text.npy")
        
        # Only process if embeddings don't exist
        if not os.path.exists(visual_out) or not os.path.exists(text_out):
            print(f"\n[Processing {vid_id}]")
            process_video_segment(vid_path, frames_dir, start_t, end_t, num_frames=50)
            
            if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0:
                extract_visual_features(frames_dir, visual_out)
                extract_text_features(summary, text_out)
            else:
                print(f"Skipping feature extraction for {vid_id} (No frames extracted).")
                continue
                
        valid_samples += 1
        
    print(f"Successfully prepared {valid_samples} videos for training.")
    
    print("Step 3: Training model")
    train_pipeline()

if __name__ == "__main__":
    train_on_full_dataset()
