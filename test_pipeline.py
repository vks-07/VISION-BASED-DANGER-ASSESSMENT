import os
import sys

# Add the src directory to the system path so we can import our modules
base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_dir)

from datasets import load_dataset
from src.data_pipeline.download_videos import download_all_videos
from src.data_pipeline.preprocess import process_video_segment
from src.feature_extraction.extract_visual import extract_visual_features
from src.feature_extraction.extract_text import extract_text_features
from src.model.train import train_pipeline

def test_entire_pipeline():
    """
    Runs a single video completely through the pipeline from download to PyTorch training 
    to verify everything connects correctly.
    """
    print("=========================================")
    print(" 🚀 STARTING END-TO-END PIPELINE TEST 🚀 ")
    print("=========================================\n")
    
    # 1. Download just the very first video
    print(">>> STEP 1: DOWNLOADING VIDEO")
    raw_vids_dir = os.path.join(base_dir, "data", "raw_videos")
    download_all_videos(raw_vids_dir, limit=1)
    
    # 2. Get the metadata for that exact video from Hugging Face
    print("\n>>> STEP 2: FETCHING GROUND TRUTH ANNOTATIONS")
    dataset = load_dataset("pranked03/ViDAS", split="train")
    first_row = dataset[0]
    
    vid_id = first_row['video_id']          # e.g., '0.mp4'
    start_t = first_row['start_coord']      # e.g., 13.0
    end_t = first_row['end_coord']          # e.g., 25.0
    summary = first_row['video_summary']    # Text description
    
    print(f"Loaded {vid_id}: Extracting from {start_t}s to {end_t}s")
    
    # 3. Preprocess: Extract exactly 50 frames using OpenCV
    print("\n>>> STEP 3: PREPROCESSING (OPENCV)")
    vid_path = os.path.join(raw_vids_dir, vid_id)
    frames_dir = os.path.join(base_dir, "data", "processed_frames", vid_id.replace('.mp4', ''))
    
    process_video_segment(vid_path, frames_dir, start_t, end_t, num_frames=50)
    
    # 4. Feature Extraction: Pass frames to CLIP and text to BERT
    print("\n>>> STEP 4: FEATURE EXTRACTION (CLIP & BERT)")
    visual_out = os.path.join(base_dir, "data", "embeddings", f"{vid_id}_visual.npy")
    text_out = os.path.join(base_dir, "data", "embeddings", f"{vid_id}_text.npy")
    
    extract_visual_features(frames_dir, visual_out)
    extract_text_features(summary, text_out)
    
    # 5. Model Training: Pass through our PyTorch MLP
    print("\n>>> STEP 5: PYTORCH MODEL TRAINING")
    # This will load the dataset class (which will find the 1 valid .npy file we just made)
    # and run a 10-epoch training loop, proving the loss calculates correctly.
    train_pipeline()
    
    print("\n=========================================")
    print(" ✅ TEST COMPLETE: PIPELINE IS WORKING ✅ ")
    print("=========================================")

if __name__ == "__main__":
    test_entire_pipeline()
