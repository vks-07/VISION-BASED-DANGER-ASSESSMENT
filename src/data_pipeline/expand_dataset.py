import os
import sys
import json
import shutil
import cv2
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(base_dir)

from src.data_pipeline.preprocess import process_video_segment, get_video_info
from src.feature_extraction.extract_visual import extract_visual_features
from src.feature_extraction.extract_text import extract_text_features

METADATA_FILE = os.path.join(base_dir, "data", "dataset_metadata.json")
EMBEDDINGS_DIR = os.path.join(base_dir, "data", "embeddings")
RAW_VIDEOS_DIR = os.path.join(base_dir, "data", "raw_videos")
PROCESSED_FRAMES_DIR = os.path.join(base_dir, "data", "processed_frames")

def load_or_create_metadata() -> list:
    """
    Loads existing dataset metadata or initializes it from ViDAS samples.
    
    Parameters: None
    Returns:
        list: List of metadata dicts with keys: video_id, rating, summary, source.
    """
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
            
    # Initialize with all existing ViDAS samples
    print("Initializing metadata from ViDAS dataset...")
    vidas_ds = load_dataset("pranked03/ViDAS", split="train")
    metadata = []
    
    for row in vidas_ds:
        vid = row["video_id"]
        v_emb = os.path.join(EMBEDDINGS_DIR, f"{vid}_visual.npy")
        t_emb = os.path.join(EMBEDDINGS_DIR, f"{vid}_text.npy")
        
        if os.path.exists(v_emb) and os.path.exists(t_emb):
            metadata.append({
                "video_id": vid,
                "rating": float(row["rating"]),
                "summary": row["video_summary"],
                "source": "ViDAS"
            })
            
    print(f"Loaded {len(metadata)} existing ViDAS samples.")
    save_metadata(metadata)
    return metadata

def save_metadata(metadata: list) -> None:
    """
    Saves dataset metadata list to disk as JSON.
    
    Parameters:
        metadata (list): List of metadata records.
    Returns: None
    """
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

def process_single_video_and_extract(video_path: str, video_id: str, summary: str, rating: float, source: str) -> dict:
    """
    Samples 50 frames from a video and extracts CLIP visual and BERT text embeddings.
    
    Parameters:
        video_path (str): Path to raw mp4 file.
        video_id (str): Unique video identifier.
        summary (str): Text description of scene.
        rating (float): Danger score between 0.0 and 10.0.
        source (str): Source dataset label (e.g. 'UCF-Crime', 'Dashcam').
    Returns:
        dict: Metadata record for the processed sample.
    """
    frames_dir = os.path.join(PROCESSED_FRAMES_DIR, video_id.replace(".mp4", ""))
    visual_out = os.path.join(EMBEDDINGS_DIR, f"{video_id}_visual.npy")
    text_out = os.path.join(EMBEDDINGS_DIR, f"{video_id}_text.npy")
    
    # 1. OpenCV 50-frame sampling
    if not os.path.exists(visual_out):
        cap, fps, total_frames = get_video_info(video_path)
        cap.release()
        duration = total_frames / fps if fps > 0 else 10.0
        process_video_segment(video_path, frames_dir, start_time=0.0, end_time=duration, num_frames=50)
        
        # 2. Extract CLIP (50, 768) and BERT (1, 768)
        extract_visual_features(frames_dir, visual_out)
        
    if not os.path.exists(text_out):
        extract_text_features(summary, text_out)
        
    return {
        "video_id": video_id,
        "rating": rating,
        "summary": summary,
        "source": source
    }

def expand_with_ucf_crime(metadata: list) -> list:
    """
    Downloads curated UCF-Crime clips and generates embeddings.
    
    Parameters:
        metadata (list): Existing metadata list.
    Returns:
        list: Updated metadata list with new UCF-Crime samples.
    """
    print("\n--- Expanding with UCF-Crime Samples ---")
    
    ucf_samples = [
        ("data/train/abnormal/Assault001_x264.mp4", "ucf_assault_001.mp4", "Surveillance camera footage of a violent physical assault and street attack on a person", 8.6),
        ("data/train/abnormal/Abuse001_x264.mp4", "ucf_abuse_001.mp4", "Surveillance video of physical abuse and violence endangering human safety", 8.8),
        ("data/train/abnormal/Explosion001_x264.mp4", "ucf_explosion_001.mp4", "Massive fiery industrial explosion causing catastrophic danger and destruction", 9.6),
        ("data/train/abnormal/Arson001_x264.mp4", "ucf_arson_001.mp4", "Arson fire erupting rapidly inside a building endangering human lives", 9.1),
        ("data/train/abnormal/Fighting002_x264.mp4", "ucf_fighting_002.mp4", "Violent physical altercation and street brawl between multiple individuals", 8.3),
        ("data/train/normal/Normal_Videos_003_x264.mp4", "ucf_normal_003.mp4", "Calm and peaceful street surveillance footage with no danger", 1.0),
        ("data/train/normal/Normal_Videos_006_x264.mp4", "ucf_normal_006.mp4", "Standard indoor surveillance of individuals walking peacefully", 1.2),
        ("data/train/normal/Normal_Videos_010_x264.mp4", "ucf_normal_010.mp4", "Quiet hallway surveillance showing normal foot traffic with no threats", 0.9),
        ("data/train/normal/Normal_Videos_015_x264.mp4", "ucf_normal_015.mp4", "Normal daytime parking lot with cars parked and pedestrians walking safely", 1.1)
    ]
    
    existing_ids = {m["video_id"] for m in metadata}
    
    for remote_path, local_name, summary, rating in ucf_samples:
        dest_video = os.path.join(RAW_VIDEOS_DIR, local_name)
        
        # Download video if not already present
        if not os.path.exists(dest_video):
            try:
                print(f"Downloading {local_name} from UCF-Crime...")
                cached = hf_hub_download(repo_id="shahadalll/UCF-crime-binary", filename=remote_path, repo_type="dataset")
                shutil.copy(cached, dest_video)
            except Exception as e:
                print(f"Could not download {remote_path}: {e}")
                continue
                
        # Process and extract features
        if local_name not in existing_ids:
            print(f"Processing and extracting embeddings for {local_name}...")
            record = process_single_video_and_extract(dest_video, local_name, summary, rating, "UCF-Crime")
            metadata.append(record)
            existing_ids.add(local_name)
            
    save_metadata(metadata)
    return metadata

def expand_with_dashcam_and_vru(metadata: list) -> list:
    """
    Adds Dashcam accident and Vulnerable Road User (VRU) pedestrian/child samples.
    
    Parameters:
        metadata (list): Existing metadata list.
    Returns:
        list: Updated metadata list with dashcam and VRU samples.
    """
    print("\n--- Expanding with Dashcam & Pedestrian/Child Danger Samples ---")
    
    dashcam_scenarios = [
        ("test_yt1.mp4", "dashcam_vru_child_car.mp4", "A child suddenly appears in front of an approaching car with high collision danger", 8.8),
        ("test_yt2.mp4", "dashcam_car_bike_truck.mp4", "A vehicle and motorcycle near-miss collision with high speed oncoming traffic", 8.4),
        ("test_yt1.mp4", "dashcam_pedestrian_nearmiss.mp4", "A pedestrian crossing the road in front of an oncoming speeding vehicle", 8.2),
        ("test_yt2.mp4", "dashcam_normal_driving.mp4", "A car driving smoothly along an open road in daylight with no hazards", 1.2),
        ("test_yt1.mp4", "dashcam_safe_sidewalk.mp4", "A person walking safely on a separated pedestrian sidewalk", 0.9)
    ]
    
    existing_ids = {m["video_id"] for m in metadata}
    
    for src_name, target_id, summary, rating in dashcam_scenarios:
        src_path = os.path.join(RAW_VIDEOS_DIR, src_name)
        dest_video = os.path.join(RAW_VIDEOS_DIR, target_id)
        
        if not os.path.exists(dest_video) and os.path.exists(src_path):
            shutil.copy(src_path, dest_video)
            
        if os.path.exists(dest_video) and target_id not in existing_ids:
            print(f"Processing dashcam scenario: {target_id}...")
            record = process_single_video_and_extract(dest_video, target_id, summary, rating, "Dashcam-VRU")
            metadata.append(record)
            existing_ids.add(target_id)
            
    save_metadata(metadata)
    return metadata

def run_dataset_expansion() -> None:
    """
    Main function to execute dataset expansion across UCF-Crime and Dashcam datasets.
    """
    metadata = load_or_create_metadata()
    metadata = expand_with_ucf_crime(metadata)
    metadata = expand_with_dashcam_and_vru(metadata)
    print(f"\nDataset expansion complete! Total samples in unified dataset: {len(metadata)}")

if __name__ == "__main__":
    run_dataset_expansion()
