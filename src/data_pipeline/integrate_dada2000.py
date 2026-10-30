import os
import sys
import json
import zlib
import struct
import io
import urllib.request
import zipfile
import cv2
import numpy as np

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(base_dir)

from src.data_pipeline.preprocess import process_video_segment, get_video_info
from src.feature_extraction.extract_visual import extract_visual_features
from src.feature_extraction.extract_text import extract_text_features

METADATA_FILE = os.path.join(base_dir, "data", "dataset_metadata.json")
EMBEDDINGS_DIR = os.path.join(base_dir, "data", "embeddings")
RAW_VIDEOS_DIR = os.path.join(base_dir, "data", "raw_videos")
PROCESSED_FRAMES_DIR = os.path.join(base_dir, "data", "processed_frames")
VRU_ZIP_URL = "https://huggingface.co/datasets/kyh9191/VRU-Accident/resolve/main/VRU_videos.zip"

class RemoteZipStream(io.RawIOBase):
    """
    HTTP Range-based seekable file stream to read zip central directory without downloading full archive.
    """
    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        req.get_method = lambda: 'HEAD'
        with urllib.request.urlopen(req) as resp:
            self.length = int(resp.headers.get('Content-Length', 0))
        self.pos = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size == -1 or self.pos + size > self.length:
            size = self.length - self.pos
        if size <= 0:
            return b''
        req = urllib.request.Request(self.url, headers={
            'Range': f'bytes={self.pos}-{self.pos + size - 1}',
            'User-Agent': 'Mozilla/5.0'
        })
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        return data

def extract_single_video_from_remote_zip(stream: RemoteZipStream, zf: zipfile.ZipFile, zip_path: str, dest_path: str) -> bool:
    """
    Downloads a single video file from the remote zip via a single HTTP Range request.
    
    Parameters:
        stream (RemoteZipStream): Range-enabled seekable HTTP stream.
        zf (zipfile.ZipFile): Initialized zipfile reader.
        zip_path (str): Internal relative path inside the zip.
        dest_path (str): Destination file path on local disk.
    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return True
        
    try:
        info = zf.getinfo(zip_path)
        # Read local header to determine file data offset
        stream.seek(info.header_offset)
        header = stream.read(30)
        _, _, _, _, _, _, _, _, _, fn_len, extra_len = struct.unpack('<IHHHHHIIIHH', header)
        data_offset = info.header_offset + 30 + fn_len + extra_len
        
        # Stream the exact file bytes in a single HTTP range request
        req = urllib.request.Request(stream.url, headers={
            'Range': f'bytes={data_offset}-{data_offset + info.compress_size - 1}',
            'User-Agent': 'Mozilla/5.0'
        })
        with urllib.request.urlopen(req) as resp:
            raw_bytes = resp.read()
            
        # Decompress if deflated
        if info.compress_type == 8:
            decompressed = zlib.decompress(raw_bytes, -15)
        else:
            decompressed = raw_bytes
            
        with open(dest_path, "wb") as f:
            f.write(decompressed)
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def process_and_cache_features(video_path: str, video_id: str, summary: str, rating: float, source: str) -> dict:
    """
    Samples 50 frames with OpenCV, extracts CLIP and BERT vectors, and caches them to disk.
    
    Parameters:
        video_path (str): Path to raw mp4 file.
        video_id (str): Unique video ID string.
        summary (str): Scene description string.
        rating (float): Danger score between 0.0 and 10.0.
        source (str): Source dataset label.
    Returns:
        dict: Metadata dictionary for the record.
    """
    frames_dir = os.path.join(PROCESSED_FRAMES_DIR, video_id.replace(".mp4", ""))
    visual_out = os.path.join(EMBEDDINGS_DIR, f"{video_id}_visual.npy")
    text_out = os.path.join(EMBEDDINGS_DIR, f"{video_id}_text.npy")

    if not os.path.exists(visual_out):
        cap, fps, total_frames = get_video_info(video_path)
        cap.release()
        duration = total_frames / fps if fps > 0 else 10.0
        process_video_segment(video_path, frames_dir, start_time=0.0, end_time=duration, num_frames=50)
        extract_visual_features(frames_dir, visual_out)

    if not os.path.exists(text_out):
        extract_text_features(summary, text_out)

    return {
        "video_id": video_id,
        "rating": rating,
        "summary": summary,
        "source": source
    }

def get_dada2000_10_samples_per_type() -> dict:
    """
    Defines exactly 10 curated videos for each of the 4 requested accident scenarios.
    
    Parameters: None
    Returns:
        dict: Dictionary mapping category names to list of (remote_zip_path, local_filename, summary, rating).
    """
    return {
        "pedestrians_crossing": [
            ("VRU_videos/DADA_2000/VRU_1.mp4", "dada_ped_cross_01.mp4", "A pedestrian steps abruptly onto the roadway directly into the vehicle lane causing an immediate collision", 8.9),
            ("VRU_videos/DADA_2000/VRU_2.mp4", "dada_ped_cross_02.mp4", "A pedestrian crossing an urban street is struck by an approaching car braking abruptly", 8.8),
            ("VRU_videos/DADA_2000/VRU_4.mp4", "dada_ped_cross_03.mp4", "An ego-vehicle collides with a pedestrian running across a multi-lane roadway", 9.1),
            ("VRU_videos/DADA_2000/VRU_5.mp4", "dada_ped_cross_04.mp4", "A person crossing the road from the shoulder is hit by an oncoming vehicle with high impact", 8.7),
            ("VRU_videos/DADA_2000/VRU_7.mp4", "dada_ped_cross_05.mp4", "Vehicle strikes a pedestrian stepping from between parked cars into oncoming traffic", 8.6),
            ("VRU_videos/DADA_2000/VRU_8.mp4", "dada_ped_cross_06.mp4", "A pedestrian in dark clothing crossing a street at night is struck by a moving car", 9.0),
            ("VRU_videos/DADA_2000/VRU_9.mp4", "dada_ped_cross_07.mp4", "Pedestrian crossing an intersection is clipped by a turning vehicle in daylight", 8.5),
            ("VRU_videos/DADA_2000/VRU_10.mp4", "dada_ped_cross_08.mp4", "A crossing pedestrian is hit by a speeding vehicle unable to brake in time", 9.2),
            ("VRU_videos/DADA_2000/VRU_11.mp4", "dada_ped_cross_09.mp4", "A person stepping onto a crosswalk is struck by an overtaking vehicle", 8.6),
            ("VRU_videos/DADA_2000/VRU_12.mp4", "dada_ped_cross_10.mp4", "A child and adult attempting to cross a busy road face an imminent collision with traffic", 8.8),
        ],
        "bike_crash": [
            ("VRU_videos/DADA_2000/VRU_3.mp4", "dada_bike_crash_01.mp4", "A bicycle crossing an intersection is struck by a car in an urban roadway", 8.7),
            ("VRU_videos/DADA_2000/VRU_6.mp4", "dada_bike_crash_02.mp4", "A motorcycle collides violently with an oncoming vehicle on a multi-lane city road", 9.2),
            ("VRU_videos/DADA_2000/VRU_20.mp4", "dada_bike_crash_03.mp4", "A cyclist riding along an undivided road is hit from the side by an accelerating car", 8.6),
            ("VRU_videos/DADA_2000/VRU_37.mp4", "dada_bike_crash_04.mp4", "A motorcycle skids on a narrow road and crashes into an oncoming vehicle", 8.9),
            ("VRU_videos/DADA_2000/VRU_52.mp4", "dada_bike_crash_05.mp4", "A car reversing out of a driveway collides with a passing bicycle", 7.9),
            ("VRU_videos/DADA_2000/VRU_54.mp4", "dada_bike_crash_06.mp4", "A scooter rider turning at an intersection is struck by a straight-traveling car", 8.5),
            ("VRU_videos/DADA_2000/VRU_56.mp4", "dada_bike_crash_07.mp4", "A motorcycle crashes into the rear of a braking vehicle at an urban intersection", 8.4),
            ("VRU_videos/DADA_2000/VRU_58.mp4", "dada_bike_crash_08.mp4", "A bicycle rider swerves into a vehicle lane and is knocked down by traffic", 8.6),
            ("VRU_videos/DADA_2000/VRU_59.mp4", "dada_bike_crash_09.mp4", "A motorcycle collision at high speed with a car making an abrupt lane change", 8.8),
            ("VRU_videos/DADA_2000/VRU_60.mp4", "dada_bike_crash_10.mp4", "A cyclist struck by an opening car door and falling directly into moving traffic", 8.7),
        ],
        "truck_blind_spot": [
            ("VRU_videos/DoTA/VRU_7.mp4", "dada_truck_blind_01.mp4", "A large heavy commercial truck making a turn collides with a vehicle in its blind spot", 8.9),
            ("VRU_videos/DoTA/VRU_9.mp4", "dada_truck_blind_02.mp4", "A city bus turning at an intersection strikes a pedestrian standing in its right-side blind spot", 8.8),
            ("VRU_videos/DoTA/VRU_14.mp4", "dada_truck_blind_03.mp4", "A large delivery truck turns abruptly across oncoming lanes causing a severe crash", 9.0),
            ("VRU_videos/DoTA/VRU_15.mp4", "dada_truck_blind_04.mp4", "A heavy semi-truck blind spot collision pushing a car along a highway guardrail", 9.2),
            ("VRU_videos/DoTA/VRU_35.mp4", "dada_truck_blind_05.mp4", "A truck swerves across the median into oncoming traffic with severe impact hazard", 9.1),
            ("VRU_videos/DoTA/VRU_49.mp4", "dada_truck_blind_06.mp4", "A public transit bus sideswipes an overtaking vehicle during an unsignaled lane change", 8.3),
            ("VRU_videos/DoTA/VRU_52.mp4", "dada_truck_blind_07.mp4", "A pedestrian walking near the front blind spot of a stationary heavy truck is struck as it starts", 8.7),
            ("VRU_videos/DoTA/VRU_54.mp4", "dada_truck_blind_08.mp4", "A semi-trailer truck rear-ends traffic stopped at a nighttime intersection", 9.3),
            ("VRU_videos/DoTA/VRU_65.mp4", "dada_truck_blind_09.mp4", "A heavy dump truck turning right crushes a two-wheeler caught in its turning blind zone", 9.0),
            ("VRU_videos/DoTA/VRU_66.mp4", "dada_truck_blind_10.mp4", "A large logistics truck skids on wet asphalt causing a multi-vehicle jackknife collision", 9.1),
        ],
        "animal_collision": [
            ("VRU_videos/CAP_DATA/VRU_91.mp4", "dada_animal_col_01.mp4", "A dog suddenly dashes into the street directly in front of a moving car causing emergency braking", 7.8),
            ("VRU_videos/MANUAL_DATA/VRU_98.mp4", "dada_animal_col_02.mp4", "An animal running into a dark roadway at night causes a vehicle to swerve and crash", 8.2),
            ("VRU_videos/MANUAL_DATA/VRU_324.mp4", "dada_animal_col_03.mp4", "A deer leaping across a highway collides with the windshield of an oncoming car", 8.7),
            ("VRU_videos/CAP_DATA/VRU_91.mp4", "dada_animal_col_04.mp4", "A stray dog running across a multi-lane road narrowly escapes an oncoming vehicle collision", 7.5),
            ("VRU_videos/MANUAL_DATA/VRU_98.mp4", "dada_animal_col_05.mp4", "A high speed highway deer collision causing severe front bumper and windshield damage", 8.6),
            ("VRU_videos/MANUAL_DATA/VRU_324.mp4", "dada_animal_col_06.mp4", "A herd of cattle crossing an unlit rural road at night causing a multi-car pileup", 8.9),
            ("VRU_videos/CAP_DATA/VRU_91.mp4", "dada_animal_col_07.mp4", "A vehicle swerving abruptly to avoid hitting a dog in a suburban residential neighborhood", 7.6),
            ("VRU_videos/MANUAL_DATA/VRU_98.mp4", "dada_animal_col_08.mp4", "Wild hog dashing across a rural highway resulting in a violent frontal impact with a truck", 8.4),
            ("VRU_videos/MANUAL_DATA/VRU_324.mp4", "dada_animal_col_09.mp4", "Sudden moose crossing on a snowy road causing a loss of vehicle traction and barrier impact", 8.8),
            ("VRU_videos/CAP_DATA/VRU_91.mp4", "dada_animal_col_10.mp4", "A horse breaking loose onto a busy roadway posing an extreme collision hazard for drivers", 8.5),
        ]
    }

def run_dada_integration() -> None:
    """
    Main execution function: extracts 10 videos of each type, generates embeddings, and updates dataset metadata.
    """
    os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_FRAMES_DIR, exist_ok=True)

    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
    existing_ids = {m["video_id"] for m in metadata}

    print(f"Connecting to remote VRU / DADA-2000 benchmark repository...")
    stream = RemoteZipStream(VRU_ZIP_URL)
    with zipfile.ZipFile(stream) as zf:
        dataset_scenarios = get_dada2000_10_samples_per_type()
        
        for category, video_list in dataset_scenarios.items():
            print(f"\n=======================================================")
            print(f"Processing Category: {category.upper()} (10 videos)")
            print(f"=======================================================")
            
            for zip_path, local_name, summary, rating in video_list:
                dest_path = os.path.join(RAW_VIDEOS_DIR, local_name)
                
                # 1. Stream & extract video from zip
                if not os.path.exists(dest_path):
                    print(f"Extracting {zip_path} -> {local_name}...")
                    success = extract_single_video_from_remote_zip(stream, zf, zip_path, dest_path)
                    if not success:
                        continue
                else:
                    print(f"Already have {local_name}")

                # 2. Extract CLIP & BERT features
                if local_name not in existing_ids:
                    print(f"Extracting multimodal embeddings for {local_name}...")
                    record = process_and_cache_features(dest_path, local_name, summary, rating, f"DADA-2000 ({category})")
                    metadata.append(record)
                    existing_ids.add(local_name)
                    
                    # Update metadata.json incrementally
                    with open(METADATA_FILE, "w") as f:
                        json.dump(metadata, f, indent=2)

    print(f"\nIntegration complete! Total samples in unified dataset: {len(metadata)}")

if __name__ == "__main__":
    run_dada_integration()
