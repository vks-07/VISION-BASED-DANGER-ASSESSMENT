import os
import yt_dlp
from datasets import load_dataset
from typing import Dict, Any, List

def fetch_dataset_urls(dataset_name: str = "pranked03/ViDAS", split: str = "train") -> List[Dict[str, Any]]:
    """
    Loads the dataset and extracts the video metadata required for downloading.
    
    Parameters:
        dataset_name (str): Hugging Face dataset identifier.
        split (str): Which split to load (e.g., 'train').
    Returns:
        list: A list of dictionaries containing 'video_id' and 'url'.
    """
    dataset = load_dataset(dataset_name, split=split)
    return [{"video_id": row["video_id"], "url": row["url"]} for row in dataset]

def download_youtube_video(url: str, output_path: str) -> bool:
    """
    Downloads a single video from YouTube using yt-dlp.
    
    Parameters:
        url (str): The YouTube URL.
        output_path (str): The exact file path to save the .mp4 file.
    Returns:
        bool: True if successful, False if download failed.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/best[ext=mp4]',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True 
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([url])
            return error_code == 0
    except Exception as e:
        print(f"Critical failure downloading {url}: {e}")
        return False

def download_all_videos(output_dir: str, limit: int = 5) -> None:
    """
    Main pipeline function to fetch dataset URLs and download the MP4s.
    
    Parameters:
        output_dir (str): Directory to save the raw videos.
        limit (int): Maximum number of videos to download. Set to 0 for all.
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    video_records = fetch_dataset_urls()
    
    if limit > 0:
        video_records = video_records[:limit]
        
    print(f"Starting download of {len(video_records)} videos to '{output_dir}'...")
    
    success_count = 0
    for record in video_records:
        vid_id = record['video_id']
        url = record['url']
        out_path = os.path.join(output_dir, vid_id)
        
        if os.path.exists(out_path):
            print(f"Already exists: {vid_id}")
            success_count += 1
            continue
            
        print(f"Downloading {vid_id} from {url} ...")
        if download_youtube_video(url, out_path):
            success_count += 1
            
    print(f"Finished! Downloaded {success_count}/{len(video_records)} videos.")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_vids_dir = os.path.join(base_dir, "data", "raw_videos")
    download_all_videos(raw_vids_dir, limit=5)
