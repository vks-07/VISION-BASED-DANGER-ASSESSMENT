import cv2
import os
import numpy as np
from typing import Tuple


def get_video_info(video_path: str) -> Tuple[cv2.VideoCapture, float, int]:
    """
    Opens a video file and retrieves its properties.

    Parameters:
        video_path (str): Path to the input video file.
    Returns:
        tuple: (VideoCapture object, FPS, total_frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, total_frames


def calculate_frame_indices(
    start_time: float, end_time: float, fps: float, total_frames: int, num_frames: int
) -> np.ndarray:
    """
    Calculates the exact frame indices to extract from the temporal segment.

    Parameters:
        start_time (float): Ground truth start time in seconds.
        end_time (float): Ground truth end time in seconds.
        fps (float): Frames per second of the video.
        total_frames (int): Total number of frames in the video.
        num_frames (int): Number of frames to extract.
    Returns:
        np.ndarray: Array of integer frame indices.
    """
    start_frame = max(0, int(start_time * fps))
    end_frame = min(total_frames - 1, int(end_time * fps))

    if start_frame >= end_frame:
        raise ValueError("Start time must be before end time.")

    # We use linspace to get exactly 'num_frames' evenly spaced across the segment
    # e.g., if start frame is 10 and end is 110, and we want 50 frames, it steps by 2
    return np.linspace(start_frame, end_frame, num_frames, dtype=int)


def extract_and_save_frames(
    cap: cv2.VideoCapture, output_dir: str, frame_indices: np.ndarray
) -> int:
    """
    Reads specific frames from the video and saves them to disk.

    Parameters:
        cap (cv2.VideoCapture): Open video capture object.
        output_dir (str): Directory to save images.
        frame_indices (np.ndarray): Array of frame indices to extract.
    Returns:
        int: Number of frames successfully saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0

    for i, frame_idx in enumerate(frame_indices):
        # jump to the exact frame we calculated instead of reading one by one to save time
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            filename = os.path.join(output_dir, f"frame_{i+1:02d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

    return saved_count


def process_video_segment(
    video_path: str,
    output_dir: str,
    start_time: float,
    end_time: float,
    num_frames: int = 50,
) -> None:
    """
    Main function to extract a fixed number of frames from a video segment.

    Parameters:
        video_path (str): Path to the input video.
        output_dir (str): Directory for output frames.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        num_frames (int): Frames to extract (default 50).
    Returns:
        None
    """
    try:
        cap, fps, total_frames = get_video_info(video_path)
        indices = calculate_frame_indices(
            start_time, end_time, fps, total_frames, num_frames
        )
        saved = extract_and_save_frames(cap, output_dir, indices)
        print(f"Saved {saved}/{num_frames} frames to {output_dir}")
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
    finally:
        if "cap" in locals():
            cap.release()


if __name__ == "__main__":
    # Test block
    process_video_segment(
        video_path="../../data/raw_videos/sample.mp4",
        output_dir="../../data/processed_frames/sample",
        start_time=5.0,
        end_time=12.0,
    )
