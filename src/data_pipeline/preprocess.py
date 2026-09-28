import cv2
import os
import numpy as np

def extract_frames(video_path, output_dir, start_time, end_time, num_frames=50):
    """
    Extracts a fixed number of equally spaced frames from a specific segment of a video.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        start_time (float): Ground truth start time of the event in seconds.
        end_time (float): Ground truth end time of the event in seconds.
        num_frames (int): Number of frames to extract (default is 50).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get the frames per second (fps) and total frames of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        print("Error: FPS is 0. Video might be corrupted or format unsupported.")
        return

    # Convert timestamps to frame indices
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Ensure indices are within the video's actual boundaries
    start_frame = max(0, start_frame)
    end_frame = min(total_frames_in_video - 1, end_frame)
    
    if start_frame >= end_frame:
        print(f"Error: Invalid temporal segment for video {video_path}")
        cap.release()
        return
        
    # Generate exactly 'num_frames' equally separated indices between start and end
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    
    print(f"Processing '{video_path}'...")
    print(f"Segment: {start_time}s to {end_time}s | Extracting {num_frames} frames...")
    
    saved_count = 0
    for i, frame_idx in enumerate(frame_indices):
        # Jump directly to the specific frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Save frame as a JPEG image (e.g., frame_01.jpg, frame_02.jpg)
            filename = os.path.join(output_dir, f"frame_{i+1:02d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        else:
            print(f"Warning: Could not read frame at index {frame_idx}")
            
    cap.release()
    print(f"Successfully saved {saved_count} frames to '{output_dir}'\n")

if __name__ == "__main__":
    # --- Example Usage ---
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Simulating a video in the raw_videos directory
    sample_video_path = os.path.join(base_dir, "data", "raw_videos", "sample_video.mp4")
    output_folder = os.path.join(base_dir, "data", "processed_frames", "sample_video")
    
    # Example ground truth annotation (e.g., dangerous event from 5.5s to 12.0s)
    ground_truth_start = 5.5
    ground_truth_end = 12.0
    
    if os.path.exists(sample_video_path):
        extract_frames(
            video_path=sample_video_path,
            output_dir=output_folder,
            start_time=ground_truth_start,
            end_time=ground_truth_end,
            num_frames=50
        )
    else:
        print(f"To test this script, place a video named 'sample_video.mp4' in 'data/raw_videos/'")
        print(f"Path looked for: {sample_video_path}")
