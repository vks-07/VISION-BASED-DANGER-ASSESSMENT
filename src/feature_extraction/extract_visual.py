import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
from typing import List, Tuple

def init_clip_model(model_name: str = "openai/clip-vit-base-patch32") -> Tuple[CLIPVisionModel, CLIPProcessor, str]:
    """
    Initializes the HuggingFace CLIP model and processor.
    
    Parameters:
        model_name (str): The specific CLIP model to load.
    Returns:
        tuple: (model, processor, device string)
    """
    model = CLIPVisionModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, processor, device

def load_images_from_folder(frames_dir: str) -> List[Image.Image]:
    """
    Loads all image files from a directory into memory.
    
    Parameters:
        frames_dir (str): Directory containing the extracted frame images.
    Returns:
        list: List of PIL Image objects.
    """
    image_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
    images = []
    
    for img_file in image_files:
        img_path = os.path.join(frames_dir, img_file)
        images.append(Image.open(img_path).convert("RGB"))
        
    return images

def process_images_through_clip(images: List[Image.Image], model: CLIPVisionModel, processor: CLIPProcessor, device: str) -> np.ndarray:
    """
    Passes a batch of images through CLIP to get their visual embeddings.
    
    Parameters:
        images (list): List of PIL Images.
        model (CLIPVisionModel): Loaded CLIP model.
        processor (CLIPProcessor): Loaded CLIP processor.
        device (str): Device to run the computation on.
    Returns:
        np.ndarray: The resulting visual embeddings.
    """
    # Processor handles all the resizing and normalization that CLIP expects
    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # pooler_output squashes the entire image into a single 1D vector (shape: 1x768).
        # Since we pass all 50 frames at once, the final shape we get back is exactly 50x768.
        embeddings = outputs.pooler_output.cpu().numpy()
        
    return embeddings

def extract_visual_features(frames_dir: str, output_path: str) -> None:
    """
    Main function coordinating the visual extraction pipeline.
    
    Parameters:
        frames_dir (str): Input directory of frames.
        output_path (str): Filepath to save the .npy embeddings.
    Returns:
        None
    """
    model, processor, device = init_clip_model()
    images = load_images_from_folder(frames_dir)
    
    if not images:
        print(f"No images found in {frames_dir}")
        return
        
    embeddings = process_images_through_clip(images, model, processor, device)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved visual embeddings to {output_path}")

if __name__ == "__main__":
    # Test block
    extract_visual_features("../../data/processed_frames/sample", "../../data/embeddings/sample_visual.npy")
