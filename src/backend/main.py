import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(base_dir)

from src.data_pipeline.preprocess import process_video_segment, get_video_info
from src.feature_extraction.extract_visual import init_clip_model, load_images_from_folder, process_images_through_clip
from src.feature_extraction.extract_text import init_bert_model, process_text_through_bert
from src.model.mlp_network import DangerScoreNetwork

app = FastAPI(title="Danger Score Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading AI Models into memory...")
clip_model, clip_processor, clip_device = init_clip_model()
bert_model, bert_tokenizer, bert_device = init_bert_model()

danger_network = DangerScoreNetwork()
model_weights_path = os.path.join(base_dir, "data", "models", "danger_model.pth")

if os.path.exists(model_weights_path):
    danger_network.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
danger_network.eval()
print("All models loaded and ready to predict!")

@app.post("/predict")
async def predict_danger_score(video: UploadFile, description: str = Form(...)):
    """
    Endpoint that accepts a video file and text description, runs the full AI pipeline, 
    and returns a continuous danger score from 0.0 to 10.0.
    """
    temp_video_path = os.path.join(base_dir, "data", "temp_upload.mp4")
    frames_dir = os.path.join(base_dir, "data", "temp_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    try:
        with open(temp_video_path, "wb") as f:
            f.write(await video.read())
            
        cap, fps, total_frames = get_video_info(temp_video_path)
        cap.release()
        duration = total_frames / fps if fps > 0 else 0
        
        process_video_segment(temp_video_path, frames_dir, start_time=0.0, end_time=duration, num_frames=50)
        
        images = load_images_from_folder(frames_dir)
        visual_emb = process_images_through_clip(images, clip_model, clip_processor, clip_device)
        
        # Temporal pooling
        visual_emb = np.mean(visual_emb, axis=0, keepdims=True)
        visual_tensor = torch.tensor(visual_emb, dtype=torch.float32)
        
        text_emb = process_text_through_bert(description, bert_model, bert_tokenizer, bert_device)
        text_tensor = torch.tensor(text_emb, dtype=torch.float32)
        
        with torch.no_grad():
            score_tensor = danger_network(visual_tensor, text_tensor)
            score = score_tensor.item()
            
        return {
            "status": "success",
            "danger_score": round(score, 2),
            "description": description
        }
        
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
