# 🚨 ViDAS: Multimodal Danger Score Predictor

ViDAS is an end-to-end Artificial Intelligence pipeline designed to act as an automated safety inspector. It takes a raw video segment and a contextual text description, fuses them using early-fusion multimodal deep learning, and outputs a continuous danger score from **0.0 (Completely Safe)** to **10.0 (Life-Threatening)**.

---

## 🧠 Architecture & Pipeline

The system uses a 4-stage pipeline:

1. **Preprocessing (OpenCV):** 
   Dynamically calculates the video duration and extracts exactly 50 uniform frames, allowing for programmatic manipulation without fully decoding the video stream.
2. **Visual Feature Extraction (CLIP):** 
   Passes frames through OpenAI's `clip-vit-base-patch32` to produce a 50-frame temporal sequence `(50, 768)` capturing visual attributes.
3. **Text Feature Extraction (BERT):** 
   Passes the user's text description through `bert-base-uncased` to extract a `1x768` vector, grounding the visual data with bidirectional semantic context.
4. **Temporal Modeling & Cross-Attention (PyTorch Engine):** 
   * **Bi-LSTM Sequence Modeling:** Reads the 50-frame sequence forward and backward to capture motion dynamics, velocity, and acceleration.
   * **Multimodal Cross-Attention:** Text queries the visual frame sequence to dynamically attend to critical hazard frames.
   * **Robust Training:** Trained with **Huber (Smooth L1) Loss** and **Pairwise Ranking Loss** to prevent regression-to-the-mean, combined with temporal frame masking and sensor noise data augmentation. Output bounded via `Sigmoid * 10.0`.

---

## 🛠️ Tech Stack

* **Computer Vision:** `OpenCV`
* **Deep Learning Framework:** `PyTorch`
* **Foundation Models:** `Hugging Face Transformers` (CLIP, BERT)
* **Backend Web Server:** `FastAPI` + `Uvicorn` (for async uploads and RAM-cached model serving)
* **Frontend UI:** `React.js` + `Vite`

---

## 📂 Project Structure

```text
📦 project_root
 ┣ 📂 src
 ┃ ┣ 📂 backend              # FastAPI server (main.py)
 ┃ ┣ 📂 data_pipeline        # OpenCV extraction & yt-dlp scraping scripts
 ┃ ┣ 📂 feature_extraction   # CLIP and BERT embedding logic
 ┃ ┣ 📂 frontend             # React.js UI
 ┃ ┗ 📂 model                # PyTorch MLP Network and Training Loop
 ┣ 📂 data                   # Local storage for mp4s, frames, embeddings, and .pth weights
 ┣ 📜 train_full_dataset.py  # Master script to download data and train from scratch
 ┗ 📜 requirements.txt       # Python dependencies
```

---

> 💡 **Documentation Links:**
> - [MANUAL.md](MANUAL.md): Complete step-by-step setup, proxy guide, and running instructions for both **macOS** and **Windows** (PowerShell & CMD).
> - [report.md](report.md): Publication-grade academic research paper and technical engineering report (What, Why, How, Why Not Alternatives, testing & loss).
> - [SETUP.md](SETUP.md): Developer architecture walkthrough and troubleshooting.
> - [presentation.md](presentation.md): Project presentation slide deck and defense script.

**1. Clone and Setup Python Environment**
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install AI dependencies
pip install -r requirements.txt
```

**2. Setup React Frontend**
```bash
cd src/frontend
npm install
```

---

## 🎮 How to Run the Application

You need two terminal windows to run the full application.

**Terminal 1: Start the AI Backend**
*(Ensure your virtual environment is activated)*
```bash
python src/backend/main.py
```
> Note: The backend will take a few seconds to load the massive CLIP and BERT models into your RAM. Wait until it prints *"All models loaded and ready to predict!"*

**Terminal 2: Start the React UI**
```bash
cd src/frontend
npm run dev
```
Open the provided `localhost` link in your browser to access the drag-and-drop web interface!

---

## 🏋️ Training the Model from Scratch

If you want to re-train the model weights (`danger_model.pth`) yourself:
```bash
source .venv/bin/activate
python train_full_dataset.py
```
This master script will automatically:
1. Download the `pranked03/ViDAS` dataset videos from YouTube.
2. Extract frames via OpenCV.
3. Generate and save `.npy` CLIP and BERT embeddings.
4. Run the 50-epoch PyTorch training loop and save the new weights.

### Expanding with UCF-Crime & Dashcam Datasets
To expand the training data with real-world surveillance anomalies (Assault, Fight, Arson, Explosion) and Dashcam traffic/pedestrian collision hazards:
```bash
python src/data_pipeline/expand_dataset.py
python -m src.model.train
```

### Integrating DADA-2000 Benchmark (Pedestrian, Bike, Truck, Animal)
To integrate 10 videos of each scenario type (Pedestrian Crossing, Bike Crash, Truck Blind Spot, Animal Collision) directly from the DADA-2000 benchmark:
```bash
python src/data_pipeline/integrate_dada2000.py
python -m src.model.train
```
This expands the training dataset to **136 multimodal samples** across ViDAS, UCF-Crime, and DADA-2000.
