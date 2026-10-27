# 🚀 Setup and Developer Guide

This guide walks you through setting up, running, training, and contributing to the **ViDAS Vision-Based Danger Assessment** project from scratch.

---

## 📋 Table of Contents
1. [Prerequisites](#-prerequisites)
2. [Project Architecture Overview](#-project-architecture-overview)
3. [Step-by-Step Installation](#-step-by-step-installation)
   - [1. Clone the Repository](#1-clone-the-repository)
   - [2. Set Up Python Virtual Environment](#2-set-up-python-virtual-environment)
   - [3. Install Backend Dependencies](#3-install-backend-dependencies)
   - [4. Set Up Frontend Dependencies](#4-set-up-frontend-dependencies)
4. [Running the Application (Quickstart)](#-running-the-application-quickstart)
5. [Training the Model from Scratch](#-training-the-model-from-scratch)
6. [Testing the Pipeline](#-testing-the-pipeline)
7. [Repository File Structure](#-repository-file-structure)
8. [Troubleshooting & Common Questions](#-troubleshooting--common-questions)

---

## 🛠 Prerequisites

Ensure the following tools are installed on your machine:
* **Git**: [Download Git](https://git-scm.com/)
* **Python**: Version `3.9`, `3.10`, or `3.11` recommended ([Python Downloads](https://www.python.org/downloads/))
* **Node.js & npm**: Node version `18.x` or later ([Node.js Downloads](https://nodejs.org/))
* **Operating System**: macOS, Linux, or Windows (PowerShell/WSL)

---

## 🧠 Project Architecture Overview

The pipeline computes a continuous danger score from **0.0 (Safe)** to **10.0 (High Danger)** using multimodal deep learning:
1. **Video Preprocessing (`OpenCV`)**: Samples 50 uniform frames across the video segment.
2. **Visual Feature Extraction (`CLIP`)**: Uses `openai/clip-vit-base-patch32` to produce 768-dimensional visual vectors across the 50 frames.
3. **Text Feature Extraction (`BERT`)**: Uses `bert-base-uncased` to encode contextual text descriptions into a 768-dimensional vector.
4. **Temporal Modeling & Cross-Attention (`PyTorch`)**:
   - **Bi-LSTM**: Captures bidirectional temporal motion dynamics across the 50 frames.
   - **Cross-Attention**: Text queries the visual frame sequence to locate exact hazardous moments.
   - **Huber + Ranking Loss**: Trained with Smooth L1 and pairwise ranking loss, plus temporal frame masking and sensor noise augmentation to prevent regression to the mean.
5. **Full-Stack Interface**: A **FastAPI** backend caching the foundation models in RAM and a **React + Vite** frontend.

---

## 💻 Step-by-Step Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vks-07/VISION-BASED-DANGER-ASSESSMENT.git
cd VISION-BASED-DANGER-ASSESSMENT
```

### 2. Set Up Python Virtual Environment
Creating an isolated virtual environment prevents dependency collisions.

* **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* **On Windows (Command Prompt / PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```

> Always verify the virtual environment is active before running Python scripts (you should see `(.venv)` in your terminal prompt).

### 3. Install Backend Dependencies
With your virtual environment activated, install all required Python libraries:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Frontend Dependencies
Open a second terminal window and install the React dependencies:
```bash
cd src/frontend
npm install
```

---

## 🎮 Running the Application (Quickstart)

The pre-trained weights (`data/models/danger_model.pth`) are already included in the repository, so you can run the app immediately without retraining.

### Step A: Start the AI Backend Server
In your first terminal (from the project root, with `.venv` activated):
```bash
python src/backend/main.py
```
* The server will load CLIP, BERT, and the PyTorch model into memory.
* Once you see: `All models loaded and ready to predict!`, the API is live at `http://localhost:8000`.
* Interactive API documentation is available at `http://localhost:8000/docs`.

### Step B: Start the React Frontend
In your second terminal (inside `src/frontend`):
```bash
npm run dev
```
* Open the local Vite URL displayed in your terminal (usually `http://localhost:5173`).
* **Usage**:
  1. Upload or drag-and-drop any `.mp4` video.
  2. Type a short description of what is occurring in the video.
  3. Click **Predict Danger Score**.
  4. View the real-time score out of 10.0 with visual color-coded risk indicators (Green: Low Risk, Orange: Moderate Risk, Red: High Danger).

---

## 🏋️ Training the Model from Scratch

If you want to re-download the dataset, extract fresh embeddings, and retrain the PyTorch MLP:

```bash
# Ensure .venv is activated from project root
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python train_full_dataset.py
```

### What this automated script does:
1. **Downloads Videos**: Uses `yt-dlp` to download videos from the Hugging Face `pranked03/ViDAS` dataset into `data/raw_videos/`.
2. **Extracts Frames**: Uses OpenCV to sample 50 uniform frames per video into `data/processed_frames/`.
3. **Generates Embeddings**: Passes frames through CLIP and summaries through BERT, saving vectors as `.npy` files in `data/embeddings/`.
4. **Trains Neural Network**: Runs 50 epochs using Adam optimizer and MSE loss, saving the final weights to `data/models/danger_model.pth`.

---

## 🧪 Testing the Pipeline

To test end-to-end processing on a single sample without running the full training script:
```bash
python test_pipeline.py
```
This downloads a single sample, extracts features, trains for 10 epochs, and confirms that MSE loss converges properly.

---

## 📂 Repository File Structure

```text
VISION-BASED-DANGER-ASSESSMENT/
│
├── data/
│   ├── models/                 # Saved PyTorch model weights (danger_model.pth)
│   ├── raw_videos/             # Downloaded MP4 videos
│   ├── processed_frames/       # Extracted 50-frame video samples
│   └── embeddings/             # Extracted CLIP & BERT numpy vectors (.npy)
│
├── src/
│   ├── backend/
│   │   └── main.py             # FastAPI backend with cached model inference
│   ├── data_pipeline/
│   │   ├── download_videos.py  # yt-dlp downloader for ViDAS YouTube links
│   │   └── preprocess.py       # OpenCV 50-frame uniform sampling
│   ├── feature_extraction/
│   │   ├── extract_visual.py   # CLIP visual feature extraction (768-dim)
│   │   └── extract_text.py     # BERT text feature extraction (768-dim)
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── App.jsx         # React UI for upload and score visualization
│   │   │   ├── App.css         # Styling and progress bar animations
│   │   │   └── main.jsx        # React root entrypoint
│   │   ├── package.json        # Node frontend dependencies
│   │   └── vite.config.js      # Vite build configuration
│   └── model/
│       ├── mlp_network.py      # PyTorch DangerScoreNetwork (LayerNorm, Dropout, 0-10 Sigmoid)
│       ├── dataset.py          # PyTorch DangerDataset with temporal mean pooling
│       └── train.py            # PyTorch MSE training loop with L2 weight decay
│
├── requirements.txt            # Python dependencies
├── train_full_dataset.py       # Master script for dataset ingestion & training
├── test_pipeline.py            # Quick end-to-end single-sample test script
├── presentation.md             # Project presentation script & slide guide
├── report.md                   # Detailed technical project report
├── README.md                   # Project overview & quick summary
└── SETUP.md                    # This complete developer setup guide
```

---

## ❓ Troubleshooting & Common Questions

### 1. `Port 8000 already in use`
If port 8000 is occupied, you can kill the existing process or run Uvicorn on another port:
```bash
# On macOS / Linux to free port 8000:
lsof -ti:8000 | xargs kill -9

# Or run FastAPI on a different port:
uvicorn src.backend.main:app --port 8001
```
*(If you change the port, update the fetch URL in `src/frontend/src/App.jsx` accordingly).*

### 2. `ModuleNotFoundError: No module named 'src'`
Run scripts from the repository root, or ensure `PYTHONPATH` includes the root directory:
```bash
export PYTHONPATH=$PYTHONPATH:.
python -m src.model.train
```

### 3. GPU vs CPU Support
The code is designed to automatically detect CUDA or MPS (Apple Silicon), with a reliable CPU fallback. You do not need a dedicated GPU to run inference; inference on a single video takes approximately 2–4 seconds on CPU.

### 4. `Warning: You are sending unauthenticated requests to the HF Hub`
This warning from Hugging Face is normal when downloading public datasets without an API token. It does not affect functionality. If you hit rate limits, you can optionally set `export HF_TOKEN="your_token"`.

### 5. Why are scores clustered around 4.5–6.0 on unseen videos?
The dataset contains human ratings with a natural mean around `~5.2`. Because the model was trained on the available 81 ViDAS samples, it encounters the *Curse of Dimensionality* on novel out-of-distribution videos, resulting in statistical regression toward the mean. Refer to `presentation.md` for the complete technical defense and analysis.

---

## 🤝 Contributing
1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make and test your changes.
3. Commit with clean, descriptive commit messages.
4. Push to your branch and open a Pull Request.
