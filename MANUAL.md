# 📖 Complete Setup and User Manual

This manual provides an exhaustive, end-to-end guide to setting up, configuring, training, and running the **Vision-Based Danger Assessment** project. It includes dedicated instructions for both **macOS** and **Windows** (Command Prompt & PowerShell), along with proxy configurations for restricted or university networks.

---

## 📑 Table of Contents
1. [System Prerequisites](#1-system-prerequisites)
2. [Cloning the Repository](#2-cloning-the-repository)
3. [Proxy Configuration Guide (For Restricted / College Networks)](#3-proxy-configuration-guide-for-restricted--college-networks)
4. [Python Virtual Environment Setup](#4-python-virtual-environment-setup)
5. [Installing Dependencies](#5-installing-dependencies)
   - [Backend Requirements (Python)](#backend-requirements-python)
   - [Frontend Requirements (Node.js)](#frontend-requirements-nodejs)
6. [Model Training & Dataset Pipeline](#6-model-training--dataset-pipeline)
   - [Option A: Fast Neural Network Training (Cached Embeddings)](#option-a-fast-neural-network-training-cached-embeddings)
   - [Option B: Full End-to-End Pipeline (ViDAS + UCF-Crime + DADA-2000)](#option-b-full-end-to-end-pipeline-vidas--ucf-crime--dada-2000)
   - [Evaluating Model Accuracy](#evaluating-model-accuracy)
7. [Running the Application (Backend & Frontend)](#7-running-the-application-backend--frontend)
   - [Terminal 1: Start the Backend Server](#terminal-1-start-the-backend-server)
   - [Terminal 2: Start the Frontend Client](#terminal-2-start-the-frontend-client)
8. [Using the Web Application](#8-using-the-web-application)
9. [Troubleshooting & Common Issues](#9-troubleshooting--common-issues)

---

## 1. System Prerequisites

Ensure the following runtimes and tools are installed on your machine:

| Tool | Recommended Version | Download / Installation Link |
| :--- | :---: | :--- |
| **Git** | `2.x+` | [git-scm.com](https://git-scm.com/) |
| **Python** | `3.10` – `3.12` | [python.org/downloads](https://www.python.org/downloads/) |
| **Node.js & npm** | Node `18.x` or `20.x` LTS | [nodejs.org](https://nodejs.org/) |
| **FFmpeg** | Latest stable | [ffmpeg.org](https://ffmpeg.org/download.html) |

### Installing Prerequisites by Operating System:

#### On macOS:
```bash
# Install Homebrew if not already installed (https://brew.sh)
brew install git python@3.11 node ffmpeg
```

#### On Windows:
1. **Python**: Download the Windows Installer from [python.org](https://www.python.org/downloads/).
   > **IMPORTANT:** During installation, check the box: **"Add Python to PATH"**.
2. **Node.js**: Download and run the **LTS installer** from [nodejs.org](https://nodejs.org/).
3. **Git**: Download and install [Git for Windows](https://git-scm.com/download/win).
4. **FFmpeg** (using Windows Package Manager `winget` in PowerShell):
   ```powershell
   winget install Gyan.FFmpeg
   ```
   *(Or download the zip from gyan.dev and add `bin` folder to your Windows System PATH).*

---

## 2. Cloning the Repository

Open your terminal or command prompt and clone the repository:

```bash
git clone https://github.com/vks-07/VISION-BASED-DANGER-ASSESSMENT.git
cd VISION-BASED-DANGER-ASSESSMENT
```

---

## 3. Proxy Configuration Guide (For Restricted / College Networks)

If you are working behind an institutional firewall, university Wi-Fi, or corporate proxy, network tools like `git`, `pip`, `npm`, and Hugging Face model downloaders might be blocked unless configured.

> **Note:** If your network has no proxy restrictions, you can skip this section and proceed directly to Section 4.

### 3.1 Terminal Environment Variables

Replace `http://proxy.yourcollege.edu:8080` with your actual proxy host and port (or `http://username:password@proxy.yourcollege.edu:8080` if authentication is required).

#### macOS / Linux (`zsh` or `bash`):
```bash
export HTTP_PROXY="http://proxy.yourcollege.edu:8080"
export HTTPS_PROXY="http://proxy.yourcollege.edu:8080"
export ALL_PROXY="http://proxy.yourcollege.edu:8080"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
```
*(To make these permanent on macOS, append them to `~/.zshrc` or `~/.bashrc`).*

#### Windows PowerShell:
```powershell
$env:HTTP_PROXY="http://proxy.yourcollege.edu:8080"
$env:HTTPS_PROXY="http://proxy.yourcollege.edu:8080"
$env:ALL_PROXY="http://proxy.yourcollege.edu:8080"
$env:NO_PROXY="localhost,127.0.0.1,0.0.0.0"
```

#### Windows Command Prompt (CMD):
```cmd
set HTTP_PROXY=http://proxy.yourcollege.edu:8080
set HTTPS_PROXY=http://proxy.yourcollege.edu:8080
set ALL_PROXY=http://proxy.yourcollege.edu:8080
set NO_PROXY=localhost,127.0.0.1,0.0.0.0
```

---

### 3.2 Tool-Specific Proxy Commands

#### Git Proxy:
```bash
# Set global proxy for Git
git config --global http.proxy http://proxy.yourcollege.edu:8080
git config --global https.proxy http://proxy.yourcollege.edu:8080

# To remove git proxy later:
git config --global --unset http.proxy
git config --global --unset https.proxy
```

#### Pip (Python Package Manager) Proxy:
```bash
# Pass proxy directly during install:
pip install --proxy http://proxy.yourcollege.edu:8080 -r requirements.txt

# Or set pip proxy globally:
pip config set global.proxy http://proxy.yourcollege.edu:8080

# To remove pip proxy later:
pip config unset global.proxy
```

#### Hugging Face / Transformers Model Download Mirror:
If your college network blocks direct connections to `huggingface.co`:
```bash
# macOS / Linux
export HF_ENDPOINT="https://hf-mirror.com"

# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

# Windows CMD
set HF_ENDPOINT=https://hf-mirror.com
```

#### NPM (Frontend) Proxy:
```bash
# Set npm proxy
npm config set proxy http://proxy.yourcollege.edu:8080
npm config set https-proxy http://proxy.yourcollege.edu:8080

# To remove npm proxy later:
npm config delete proxy
npm config delete https-proxy
```

---

## 4. Python Virtual Environment Setup

Always use a virtual environment so dependencies do not conflict with your global system Python.

### On macOS / Linux:
```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate
```
*(Your prompt will show `(.venv)` when active).*

---

### On Windows (PowerShell):
If you get an error saying `running scripts is disabled on this system`, enable script execution first:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Then create and activate the virtual environment:
```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
.venv\Scripts\Activate.ps1
```

---

### On Windows (Command Prompt - CMD):
```cmd
:: 1. Create virtual environment
python -m venv .venv

:: 2. Activate virtual environment
.venv\Scripts\activate.bat
```

---

## 5. Installing Dependencies

Make sure your virtual environment is activated before installing dependencies.

### Backend Requirements (Python)

Upgrade `pip` and install all necessary deep learning, vision, and API libraries:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **For Windows Users with an NVIDIA GPU (Optional):**
> If you have a dedicated NVIDIA GPU and wish to enable CUDA acceleration:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Otherwise, the default CPU PyTorch installed via `requirements.txt` runs smoothly across all platforms.

---

### Frontend Requirements (Node.js)

Open your terminal, navigate to the frontend directory, install npm packages, and return to the root folder:

```bash
cd src/frontend
npm install
cd ../..
```

---

## 6. Model Training & Dataset Pipeline

The unified dataset consists of **136 multimodal sequences** combining:
- **ViDAS Dataset (82 samples):** Dilapidated bridges, fire acrobatics, parkour, everyday activities.
- **UCF-Crime Dataset (9 samples):** Assault, abuse, arson, explosion, street violence.
- **Dashcam-VRU (5 samples):** Pedestrian near-misses, crosswalk braking, normal highway driving.
- **DADA-2000 Accident Benchmark (40 samples):** 10 Pedestrians Crossing, 10 Bike/Motorcycle Crashes, 10 Truck Blind Spots, 10 Animal Collisions.

All multimodal feature embeddings (CLIP visual vectors + BERT textual vectors) and pre-trained weights ([`data/models/danger_model.pth`](file:///Users/varun/Desktop/7th%20Sem%20project/data/models/danger_model.pth)) are already included in the repository.

### Option A: Fast Neural Network Training (Cached Embeddings)
To train the neural network architecture directly on the precomputed 136 samples (takes **~10–15 seconds**):

- **macOS / Linux:**
  ```bash
  python src/model/train.py
  ```
- **Windows (PowerShell / CMD):**
  ```powershell
  python src\model\train.py
  ```

---

### Option B: Full End-to-End Pipeline (ViDAS + UCF-Crime + DADA-2000)
To run the entire pipeline from scratch (verifies video downloads, extracts CLIP & BERT embeddings, and trains the model for 50 epochs):

```bash
python train_full_dataset.py
```

---

### Evaluating Model Accuracy
To compute the Mean Absolute Error (MAE), Pearson correlation ($r$), and Spearman rank correlation ($\rho$) across all 136 dataset samples:

```bash
python -c "
import json, os, numpy as np, torch
from scipy.stats import pearsonr, spearmanr
from src.model.mlp_network import DangerScoreNetwork

with open('data/dataset_metadata.json') as f:
    items = json.load(f)

model = DangerScoreNetwork()
model.load_state_dict(torch.load('data/models/danger_model.pth', map_location='cpu'))
model.eval()

preds, actuals = [], []
with torch.no_grad():
    for item in items:
        v_p = f'data/embeddings/{item[\"video_id\"]}_visual.npy'
        t_p = f'data/embeddings/{item[\"video_id\"]}_text.npy'
        if os.path.exists(v_p) and os.path.exists(t_p):
            v = torch.from_numpy(np.load(v_p)).float()
            t = torch.from_numpy(np.load(t_p)).float()
            preds.append(model(v, t).item())
            actuals.append(float(item['rating']))

mae = np.mean(np.abs(np.array(preds) - np.array(actuals)))
r, _ = pearsonr(preds, actuals)
rho, _ = spearmanr(preds, actuals)
print(f'Samples Evaluated: {len(preds)}')
print(f'Overall MAE: {mae:.4f} / 10')
print(f'Pearson r: {r:.4f}')
print(f'Spearman rho: {rho:.4f}')
"
```

**Expected Performance:**
- **MAE:** `~0.178 / 10.0` (~1.78% error)
- **Pearson $r$:** `~0.976` (Linear score calibration)
- **Spearman $\rho$:** `~0.982` (Monotonic hazard ranking)

---

## 7. Running the Application (Backend & Frontend)

To run the full stack, keep two terminal windows open:

### Terminal 1: Start the Backend Server

Ensure your virtual environment is activated, then run:

#### macOS / Linux:
```bash
source .venv/bin/activate
python src/backend/main.py
```

#### Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
python src\backend\main.py
```

#### Windows (CMD):
```cmd
.venv\Scripts\activate.bat
python src\backend\main.py
```

**Output:**
```text
Loading AI Models into memory...
All models loaded and ready to predict!
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
- The backend API is active at: `http://localhost:8000`
- Interactive OpenAPI / Swagger docs: `http://localhost:8000/docs`

---

### Terminal 2: Start the Frontend Client

Open a second terminal window, navigate to the frontend folder, and launch the Vite development server:

#### macOS / Linux / Windows:
```bash
cd src/frontend
npm run dev
```

**Output:**
```text
  VITE v7.x.x  ready in 180 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```
- Open your browser and go to: **`http://localhost:5173`**

---

## 8. Using the Web Application

1. **Upload Video:**
   - Click the video upload area or drag and drop a video file (`.mp4`, `.mov`, `.avi`).
2. **Provide Context Description:**
   - Enter descriptive context in the text prompt (e.g., *"A child abruptly darts into the road in front of approaching vehicle"* or *"Walking down a quiet sidewalk on a sunny morning"*).
3. **Run Assessment:**
   - Click **"Assess Danger"**.
4. **Interpret Results:**
   - **Continuous Danger Gauge:** Evaluates hazard severity from **0.0** to **10.0**.
   - **Risk Classification:**
     - **High Danger (Score $\ge 6.5$):** Immediate collision hazard, physical violence, high risk.
     - **Moderate Risk (Score $3.5 - 6.4$):** Precautionary situation, traffic congestion, potential obstacle.
     - **Low Risk (Score $< 3.5$):** Safe everyday activity, clear roadway, controlled movement.

---

## 9. Troubleshooting & Common Issues

### Issue 1: PowerShell Script Execution Restricted on Windows
- **Error:** `.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`
- **Solution:** Open PowerShell as Administrator or run:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```

---

### Issue 2: Port 8000 or 5173 Already in Use
If another process is using port 8000 or 5173:

- **macOS / Linux:**
  ```bash
  # Find and kill process on port 8000
  lsof -ti:8000 | xargs kill -9
  # Find and kill process on port 5173
  lsof -ti:5173 | xargs kill -9
  ```

- **Windows (Command Prompt / PowerShell):**
  ```cmd
  # Find process ID using port 8000
  netstat -ano | findstr :8000
  # Kill the process using its PID (e.g., PID 1234)
  taskkill /PID 1234 /F
  ```

---

### Issue 3: SSL / Certificate Verification Errors Behind College Proxy
- **Error:** `urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>` or `SSLError`.
- **Solution:** Configure Python or pip to trust certificates:
  ```bash
  pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
  ```

---

### Issue 4: Hugging Face Model Download Timeout
- **Error:** `ReadTimeoutError` or `ConnectTimeoutError` when loading CLIP (`openai/clip-vit-base-patch32`) or BERT (`bert-base-uncased`).
- **Solution:** Set the official Hugging Face mirror endpoint:
  ```bash
  # macOS
  export HF_ENDPOINT=https://hf-mirror.com

  # Windows PowerShell
  $env:HF_ENDPOINT="https://hf-mirror.com"
  ```

---

### Issue 5: OpenCV Video Reading Error (`cv2.VideoCapture` fails)
- **Error:** Unable to decode `.mp4` video frames.
- **Solution:** Ensure `ffmpeg` is installed and accessible in your system PATH:
  ```bash
  ffmpeg -version
  ```
  If not found, install via `brew install ffmpeg` (macOS) or `winget install Gyan.FFmpeg` (Windows).
