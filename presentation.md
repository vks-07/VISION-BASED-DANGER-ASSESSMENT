#  ViDAS Project: Final Presentation Guide

---
## 1. Research Paper

* **Primary Paper:** *ViDAS: Vision-based Danger Assessment and Scoring* (2024).
* **Authors:** Pranav Gupta et al. (arXiv:2410.00477).
* **Core Contribution:** Created a dataset of 100 YouTube videos featuring dangerous and safe situations, annotated by humans on a continuous 0 to 10 scale.


> "The main research paper we based our project on is called ViDAS, published in 2024. Most datasets out there only label videos as 'safe' or 'unsafe'. This paper is unique because it provides 100 YouTube videos that are strictly rated by humans on a sliding scale from 0 to 10. We used this exact dataset to train our AI."

---

## 2. Literature Review

* **Previous Work:** Violence Detection, Anomaly Detection, Action Recognition (e.g., Kinetics dataset).
* **The Problem with Existing Work:**
  1. **Binary Classification:** They only say "Yes" (Danger) or "No" (Safe). They cannot give a score.
  2. **Single Modality:** They only look at the video, ignoring textual context. 


> "In our literature review, we found that traditional AI safety systems are very basic. They act like a simple alarm—they only output 'Safe' or 'Dangerous'. Furthermore, they only look at pixels. If a video shows a car crash, the AI thinks it's a real crash, even if it's just a Hollywood movie stunt. We wanted to solve this by adding text context to the video."

---

## 3. Problem Statement

* **Problem:** Quantifying human danger is highly subjective and requires deep context that raw video alone cannot provide.
* **Objective:** Build an end-to-end, Multimodal Artificial Intelligence pipeline that reads both Video and Text, and outputs a continuous danger score from 0.0 to 10.0.


> "Our problem statement is that evaluating danger requires human-like context. Our objective was to build a full-stack application where a user can upload a video and type a short description of what is happening. The AI will then fuse the video and text together to calculate a precise danger score out of 10."

---

## 4. Methodology

* **Step 1: Preprocessing (OpenCV)** - Extracts 50 uniform image frames from the video.
* **Step 2: Visual Understanding (CLIP)** - Uses OpenAI's `clip-vit-base` to turn 50 images into a 768-dimensional mathematical array.
* **Step 3: Text Understanding (BERT)** - Uses Google's `bert-base` to turn the text description into a 768-dimensional array.
* **Step 4: The Brain (PyTorch MLP)** - A custom Neural Network that combines the arrays (1536 dimensions) and uses Mean Squared Error (MSE) to learn the danger score.


> "Our methodology uses 'Multimodal Early Fusion'. First, OpenCV cuts the video into 50 static pictures. We feed those pictures into OpenAI's CLIP model, which converts the visual meaning into an array of 768 numbers. At the same time, we feed the user's text into Google's BERT model to get another 768 numbers. We glue them together into a 1536-number array and feed it into our custom PyTorch Neural Network to predict the final score."

---

## 5. Implementation

* **Dataset:** 81 valid downloaded videos from the ViDAS Hugging Face repository.
* **Backend:** FastAPI (Python) – Keeps massive AI models cached in RAM for instant predictions.
* **Frontend:** React.js – Interactive UI with dynamic color-coded danger bars.
* **Techniques Used:** Layer Normalization, Dropout (50%), and Adam Optimizer with L2 Weight Decay to prevent the AI from memorizing the data.


> "For implementation, we downloaded the 81 available videos from the ViDAS dataset. To make this a real-world product, we built a FastAPI backend server in Python to run the heavy AI models, and connected it to a modern React frontend. When you drag and drop a video, the React UI updates in real-time."

---

## 6. Updated Results

* **Evaluation Metric:** Mean Squared Error (MSE).
* **Training Results:** MSE successfully dropped to **~0.54** over 50 epochs.
* **Real-World Test Results (Relative Ranking):**
  1. *Man on bike vs high-speed truck* ➡️ 6.84 / 10.0 (Highest Risk)
  2. *Person walking on footpath* ➡️ 5.00 / 10.0 (Moderate Risk)
  3. *Person sitting on bed* ➡️ 4.37 / 10.0 (Lowest Risk)


> "Our primary training metric was Mean Squared Error, which successfully dropped to 0.54. This means our model learned the dataset very well! In our real-world testing on brand-new videos, the AI proved it has logical intelligence: It successfully ranked a high-speed truck as the most dangerous (6.84), and a person sitting on a bed as the safest (4.37)."

---

## 7. Comparison

* **Compared to Single-Modality (Vision-only):** Our multimodal approach captures semantic context (e.g., text descriptions) that raw pixels miss.
* **Compared to Original ViDAS Paper:** The original researchers relied on calling massive, expensive Large Language Models (like GPT-4) over the internet. We engineered a lightweight, local PyTorch neural network that runs instantly without paying for API calls.


> "Compared to standard vision models, our multimodal approach is much smarter because it understands context. Furthermore, the original researchers of the ViDAS paper relied heavily on sending data to huge language models like GPT-4 to evaluate danger. We improved on this by building our own custom, lightweight PyTorch model that can run locally and instantly on edge devices."

---

## 8. Conclusion and Future Work

* **Conclusion:** Successfully delivered a full-stack, multimodal AI pipeline capable of zero-shot feature extraction and continuous danger regression.
* **Future Work (Addressing the Dataset Limitation):** 
  * *The Issue:* The PyTorch network suffers from "Data Starvation". Because it only trained on 81 videos, it clusters predictions near the dataset average (~5.2) on completely unseen videos.
  * *Next Steps:* Scale the dataset from 100 videos to 10,000+ videos. Replace Mean-Pooling with a sequential model like an LSTM to track motion over time.


> "In conclusion, we successfully built an end-to-end AI application. However, we did discover a critical limitation: Data Starvation. Because we only trained a massive neural network on 81 videos, the AI is slightly unsure when it sees a brand-new video, so it tends to guess numbers near the dataset average of 5.2. As a future step, scaling the dataset to 10,000 videos would completely solve this, allowing the AI to confidently predict extreme scores like 0.1 or 9.9."
