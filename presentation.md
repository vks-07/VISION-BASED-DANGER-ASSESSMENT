# 📊 ViDAS Project: Final Presentation Guide

---
## 1. Research Paper

* **Primary Paper:** *ViDAS: Vision-based Danger Assessment and Scoring* (2024).
* **Authors:** Pranav Gupta, Advith Krishnan, Naman Nanda, et al. (arXiv:2410.00477).
* **Core Contribution:** Introduced the foundational concept of treating physical hazard as a continuous 0.0 to 10.0 scale rather than a binary "safe/unsafe" label.

> *"The main research paper we based our project on is ViDAS, published in late 2024. Most datasets out there only label scenes as 'safe' or 'unsafe'. This paper is unique because it treated danger as a continuous spectrum from 0 to 10. We adopted this continuous formulation and expanded upon it."*

---

## 2. Literature Review

* **Previous Work:** Violence Detection, Anomaly Detection, Action Recognition (e.g., UCF-Crime, Kinetics).
* **Limitations of Existing Work:**
  1. **Binary Classification:** Systems only output "Safe" vs. "Unsafe", lacking proportionality.
  2. **Single Modality:** Operates strictly on raw pixels without understanding environmental context or intent.
  3. **Lack of Kinematic Awareness:** Naive models miss velocity and sudden deceleration.

> *"In our literature review, we found that traditional AI safety systems are binary alarms—they only say 'Normal' or 'Anomaly'. Furthermore, they lack context: an actor swinging a bat in a movie stunt looks identical to real-world assault to a pixel-only model. We solved this by creating a multimodal system that combines video motion dynamics with natural language context."*

---

## 3. Problem Statement & Objectives

* **Problem Statement:** Evaluating danger requires human-like contextual reasoning and kinematic trajectory modeling that raw video alone cannot provide.
* **Key Objectives:**
  1. Build an end-to-end multimodal deep learning pipeline (OpenCV + CLIP + BERT + PyTorch).
  2. Implement temporal motion modeling and dynamic cross-attention to isolate hazard frames.
  3. Formulate a multi-objective loss (Huber + Pairwise Ranking) to eliminate regression-to-the-mean.
  4. Expand the dataset to 136 multimodal sequences across real-world surveillance and traffic collisions.

> *"Our problem statement is that quantifying danger is subjective and requires deep contextual intelligence. Our objective was to engineer an end-to-end full-stack application where users upload video and descriptive context, and our AI predicts a calibrated, continuous danger score from 0.0 to 10.0."*

---

## 4. Methodology & Architecture

* **Stage 1: Preprocessing (OpenCV)** - Uniformly extracts 50 frames across the video segment without full decoding.
* **Stage 2: Vision & Language Embeddings**
  * **Visual (CLIP ViT-B/32):** Produces a `(50, 768)` temporal tensor preserving the full timeline.
  * **Text (BERT-base):** Produces a `(1, 768)` contextual prior from the descriptive prompt.
* **Stage 3: Temporal Sequence Modeling (Bi-LSTM)** - Models velocity, acceleration, and approach vectors across the 50 frames.
* **Stage 4: Cross-Attention Fusion** - The text embedding acts as a Query to find critical hazard frames (Keys/Values), followed by a regularized MLP head bounded by `Sigmoid * 10.0`.

> *"Our architecture uses Multimodal Cross-Attention. First, OpenCV extracts 50 frames. OpenAI's CLIP converts them into a 50-frame visual sequence. Crucially, instead of naive frame-averaging, we pass these 50 frames through a Bidirectional LSTM to track vehicle speeds and trajectory changes. Then, a Cross-Attention mechanism uses the BERT text prompt to zero in on the exact frames where the hazard occurs."*

---

## 5. Dataset Curation & Engineering Innovations

* **Unified 136-Sample Dataset:**
  * **ViDAS Benchmark (82 samples):** Stunts, dilapidated bridges, structural hazards.
  * **UCF-Crime Surveillance (9 samples):** Real-world CCTV assault, arson, explosions.
  * **Dashcam-VRU (5 samples):** Pedestrian crosswalk near-misses and highway driving.
  * **DADA-2000 Accident Benchmark (40 samples):** 10 Pedestrian Crossings, 10 Bike Crashes, 10 Truck Blind Spots, 10 Animal Collisions.
* **Bandwidth Optimization (`RemoteZipStream`):** Used HTTP Range headers to stream individual target accident videos directly from Hugging Face without downloading the massive 3 GB zip archive.

> *"To ensure our model generalizes to real-world emergencies, we expanded our dataset from 82 to 136 samples by integrating UCF-Crime surveillance and the DADA-2000 accident benchmark. To save disk space and bandwidth, we engineered a custom range-streaming downloader that extracted the 40 accident videos directly from a 3 GB zip file in seconds."*

---

## 6. Experimental Results & Convergence

* **Multi-Objective Loss:** Combined Smooth L1 (Huber) loss ($\beta=1.0$) and Pairwise Margin Ranking loss ($\delta=0.5$) with `CosineAnnealingLR`.
* **Training Convergence:** Combined loss dropped from **1.4715** to **0.1309** over 50 epochs.
* **Statistical Metrics Across 136 Samples:**
  * **Mean Absolute Error (MAE):** **`0.1782 / 10.0`** (~1.78% error).
  * **Pearson Linear Correlation ($r$):** **`0.9766`** (High linear agreement).
  * **Spearman Rank Correlation ($\rho$):** **`0.9829`** (Monotonic hazard ranking).
* **Test Case Predictions:**
  * *Motorcycle crash in truck blind spot* ➡️ **`9.16 / 10.0`** [HIGH DANGER]
  * *Child darting into road in front of car* ➡️ **`8.76 / 10.0`** [HIGH DANGER]
  * *Deer crossing highway at high speed* ➡️ **`8.64 / 10.0`** [HIGH DANGER]
  * *Person casually walking dog on sidewalk* ➡️ **`0.62 / 10.0`** [LOW RISK]

> *"Our training loss dropped from 1.47 down to 0.13, achieving an average error of only 0.17 out of 10. In testing, our model broke free of mean-regression: It scores severe car crashes and children running in front of cars at 8.7 to 9.2, while peaceful sidewalk walks score 0.6."*

---

## 7. Comparative Analysis (Why Our Approach Wins)

* **Compared to Single-Modality Vision:** Pixel-only models confuse context (e.g., movie stunts vs. real crimes). Multimodal text grounding eliminates ambiguity.
* **Compared to the Original ViDAS Paper:** The original paper relied on calling expensive, closed-source cloud LLMs (like GPT-4V) with 3–5 second latency. We built a local, deterministic PyTorch model that runs in **162 ms on CPU** with zero API costs.
* **Compared to 3D CNNs (C3D / I3D):** 3D CNNs require massive GPU memory and cannot understand open-vocabulary text context.

> *"Compared to the original ViDAS research that used expensive cloud LLMs, our model runs locally in 162 milliseconds on standard CPU hardware with zero API costs, making it truly viable for real-time edge deployment."*

---

## 8. Conclusion & Future Work

* **Conclusion:** Successfully designed, trained, and deployed a full-stack, multimodal danger assessment system with high ranking accuracy and sub-200ms latency.
* **Future Work:**
  1. **Autonomous Text Captioning:** Integrate local lightweight vision-language models (e.g., SmolVLM / Moondream) to generate text context automatically from video.
  2. **Spatial Hazard Localization:** Extend the architecture to generate 2D spatial bounding boxes highlighting *where* the hazard is located.
  3. **Embedded Deployment:** Optimize weights with ONNX Runtime for deployment on NVIDIA Jetson dashcams.

> *"In conclusion, we delivered a production-ready, multimodal deep learning pipeline for continuous danger assessment. In future iterations, we plan to incorporate spatial bounding box localization and port the model to NVIDIA Jetson edge hardware for live in-vehicle collision prevention."*
