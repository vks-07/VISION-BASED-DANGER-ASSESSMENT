# 📊 Project Presentation & Defense Script

This document is formatted exactly to match your requested 8-point rubric. You can copy the **"Slide Content"** directly into your PowerPoint slides, and use the **"Speaker Notes"** as your script when presenting.

---

## 1. Research Paper
**Slide Content:**
* **Primary Reference:** *ViDAS: Vision-based Danger Assessment and Scoring* (2024).
* **Authors:** Pranav Gupta, Advith Krishnan, Naman Nanda, et al.
* **Source:** arXiv:2410.00477 (cs.CV)
* **Core Contribution:** A novel dataset of 100 YouTube videos with human-annotated danger ratings on a continuous 0–10 scale.

**Speaker Notes:**
> *"The foundational paper for this project is the 2024 ViDAS paper. While most research focuses on binary anomaly detection, this paper introduces a dataset that treats danger as a continuous spectrum from 0 to 10, complete with temporal coordinates and text summaries."*

---

## 2. Literature Review
**Slide Content:**
* **Existing Work:** Action Recognition (Kinetics), Violence Detection, Anomaly Detection.
* **Limitations of Current Systems:** 
  * Usually binary (e.g., Safe vs. Unsafe) or categorical.
  * Mostly rely on single modalities (only video).
* **The Research Gap:** Lack of *continuous, multimodal* danger assessment that fuses semantic visual understanding with contextual text descriptions.

**Speaker Notes:**
> *"In our literature review, we found that traditional systems like violence or anomaly detectors are strictly binary. They tell you IF something is wrong, but not HOW wrong it is. Furthermore, they lack context. We are addressing this gap by building a multimodal system that combines what is happening visually with contextual text to predict a continuous danger score."*

---

## 3. Problem Statement
**Slide Content:**
* **Problem Statement:** Automated systems struggle to quantify human danger because risk is highly contextual and subjective. Existing AI models fail to capture the nuance between a controlled stunt and a life-threatening hazard.
* **Objectives:** 
  1. To build an end-to-end multimodal AI pipeline.
  2. To fuse visual embeddings (video frames) and text embeddings (descriptions).
  3. To predict a continuous danger score (0.0 to 10.0) using a custom PyTorch regression network.

**Speaker Notes:**
> *"Our problem statement centers on the difficulty of quantifying danger. Our objective was to engineer an end-to-end pipeline that ingests raw video and textual context, extracts high-dimensional semantic features, and uses a custom neural network to output a precise danger score on a 10-point scale."*

---

## 4. Methodology
**Slide Content:**
* **Architecture: 4-Stage Multimodal Pipeline**
  1. **Preprocessing (OpenCV):** Extracts exactly 50 uniform frames per video segment.
  2. **Visual Feature Extraction (CLIP):** Uses `openai/clip-vit-base-patch32` to generate semantic visual vectors. Uses *Temporal Mean-Pooling* to compress 50 frames into a `1x768` vector.
  3. **Text Feature Extraction (BERT):** Uses `bert-base-uncased` to generate a `1x768` contextual vector from the summary.
  4. **Regression Brain (PyTorch MLP):** 
     * Concatenates vectors (1536 dimensions).
     * Applies `LayerNorm` to prevent exploding gradients.
     * Uses `Dropout (0.5)` to prevent overfitting.
     * Outputs score via `Sigmoid * 10.0` scaling.

**Speaker Notes:**
> *"Our architecture relies on early-fusion multimodal Deep Learning. First, OpenCV extracts 50 frames. We pass these through OpenAI's CLIP model and use temporal mean-pooling to create a single visual summary vector. Simultaneously, BERT extracts text embeddings. We fuse these into a 1536-dimensional vector and pass it through a PyTorch Multi-Layer Perceptron. Crucially, we use Layer Normalization and Dropout to stabilize training, and a Sigmoid activation bounded by 10 to ensure valid score outputs."*

---

## 5. Implementation
**Slide Content:**
* **Dataset:** ViDAS Dataset (Filtered ~81 valid YouTube videos after removing dead links).
* **Tech Stack:**
  * **AI/ML:** PyTorch, Hugging Face Transformers, OpenCV, NumPy.
  * **Backend:** FastAPI (Python) – Model caching and real-time inference.
  * **Frontend:** React.js + Vite – Interactive UI with dynamic progress/danger bars.
* **Experimental Setup:** Trained on Mac environment using the Adam Optimizer.

**Speaker Notes:**
> *"For implementation, we used the ViDAS dataset from Hugging Face, writing custom yt-dlp scripts to scrape the raw mp4 files. The AI was built in PyTorch. To make it a usable product, we built a FastAPI backend that caches these massive models in RAM, connected to a modern React frontend for real-time drag-and-drop inference."*

---

## 6. Updated Results
**Slide Content:**
* **Evaluation Metric:** Mean Squared Error (MSE).
* **Initial Architecture Results:** Massive exploding gradients (MSE > 5.0, predictions stuck at 10.0).
* **Final Optimized Results:**
  * Added `LayerNorm` and `Weight Decay (1e-3)`.
  * Shrank network to 128 neurons + 50% Dropout.
  * **Final Training MSE:** Stabilized to `~0.54`.
* *(Tip: You can draw a simple downward line graph on your slide showing loss dropping from 5.5 to 0.54 over 50 epochs).*

**Speaker Notes:**
> *"Our primary evaluation metric is Mean Squared Error. Initially, the high dimensionality of our vectors caused vanishing gradients, resulting in the model predicting max danger for everything. By implementing Layer Normalization, applying L2 weight decay, and aggressively shrinking the network to fight overfitting, we successfully drove the MSE down to 0.54 over 50 epochs."*

---

## 7. Comparison
**Slide Content:**
* **Compared to Single-Modality:** Vision-only models struggle with context (e.g., a movie stunt vs. real danger). Adding BERT text embeddings provides critical contextual grounding.
* **Compared to the Original ViDAS Paper:**
  * *Original Paper:* Relied heavily on calling massive, slow, closed-source LLMs (like GPT-4) to evaluate danger.
  * *Our Method:* We built a deterministic, lightweight PyTorch MLP for the regression head. This allows for lightning-fast, local edge-inference without paying for API calls.

**Speaker Notes:**
> *"When comparing our approach to others, single-modality vision models fail because they lack context. Multimodal fusion solves this. Furthermore, while the original ViDAS paper focused on using massive language models for evaluation, we engineered a lightweight PyTorch neural network. This allows our application to run locally and rapidly without relying on expensive, slow API calls."*

---

## 8. Conclusion and Future Work
**Slide Content:**
* **Conclusion:** Successfully built an end-to-end, full-stack multimodal AI application capable of extracting real-time video/text features and predicting continuous danger scores.
* **Current Limitations:** The "Curse of Dimensionality" / Data Starvation. 81 videos are too few for deep generalization on out-of-distribution unseen videos.
* **Next Steps / Future Work:**
  1. **Scale Dataset:** Expand training data from 100 to 10,000+ videos for robust real-world generalization.
  2. **Temporal Dynamics:** Replace the simple "Mean Pooling" of frames with an LSTM or Video-Swin Transformer to capture motion over time (e.g., the *speed* of a falling object).

**Speaker Notes:**
> *"In conclusion, we successfully delivered a full-stack AI pipeline. However, we identified a critical limitation: Data Starvation. Training a 1536-dimensional network on 81 videos causes it to memorize data rather than generalize perfectly to unseen videos. Our next steps would be to scale the dataset to 10,000+ videos, and replace our mean-pooling technique with an LSTM to better understand temporal motion and speed."*
