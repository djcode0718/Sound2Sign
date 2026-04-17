# 🤟 Sound2Sign: Data-Efficient Sign Language Generation

## 📌 Overview

**Sound2Sign** is an AI-powered system that converts **English text or speech into sign language animations**.

Unlike traditional deep learning approaches that require large datasets, this project uses a **hybrid, data-efficient pipeline** combining:

* dataset-driven motion retrieval
* GRU-based transition modeling
* cosine interpolation for smoothness

This enables the system to generate **natural, human-like sign transitions using minimal training data**.

---

## 🎯 Motivation

Most modern approaches to sign language generation rely on large-scale neural networks trained end-to-end. While effective, these methods:

* are highly **data-hungry**
* require **expensive training**
* often struggle with **smooth transitions and fine motion details**

This project takes a different approach:

> Instead of learning everything, we reuse real motion data and learn only the **transitions between signs**.

By combining **learned motion dynamics (GRU)** with **mathematical smoothing (cosine interpolation)**, we achieve realistic motion while keeping the system lightweight and efficient.

---

## ⚙️ Features

* 🎤 Voice input (speech-to-text via Faster-Whisper)
* ⌨️ Text input support
* 🧠 Gloss generation (rule-based + LLM-assisted)
* 🤖 GRU-based motion transition generation
* 🎯 Cosine interpolation for enhanced smoothness
* 🎞️ Real-time rendering using OpenCV + MediaPipe
* 🌐 Interactive UI with Streamlit
* ⏬ Downloadable sign language video output

---

## 🧠 How It Works

### 1. Input Processing

* User provides input via text or microphone
* Speech is converted to text using **Faster-Whisper**

---

### 2. Gloss Generation

* Input sentence is converted into a **gloss sequence**
* Uses rule-based linguistic heuristics

```text
Input: "I am going to school"
Gloss: "I SCHOOL GO"
```

---

### 3. Dataset Mapping

* Gloss words are mapped using:

```text
gloss_mapping_f8.json
```

* Each word corresponds to dataset video IDs
* These are converted into preprocessed `.npy` motion sequences

---

### 4. Motion Retrieval

* Motion data is loaded from:

```text
processed_npy_flattened/
```

* Each file contains:

```text
(frames × 258 keypoint features)
```

---

### 5. Hybrid Transition Generation

To connect two signs smoothly:

#### 🔹 GRU (Learned Dynamics)

* Learns human motion patterns
* Generates realistic intermediate frames

#### 🔹 Cosine Interpolation (Mathematical Smoothing)

* Ensures smooth transitions
* Reduces abrupt motion changes

#### 🔹 Final Output

```text
Final Transition = α × GRU + (1 − α) × Cosine
```

---

### 6. Rendering

* Keypoints are rendered using:

  * OpenCV
  * MediaPipe connections

* Final output:

```text
Sign Language Animation (.mp4)
```

---

## 🧱 Tech Stack

* **Python 3.11**
* **TensorFlow 2.19 (Keras 3.x)**
* **Streamlit**
* **MediaPipe**
* **OpenCV**
* **Faster-Whisper**
* **NumPy**

---

## 📦 Installation

### 1. Create environment

```bash
conda create -n s2s_tf python=3.11
conda activate s2s_tf
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run code/app.py
```

---

## 📁 Project Structure

```text
s2s/
│
├── code/
│   ├── app.py                  # Streamlit UI
│   ├── gru_utils.py           # GRU transition logic
│   ├── data_utils.py          # interpolation functions
│   ├── renderer.py            # frame rendering
│   ├── translator.py          # gloss generation
│
├── models/
│   └── gru_model.keras        # trained GRU model
│
├── processed_npy_flattened/   # motion dataset
├── facial_data_npy_full/      # facial expressions
├── gloss_mapping_f8.json      # gloss → dataset mapping
│
├── requirements.txt
└── runtime.txt
```

---

## ⚠️ Notes

* This system is **dataset-driven**, not fully generative
* Gloss generation is heuristic-based and may not match full linguistic grammar
* Output quality depends on dataset coverage

---

## 🚀 Future Work

The goal is to **enhance realism and flexibility while preserving data efficiency**:

* 🔹 Adaptive blending between GRU and cosine interpolation
* 🔹 Improved facial expression modeling for better emotional cues
* 🔹 Smarter gloss generation aligned with dataset vocabulary
* 🔹 Real-time performance optimization for faster rendering
* 🔹 Modular support for multiple sign language datasets

> Future improvements will continue to focus on **efficient, hybrid methods** rather than fully data-heavy end-to-end models.

---

## 👨‍💻 Author

Developed as part of an AI/ML project exploring efficient motion generation and multimodal translation.

---

## ⭐ If you like this project

Give it a ⭐ and feel free to contribute!
