# 🤟 Sound2Sign: Data-Efficient Sign Language Generation

## 📌 Overview

**Sound2Sign** is an AI-powered system that converts **English text or speech into sign language animations**.

Unlike traditional deep learning approaches that require large datasets, this project uses a **hybrid, data-efficient pipeline** combining:

* dataset-driven motion retrieval
* GRU-based transition modeling
* cosine interpolation for smoothness
* a dedicated facial expression (non-manual marker) pipeline

This enables the system to generate **natural, human-like sign transitions using minimal training data**, while still preserving the grammatical information carried by facial expression — something sign languages depend on as much as hand motion.

---

## 🎯 Motivation

Most modern approaches to sign language generation rely on large-scale neural networks trained end-to-end. While effective, these methods:

* are highly **data-hungry**
* require **expensive training**
* often struggle with **smooth transitions and fine motion details**
* frequently treat facial expression as an afterthought, even though sign languages use it grammatically

This project takes a different approach:

> Instead of learning everything, we reuse real motion data and learn only the **transitions between signs** — for both hands and face.

By combining **learned motion dynamics (GRU)**, **mathematical smoothing (cosine interpolation)**, and a **rule-driven facial expression layer**, we achieve realistic, linguistically meaningful output while keeping the system lightweight.

---

## ⚙️ Features

* 🎤 Voice input (speech-to-text via Faster-Whisper, with a `speech_recognition` microphone fallback path)
* ⌨️ Text input support
* 🧠 Gloss generation via a locally-hosted LLM (Mistral, served through Ollama) using a strict structural-transformation prompt
* 😐 Facial expression (non-manual marker) generation aligned one-to-one with each gloss token
* 🤖 GRU-based motion transition generation for body pose
* 🎯 Cosine interpolation for enhanced smoothness (used both standalone and blended with GRU output)
* 🎞️ Real-time rendering using OpenCV + MediaPipe (body skeleton, hands, and full face mesh)
* ⏱️ Adjustable playback speed (Normal 60 FPS / Slow 30 FPS)
* 🌐 Interactive UI with Streamlit
* ⏬ Downloadable sign language video output

---

## 🧠 How It Works

### 1. Input Processing

* User provides input via text or microphone
* Speech is converted to text using **Faster-Whisper** (`audio_processor.py`)

---

### 2. Gloss + Facial Expression Generation

Unlike a purely rule-based system, gloss generation is delegated to a **locally running LLM (Mistral via Ollama)**, prompted to act as a deterministic structural transformer rather than a free-form translator. The prompt enforces:

* removal of articles, auxiliary verbs, and punctuation
* preposition removal while preserving their objects
* verb lemmatization
* a fixed word order: `TIME(S) → SUBJECT(S) → OBJECT(S) → LOCATION(S) → VERB(S)`
* WH-words moved to the end of the sequence
* negation placed after the verb

Critically, the same call also produces a **facial expression tag for every gloss token**, chosen from a fixed vocabulary:

```
static, eyebrows-up, eyebrows-down, head-shake,
happy-exp, sad-exp, angry-exp, surprise
```

These aren't cosmetic — in sign languages, eyebrow position marks question type, head movement marks negation, and expression carries affect that hands alone don't convey. If the LLM returns a mismatched number of gloss tokens and expressions, the app pads or truncates expressions to `static` so the two sequences always stay aligned.

```text
Input: "I am going to school"
Gloss: "I SCHOOL GO"
Expressions: "static static static"
```

---

### 3. Dataset Mapping

* Gloss words are mapped using `gloss_mapping_f8.json` to candidate dataset video IDs
* Words not present in the mapping are skipped and flagged in the UI as missing from the dataset

---

### 4. Motion Retrieval

**Body motion** is loaded from `processed_npy_flattened/`, where each file holds `(frames × 258 keypoint features)`.

**Facial motion** is loaded separately from `facial_data_npy_full/<expression>/`, keyed by the expression tag assigned in step 2. Lookup uses a fallback chain:

1. Try the requested expression's folder
2. If missing/empty, fall back to `static`
3. If even `static` is unavailable, fall back to a zeroed array so rendering never crashes

Facial motion is then time-stretched or looped (`sync_face_to_duration`) to match the duration of the body clip it's paired with, so hands and face stay synchronized.

---

### 5. Hybrid Transition Generation

To connect two consecutive signs smoothly, **body and face are bridged separately**:

#### 🔹 Body: GRU (Learned Dynamics)

* An autoregressive GRU predicts intermediate body frames conditioned on the preceding motion context

#### 🔹 Body: Cosine Interpolation (Mathematical Smoothing)

* A second, independent cosine-eased bridge is computed directly between the last and next body frame

#### 🔹 Body: Two-Stage Hybrid Blend

The GRU bridge is itself already a blend of its own autoregressive output with a cosine curve (blend weight ramping 0.2 → 0.8 across the bridge). That result is then blended a second time with the standalone cosine bridge:

```text
Final Body Transition = 0.7 × GRU-hybrid + 0.3 × Cosine
```

#### 🔹 Face: Linear Interpolation

* Facial transitions use simple linear interpolation between the last face frame of one sign and the first face frame of the next — kept intentionally simpler than the body pipeline since expression changes are typically more abrupt/discrete than hand motion.

---

### 6. Rendering

Keypoints are rendered per-frame using OpenCV, with:

* a static shoulder/hip skeleton frame
* arm and hand joints reconstructed via direction vectors and reference bone lengths
* MediaPipe hand connections for fingers
* the full MediaPipe face mesh — tesselation, face oval, lips, eyes, and irises — realigned each frame to the neck anchor point so the face doesn't drift relative to the body

Final output is written frame-by-frame to an `.mp4` via `cv2.VideoWriter` at the user-selected FPS (60 or 30).

---

## 🧱 Tech Stack

* **Python 3.11**
* **TensorFlow 2.19 (Keras 3.x)** — GRU transition model
* **Streamlit** + **streamlit-mic-recorder** — UI and in-browser mic capture
* **Ollama** running the **Mistral** model — gloss + expression generation (LLM must be running locally; this is a hard dependency, not optional)
* **MediaPipe** — hand/face landmark connections for rendering
* **OpenCV** — frame rendering and video writing
* **Faster-Whisper** — primary speech-to-text
* **SpeechRecognition** — secondary/CLI microphone input path
* **Requests** — HTTP calls to the local Ollama API
* **NumPy**

---

## 📦 Installation

### 1. Install and start Ollama, then pull Mistral

The gloss/expression generation step depends on a **locally running Ollama server**. The app will fail (or return empty gloss/expressions) if this isn't running.

```bash
# Install Ollama separately (see ollama.com), then:
ollama pull mistral
ollama serve
```

By default the app expects Ollama at `http://localhost:11434/api/generate` (configurable in `config.py`).

### 2. Create environment

```bash
conda create -n s2s_tf python=3.11
conda activate s2s_tf
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure paths

`code/config.py` currently hardcodes `BASE_PATH` to a specific machine's directory. Update `BASE_PATH` (and confirm `FEATURES_DIR`, `JSON_PATH`, `FACE_NPY_DIR`, `OUTPUT_VIDEO_PATH` derive correctly from it) to match your own project location before running.

---

## ▶️ Run the App

```bash
streamlit run code/app.py
```

Make sure Ollama is already running (`ollama serve`) before launching Streamlit.

---

## 📁 Project Structure

```text
s2s/
│
├── code/
│   ├── app.py                  # Streamlit UI + main pipeline orchestration
│   ├── config.py               # Paths, rendering constants, MediaPipe connection sets
│   ├── audio_processor.py      # Faster-Whisper transcription + CLI mic fallback
│   ├── translator.py           # LLM (Ollama/Mistral) gloss + expression generation
│   ├── data_utils.py           # facial data lookup + body/face interpolation helpers
│   ├── gru_utils.py            # GRU transition model wrapper + hybrid blending
│   ├── renderer.py             # per-frame body/hand/face rendering
│
├── train_gru.py                # GRU training script (run in Google Colab, not locally)
│
├── models/
│   └── gru_model.keras         # trained GRU weights
│
├── processed_npy_flattened/    # body motion dataset (258-feature keypoint sequences)
├── facial_data_npy_full/       # facial expression dataset, one subfolder per expression
├── gloss_mapping_f8.json       # gloss word → dataset video ID mapping
│
├── requirements.txt
└── runtime.txt
```

---

## ⚠️ Notes

* This system is **dataset-driven**, not fully generative — output quality depends entirely on dataset coverage for both body motion and facial expressions.
* Gloss and expression generation depend on an external LLM call (Ollama/Mistral) rather than being purely deterministic; results can vary slightly between runs even at temperature 0 depending on the model/version installed locally.
* If a requested facial expression has no data on disk, the system silently substitutes `static` (or zeros as a last resort) — check console output for these fallback warnings if animations look emotionally flat.
* `BASE_PATH` in `config.py` is machine-specific and must be updated before running on a new setup.

---

## 🚀 Future Work

The goal is to **enhance realism and flexibility while preserving data efficiency**:

* 🔹 Adaptive blending between GRU and cosine interpolation (currently fixed weights)
* 🔹 GRU-based (rather than purely linear) facial transition modeling, to match the sophistication of the body pipeline
* 🔹 Smarter gloss generation aligned with dataset vocabulary, reducing reliance on a live LLM call
* 🔹 Real-time performance optimization for faster rendering
* 🔹 Modular support for multiple sign language datasets

> Future improvements will continue to focus on **efficient, hybrid methods** rather than fully data-heavy end-to-end models.

---

## 👨‍💻 Author

Developed as part of an AI/ML project exploring efficient motion generation and multimodal translation.

---

## ⭐ If you like this project

Give it a ⭐ and feel free to contribute!
