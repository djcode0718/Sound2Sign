# Sound2Sign: AI-Powered ISL Interpreter 🤟

Sound2Sign is a comprehensive pipeline that translates English speech or text into **Indian Sign Language (ISL)** animations. It utilizes Large Language Models (LLMs) for linguistic glossing and a custom rendering engine for 3D-style skeletal animations.

## 🚀 Features
- **Voice-to-Sign**: Real-time transcription using `Faster-Whisper`.
- **Linguistic Engine**: Uses `Ollama` (Mistral) to transform English into ISL Gloss (TIME-SUBJECT-OBJECT-VERB) and facial expressions.
- **Skeletal Synthesis**: Generates smooth animations by interpolating MediaPipe landmarks.
- **Web Interface**: A clean, interactive UI built with `Streamlit`.
- **Hybrid Rendering**: Combines pre-recorded sign data with dynamic facial expression synthesis.

## 🛠️ Architecture
1. **Audio Processor**: Captures voice and transcribes it using OpenAI's Whisper (via `faster-whisper`).
2. **Translator**: A rule-based LLM engine that outputs JSON containing ISL Gloss and facial cues.
3. **Data Utility**: Bridges the gap between sign tokens using mathematical interpolation or GRU-based transitions.
4. **Renderer**: A custom OpenCV-based engine that draws the skeleton, hands, and face mesh in a synchronized 800x800 canvas.

## 📂 Project Structure
- `app.py`: The Streamlit web application.
- `main.py`: The core CLI pipeline.
- `translator.py`: Handles interaction with Ollama/Mistral.
- `renderer.py`: Logic for drawing the body, hands, and face mesh.
- `data_utils.py`: Manages coordinate bridging and facial data synchronization.
- `config.py`: Global paths and rendering constants.

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Sound2Sign.git](https://github.com/YOUR_USERNAME/Sound2Sign.git)
   cd Sound2Sign
   
Install Dependencies:

Bash

pip install -r requirements.txt
External Requirements:

Install Ollama and pull the Mistral model:

Bash

ollama pull mistral
Ensure you have the processed_npy_flattened and facial_data_npy_full datasets in the path specified in config.py.

🖥️ Usage
To run the Web UI:
Bash

streamlit run app.py
To run the CLI:
Bash

python main.py
🧠 Future Enhancements
Integration of a GRU (Gated Recurrent Unit) model for more fluid transitions between signs.

Support for a larger vocabulary of ISL signs.

Deployment via Docker for easier scaling.
