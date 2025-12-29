import streamlit as st
import os
import json
import re
import cv2
import numpy as np
from streamlit_mic_recorder import mic_recorder
import audio_processor
import translator
import data_utils
import renderer
from config import *

# Page Setup
st.set_page_config(page_title="Sound2Sign", page_icon="🤟", layout="centered")

st.title("🤟 Sound2Sign: ISL Interpreter")
st.markdown("Convert English speech or text into **Indian Sign Language (ISL)** animations.")

# 1. SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("Settings")
    speed = st.radio("Playback Speed", ["Normal (60 FPS)", "Slow (30 FPS)"], index=0)
    playback_fps = 30 if "Slow" in speed else 60
    
    st.info("This app uses Mistral (via Ollama) for linguistic translation.")

# 2. INPUT SELECTION
tab1, tab2 = st.tabs(["💬 Text Input", "🎙️ Voice Input"])

eng_input = ""

with tab1:
    eng_input = st.text_input("Enter English sentence:", placeholder="Type here...")

with tab2:
    st.write("Click to record your voice:")
    audio_data = mic_recorder(
        start_prompt="⏺️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key='recorder'
    )
    
    if audio_data:
        # Save temp file for Whisper
        temp_path = "temp_web_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_data['bytes'])
        
        with st.spinner("Transcribing audio..."):
            eng_input = audio_processor.transcriber.transcribe_audio_file(temp_path)
            st.success(f"Recognized: {eng_input}")
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

# 3. PROCESSING PIPELINE
if st.button("Generate Sign Language Video") and eng_input:
    with st.status("Processing Pipeline...", expanded=True) as status:
        # Step 1: Translation
        st.write("Translating English to ISL Gloss...")
        gloss_str, exp_str = translator.get_isl_from_ollama(eng_input)
        st.code(f"Gloss: {gloss_str}\nExpressions: {exp_str}")

        # Step 2: Data Loading & Mapping
        st.write("Fetching sign coordinates...")
        with open(JSON_PATH) as f:
            raw_map = json.load(f)
            gloss_map = {k.upper(): v for k, v in raw_map.items()}
        
        raw_words = re.sub(r'[^\w\s]', '', gloss_str).split()
        raw_exps = exp_str.split()
        
        final_words, final_exps = [], []
        for i in range(len(raw_words)):
            word, expr = raw_words[i].upper(), raw_exps[i].lower()
            if word in gloss_map:
                final_words.append(word)
                final_exps.append(expr)
            else:
                st.warning(f"Word '{word}' missing from dataset.")

        if not final_words:
            st.error("No matching signs found in dataset.")
            st.stop()

        # Step 3: Synthesis
        st.write("Synthesizing body and face frames...")
        full_body, full_face = [], []
        
        for w, ex in zip(final_words, final_exps):
            body_data = None
            video_filename = np.random.choice(gloss_map[w]) 
            video_id = os.path.splitext(video_filename)[0] 
            expected_npy_name = f"{video_id}_{w.lower()}.npy"
            
            for f_name in os.listdir(FEATURES_DIR):
                if f_name.lower() == expected_npy_name.lower():
                    body_data = np.load(os.path.join(FEATURES_DIR, f_name))
                    break
            
            if body_data is None: continue 

            face_src = data_utils.get_face_npy(ex)

            if full_body:
                b_br = data_utils.bridge(full_body[-1][-1], body_data[0])
                full_body.append(b_br)
                f_a, f_b = full_face[-1][-1], face_src[0]
                f_bridge = [(1-t/(len(b_br)-1))*f_a + (t/(len(b_br)-1))*f_b for t in range(len(b_br))]
                full_face.append(np.array(f_bridge))

            full_body.append(body_data)
            full_face.append(data_utils.sync_face_to_duration(face_src, len(body_data)))

        # Step 4: Video Generation
        st.write("Rendering final video...")
        final_b, final_f = np.concatenate(full_body), np.concatenate(full_face)
        vw = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), playback_fps, (W, H))
        
        ref_b = final_b[0]
        for i in range(len(final_b)):
            frame = renderer.draw_frame(final_b[i], final_f[i], ref_b)
            vw.write(frame)
        vw.release()
        
        status.update(label="Rendering Complete!", state="complete", expanded=False)

    # 4. DISPLAY RESULTS
    st.divider()
    st.subheader("Final Output")
    
    # Check if file exists and display
    if os.path.exists(OUTPUT_VIDEO_PATH):
        # We need to convert it to a format browsers like (H.264)
        # Note: If it doesn't play, you might need to use ffmpeg to re-encode,
        # but standard mp4v often works in Streamlit on Mac.
        video_file = open(OUTPUT_VIDEO_PATH, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        
        st.download_button(
            label="⬇️ Download Video",
            data=video_bytes,
            file_name="isl_sign_output.mp4",
            mime="video/mp4"
        )