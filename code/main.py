import os, json, re, random, subprocess
import numpy as np
import cv2
from tqdm import tqdm
from config import *
import audio_processor
import translator
import data_utils
import renderer

def run_pipeline():
    eng = audio_processor.get_input_text()
    if not eng: return

    gloss_str, exp_str = translator.get_isl_from_ollama(eng)
    
    print(f"--- AI Gloss: {gloss_str} ---")
    print(f"--- AI Expressions: {exp_str} ---")

    if not os.path.exists(JSON_PATH):
        print(f"CRITICAL ERROR: {JSON_PATH} not found!")
        return

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
            print(f"--- [!] Word '{word}' not found in mapping. Skipping. ---")
    
    if not final_words:
        print("--- NO MATCHING SIGNS FOUND ---")
        return

    print("-" * 30)
    print(f"RENDERING: {' '.join(final_words)}")
    speed_choice = input("Enter Speed Mode - [N]ormal or [S]low: ").lower()
    playback_fps = 30 if speed_choice == 's' else 60

    full_body, full_face = [], []
    
    for w, ex in zip(final_words, final_exps):
        body_data = None
        video_filename = random.choice(gloss_map[w]) 
        video_id = os.path.splitext(video_filename)[0] 
        expected_npy_name = f"{video_id}_{w.lower()}.npy"
        
        for f in os.listdir(FEATURES_DIR):
            if f.lower() == expected_npy_name.lower():
                body_data = np.load(os.path.join(FEATURES_DIR, f))
                break
        
        if body_data is None: continue 

        # Always returns an array (requested emotion, static, or zeros)
        face_src = data_utils.get_face_npy(ex)

        if full_body:
            # Body Bridge
            b_br = data_utils.bridge(full_body[-1][-1], body_data[0])
            full_body.append(b_br)
            
            # Face Bridge
            f_a, f_b = full_face[-1][-1], face_src[0]
            f_bridge = [(1-t/(len(b_br)-1))*f_a + (t/(len(b_br)-1))*f_b for t in range(len(b_br))]
            full_face.append(np.array(f_bridge))

        full_body.append(body_data)
        full_face.append(data_utils.sync_face_to_duration(face_src, len(body_data)))

    final_b, final_f = np.concatenate(full_body), np.concatenate(full_face)
    vw = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), playback_fps, (W, H))
    
    ref_b = final_b[0]
    print("--- Generating Frames ---")
    for i in tqdm(range(len(final_b))):
        frame = renderer.draw_frame(final_b[i], final_f[i], ref_b)
        vw.write(frame)
    vw.release()
    
    print(f"\n--- SUCCESS: Video saved at {OUTPUT_VIDEO_PATH} ---")
    subprocess.run(['open', '-R', os.path.normpath(OUTPUT_VIDEO_PATH)])

if __name__ == "__main__":
    run_pipeline()




# GRU_TRAINED_MAIN.PY

# import os, json, re, random, subprocess
# import numpy as np
# import cv2
# from tqdm import tqdm
# from config import *
# import audio_processor
# import translator
# import data_utils
# import renderer
# # --- NEW IMPORT ---
# from gru_utils import GRUTransitionManager

# def run_pipeline():
#     # --- INITIALIZE GRU MANAGER ---
#     # Assuming your model is saved in a 'models' folder inside BASE_PATH
#     model_path = os.path.join(BASE_PATH, 'models', 'islvt_sentence_gru.h5')
#     gru_manager = GRUTransitionManager(model_path)

#     eng = audio_processor.get_input_text()
#     if not eng: return

#     gloss_str, exp_str = translator.get_isl_from_ollama(eng)
    
#     print(f"--- AI Gloss: {gloss_str} ---")
#     print(f"--- AI Expressions: {exp_str} ---")

#     if not os.path.exists(JSON_PATH):
#         print(f"CRITICAL ERROR: {JSON_PATH} not found!")
#         return

#     with open(JSON_PATH) as f:
#         raw_map = json.load(f)
#         gloss_map = {k.upper(): v for k, v in raw_map.items()}
    
#     raw_words = re.sub(r'[^\w\s]', '', gloss_str).split()
#     raw_exps = exp_str.split()
    
#     final_words, final_exps = [], []
#     for i in range(len(raw_words)):
#         word, expr = raw_words[i].upper(), raw_exps[i].lower()
#         if word in gloss_map:
#             final_words.append(word)
#             final_exps.append(expr)
#         else:
#             print(f"--- [!] Word '{word}' not found in mapping. Skipping. ---")
    
#     if not final_words:
#         print("--- NO MATCHING SIGNS FOUND ---")
#         return

#     print("-" * 30)
#     print(f"RENDERING: {' '.join(final_words)}")
#     speed_choice = input("Enter Speed Mode - [N]ormal or [S]low: ").lower()
#     playback_fps = 30 if speed_choice == 's' else 60

#     full_body, full_face = [], []
    
#     for w, ex in zip(final_words, final_exps):
#         body_data = None
#         video_filename = random.choice(gloss_map[w]) 
#         video_id = os.path.splitext(video_filename)[0] 
#         expected_npy_name = f"{video_id}_{w.lower()}.npy"
        
#         for f in os.listdir(FEATURES_DIR):
#             if f.lower() == expected_npy_name.lower():
#                 body_data = np.load(os.path.join(FEATURES_DIR, f))
#                 break
        
#         if body_data is None: continue 

#         face_src = data_utils.get_face_npy(ex)

#         if full_body:
#             # --- MODIFIED: GRU BODY BRIDGE ---
#             # We take the last 5 frames of the previous sign as context for the GRU
#             context_frames = full_body[-1][-5:]
#             target_first_frame = body_data[0]
            
#             # Generate 10 frames of smooth transition
#             b_br = gru_manager.generate_bridge(context_frames, target_first_frame, bridge_length=10)
#             full_body.append(b_br)
            
#             # --- FACE BRIDGE (Keep Interpolation) ---
#             f_a, f_b = full_face[-1][-1], face_src[0]
#             # Match face bridge length to the new GRU body bridge length
#             f_bridge = [(1-t/(len(b_br)-1))*f_a + (t/(len(b_br)-1))*f_b for t in range(len(b_br))]
#             full_face.append(np.array(f_bridge))

#         full_body.append(body_data)
#         full_face.append(data_utils.sync_face_to_duration(face_src, len(body_data)))

#     final_b, final_f = np.concatenate(full_body), np.concatenate(full_face)
#     vw = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), playback_fps, (W, H))
    
#     ref_b = final_b[0]
#     print("--- Generating Frames ---")
#     for i in tqdm(range(len(final_b))):
#         frame = renderer.draw_frame(final_b[i], final_f[i], ref_b)
#         vw.write(frame)
#     vw.release()
    
#     print(f"\n--- SUCCESS: Video saved at {OUTPUT_VIDEO_PATH} ---")
#     # Compatibility fix for different OS 'open' commands
#     try:
#         if os.name == 'posix': # Mac/Linux
#             subprocess.run(['open', '-R', os.path.normpath(OUTPUT_VIDEO_PATH)])
#         else: # Windows
#             os.startfile(os.path.dirname(OUTPUT_VIDEO_PATH))
#     except:
#         pass

# if __name__ == "__main__":
#     run_pipeline()