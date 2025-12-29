import os, json, random, requests, subprocess, re
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
# import whisper
import speech_recognition as sr
from tqdm import tqdm
from faster_whisper import WhisperModel
import speech_recognition as sr
import os

# ===================== LOCAL CONFIGURATION =====================
BASE_PATH = r"/Users/sj/Documents/sound2sign"
FEATURES_DIR = os.path.join(BASE_PATH, 'processed_npy_flattened')
JSON_PATH = os.path.join(BASE_PATH, 'gloss_mapping_f8.json')
FACE_NPY_DIR = os.path.join(BASE_PATH, 'facial_data_npy_full')
OLLAMA_URL = "http://localhost:11434/api/generate"

# Rendering Constants
W, H = 800, 800
SCALE = 350
FACE_SCALE_MULT = 1.5 
OFFSET_X, OFFSET_Y = 400, 400
CAMERA_Y_SHIFT, NECK_OFFSET = -0.25, 0.45
F_TRANSITION_LEN = 8

STATIC_BODY = {11: (-0.45, 0.0), 12: (0.45, 0.0), 23: (-0.35, 2.038), 24: (0.35, 2.038)}
mp_fmc = mp.solutions.face_mesh_connections
HAND_CONNECTIONS = mp.solutions.holistic.HAND_CONNECTIONS

# # ===================== 1. INPUT LOGIC (VOICE/TEXT) =====================
# model = whisper.load_model("base")
# def get_input_text():
#     mode = input("\n[V]oice Input or [T]ype English? ").lower()
#     if mode == 'v':
#         recognizer = sr.Recognizer()
#         try:
#             with sr.Microphone() as source:
#                 print("--- Calibrating... ---")
#                 recognizer.adjust_for_ambient_noise(source, duration=1)
#                 print("--- Speak Now ---")
#                 audio = recognizer.listen(source, timeout=5)
#             temp_path = "live_audio.wav"
#             with open(temp_path, "wb") as f: f.write(audio.get_wav_data())
#             print("--- Transcribing (Whisper) ---")
#             # model = whisper.load_model("base")
#             result = model.transcribe(temp_path)
#             os.remove(temp_path)
#             return result["text"].strip()
#         except Exception as e:
#             print(f"Mic Error: {e}. Switching to text mode.")
#     return input("Enter English sentence: ")


# ===================== WHISPER MODULE =====================
class WhisperTranscriber:
    def __init__(self, model_size="base"):
        # Optimized for Mac: int8 quantization makes it very fast on CPU
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe_audio_file(self, audio_file_path: str) -> str:
        if not os.path.exists(audio_file_path):
            return ""
        
        # beam_size=5 provides a good balance between speed and accuracy
        segments, info = self.model.transcribe(audio_file_path, beam_size=5)
        transcribed_text = "".join(segment.text for segment in segments).strip()
        return transcribed_text

# Initialize the transcriber GLOBALLY so it loads once
print("--- Initializing Faster-Whisper ---")
transcriber = WhisperTranscriber(model_size="base")

# ===================== 1. INPUT LOGIC (VOICE/TEXT) =====================

def get_input_text():
    mode = input("\n[V]oice Input or [T]ype English? ").lower()
    
    if mode == 'v':
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print("--- Calibrating Mic... ---")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                
                print("--- Speak Now ---")
                # Increased timeout/phrase_limit for more natural speech
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            temp_path = "live_audio.wav"
            with open(temp_path, "wb") as f: 
                f.write(audio.get_wav_data())
             
            print("--- Transcribing (Faster-Whisper) ---")
            text = transcriber.transcribe_audio_file(temp_path)
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            if text:
                print(f"Recognized: {text}")
                return text
            else:
                print("Could not understand audio. Switching to text.")
                
        except Exception as e:
            print(f"Mic Error: {e}. Switching to text mode.")
            
    return input("Enter English sentence: ")


def get_isl_from_ollama(english_text):
    prompt = f"""
You are a rule-based English-to-Gloss converter.

You are NOT performing natural language translation.
You are applying deterministic linguistic heuristics to approximate
Indian Sign Language (ISL) gloss structure.

Your task is to convert an English sentence into:
1) A GLOSS sequence (uppercase tokens)
2) A FACIAL EXPRESSION sequence aligned one-to-one with each gloss token

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------

Output MUST be valid JSON ONLY.

{{
  "gloss": "WORD1 WORD2 WORD3",
  "expressions": "expr1 expr2 expr3"
}}

The number of gloss tokens MUST exactly match the number of expressions.

Do NOT include explanations, comments, or extra text.

--------------------------------------------------
GLOSS GENERATION RULES
--------------------------------------------------

1. REMOVE the following word types:
   - Articles: a, an, the
   - Auxiliary verbs: do, does, did
   - Punctuation marks

2. PREPOSITIONS:
   - Remove prepositions: from, to, of, in, on, at
   - IMPORTANT: When removing a preposition, DO NOT remove its object.
     Example:
       "from the school" → "SCHOOL"

3. KEEP only semantic content words:
   - nouns
   - main verbs
   - pronouns
   - question words

4. VERB NORMALIZATION:
   - Convert all verbs to their base (lemma) form.
   - Do NOT use tense-inflected forms (e.g., played, eating, goes).
   - Tense must be expressed ONLY through time words if present.

5. WORD ORDER (heuristic ISL structure):
   - TIME → SUBJECT → OBJECT → VERB
   - If no time word exists, start with SUBJECT.
   - Reordering is allowed, substitution is NOT.

6. WH-QUESTIONS:
   - WH-word must appear at the END of the gloss.

7. NEGATION:
   - Keep negation words (e.g., NOT, NEVER).
   - Place negation AFTER the verb.

8. DO NOT add classifiers, aspect markers, or new vocabulary.

9. ALL gloss tokens MUST be in UPPERCASE.

--------------------------------------------------
FACIAL EXPRESSION RULES
--------------------------------------------------

Allowed expressions ONLY:
- static
- eyebrows-up
- eyebrows-down
- head-shake
- happy-exp
- sad-exp
- angry-exp
- surprise

Rules:

1. YES/NO QUESTIONS:
   - All gloss tokens → eyebrows-up

2. WH-QUESTIONS:
   - WH-word → eyebrows-down
   - All other tokens → static

3. NEGATION:
   - Negation token → head-shake

4. EMOTIONAL EXPRESSIONS:
   - Apply ONLY if an explicit emotion word is present
     (e.g., happy, sad, angry, surprised).
   - Apply emotion ONLY to that word.
   - Do NOT infer emotion from punctuation.

5. STATEMENTS:
   - All tokens → static

6. Each gloss token MUST have exactly one facial expression.

--------------------------------------------------
IMPORTANT CONSTRAINTS
--------------------------------------------------

- Do NOT invent words.
- Do NOT infer missing meaning.
- If a word does not exist in sign language vocabulary, still output it.
- Filtering or validation happens downstream.

--------------------------------------------------
INPUT
--------------------------------------------------

English Sentence:
"{english_text}"

"""



    payload = {
        "model": "mistral",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0}
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        data = json.loads(response.json()["response"])
        return data.get("gloss", ""), data.get("expressions", "")
    except Exception as e:
        print("Ollama Error:", e)
        return "", ""

# ===================== 3. CORE UTILITIES =====================

def bridge(a, b):
    d = np.linalg.norm(a[195:197] - b[195:197])
    n = int(np.clip(d * 35, 8, 20))
    out = [((1-(0.5-0.5*np.cos(np.pi*(i/(n-1)))))*a + (0.5-0.5*np.cos(np.pi*(i/(n-1))))*b) for i in range(n)]
    return np.array(out)

def get_face_npy(expression):
    folder_path = os.path.join(FACE_NPY_DIR, expression.strip().lower())
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        if files: return np.load(os.path.join(folder_path, random.choice(files)))
    return None

def sync_face_to_duration(face_data, target_len):
    out = []
    while len(out) < target_len:
        to_add = min(len(face_data), target_len - len(out))
        out.extend(face_data[:to_add])
        if len(out) < target_len:
            a, b = face_data[-1], face_data[0]
            for t in range(min(F_TRANSITION_LEN, target_len - len(out))):
                out.append((1-(t/F_TRANSITION_LEN))*a + (t/F_TRANSITION_LEN)*b)
    return np.array(out[:target_len])

def P(pt): return (int(pt[0]*SCALE)+OFFSET_X, int((pt[1]+CAMERA_Y_SHIFT)*SCALE)+OFFSET_Y)

# ===================== 4. RENDERER WITH PAIRED FILTERING =====================

# def run_pipeline(eng, gloss_str, exp_str):
#     out_name = os.path.join(BASE_PATH, 'FINAL_SIGN_VIDEO.mp4')
#     with open(JSON_PATH) as f:
#         raw_map = json.load(f)
#         # Normalize all keys to uppercase for reliable matching
#         gloss_map = {k.upper(): v for k, v in raw_map.items()}
    
#     # Clean gloss words (strip punctuation)
#     raw_words = re.sub(r'[^\w\s]', '', gloss_str).split()
#     raw_exps = exp_str.split()
    
#     # --- PAIRED FILTERING: Remove missing words AND their expressions simultaneously ---
#     final_words, final_exps = [], []
#     min_len = min(len(raw_words), len(raw_exps))
    
#     for i in range(min_len):
#         word, expr = raw_words[i].upper(), raw_exps[i].lower()
#         if word in gloss_map:
#             final_words.append(word)
#             final_exps.append(expr)
    
#     if not final_words:
#         print("--- NO MATCHING SIGNS FOUND IN DATASET ---")
#         return

#     print("-" * 30)
#     print(f"OLLAMA OUTPUT:\n GLOSS: {gloss_str}\n EXPR:  {exp_str}")
#     print(f"ACTUAL RENDERING:\n GLOSS: {' '.join(final_words)}\n EXPR:  {' '.join(final_exps)}")
#     print("-" * 30)

#     full_body, full_face = [], []
#     for w, ex in zip(final_words, final_exps):
#         body_data = None
#         # Select random video ID from JSON list
#         video_filename = random.choice(gloss_map[w]) # e.g., "00335.mp4"
#         video_id = os.path.splitext(video_filename)[0] # e.g., "00335"
        
#         # Look for the npy file named "video id_gloss.npy"
#         expected_npy_name = f"{video_id}_{w.lower()}.npy"
        
#         for f in os.listdir(FEATURES_DIR):
#             if f.lower() == expected_npy_name.lower():
#                 body_data = np.load(os.path.join(FEATURES_DIR, f))
#                 break
        
#         if body_data is None: continue 

#         face_src = get_face_npy(ex)
#         if full_body:
#             b_br = bridge(full_body[-1][-1], body_data[0]); full_body.append(b_br)
#             f_a, f_b = full_face[-1][-1], (face_src[0] if face_src is not None else full_face[0][0])
#             full_face.append(np.array([(1-t/(len(b_br)-1))*f_a + (t/(len(b_br)-1))*f_b for t in range(len(b_br))]))

#         full_body.append(body_data)
#         full_face.append(sync_face_to_duration(face_src, len(body_data)) if face_src is not None else np.repeat([full_face[-1][-1] if full_face else get_face_npy('static')[0]], len(body_data), axis=0))

#     final_b, final_f = np.concatenate(full_body), np.concatenate(full_face)
#     vw = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
    
#     ref = final_b[0]
#     def dist(a,b): return np.linalg.norm(np.array(a)-np.array(b))
#     L_UP_L, L_LO_L = dist((ref[44],ref[45]),(ref[52],ref[53])), dist((ref[52],ref[53]),(ref[60],ref[61]))
#     L_UP_R, L_LO_R = dist((ref[48],ref[49]),(ref[56],ref[57])), dist((ref[56],ref[57]),(ref[64],ref[65]))
#     F_SC = SCALE * FACE_SCALE_MULT
#     def PF(pt): return (int(pt[0]*F_SC)+OFFSET_X, int((pt[1]+CAMERA_Y_SHIFT)*F_SC)+OFFSET_Y)

#     print("--- Rendering Video ---")
#     for i in tqdm(range(len(final_b))):
#         img = np.zeros((H, W, 3), np.uint8)
#         fb, ff = final_b[i], final_f[i]
#         for a,b in [(11,12),(11,23),(12,24),(23,24)]: cv2.line(img, P(STATIC_BODY[a]), P(STATIC_BODY[b]), (255,255,255), 3)
#         dir_le = (np.array([fb[52]-fb[44], fb[53]-fb[45]])) / (np.linalg.norm([fb[52]-fb[44], fb[53]-fb[45]])+1e-6)
#         dir_re = (np.array([fb[56]-fb[48], fb[57]-fb[49]])) / (np.linalg.norm([fb[56]-fb[48], fb[57]-fb[49]])+1e-6)
#         j13, j14 = (STATIC_BODY[12][0]+dir_le[0]*L_UP_L, STATIC_BODY[12][1]+dir_le[1]*L_UP_L), (STATIC_BODY[11][0]+dir_re[0]*L_UP_R, STATIC_BODY[11][1]+dir_re[1]*L_UP_R)
#         dir_lw = (np.array([fb[60]-fb[52], fb[61]-fb[53]])) / (np.linalg.norm([fb[60]-fb[52], fb[61]-fb[53]])+1e-6)
#         dir_rw = (np.array([fb[64]-fb[56], fb[65]-fb[57]])) / (np.linalg.norm([fb[64]-fb[56], fb[65]-fb[57]])+1e-6)
#         j15, j16 = (j13[0]+dir_lw[0]*L_LO_L, j13[1]+dir_lw[1]*L_LO_L), (j14[0]+dir_rw[0]*L_LO_R, j14[1]+dir_rw[1]*L_LO_R)
#         for s,e in [(P(STATIC_BODY[12]),P(j13)),(P(j13),P(j15)),(P(STATIC_BODY[11]),P(j14)),(P(j14),P(j16))]: cv2.line(img, s, e, (255,255,255), 2)
#         for side, w_idx, s_idx in [('l',j15,132),('r',j16,195)]:
#             wr_p, root_p = P(w_idx), P((fb[s_idx], fb[s_idx+1]))
#             pts = {k: (wr_p[0] + P((fb[s_idx+k*3], fb[s_idx+k*3+1]))[0] - root_p[0], wr_p[1] + P((fb[s_idx+k*3], fb[s_idx+k*3+1]))[1] - root_p[1]) for k in range(21)}
#             for c in HAND_CONNECTIONS: cv2.line(img, pts[c[0]], pts[c[1]], (0,255,0) if side=='l' else (0,0,255), 2)
#         mid = ((STATIC_BODY[11][0]+STATIC_BODY[12][0])/2, (STATIC_BODY[11][1]+STATIC_BODY[12][1])/2)
#         t_n, s_n = P((mid[0], mid[1]-NECK_OFFSET)), PF((ff[1,0], ff[1,1]))
#         dx, dy = t_n[0]-s_n[0], t_n[1]-s_n[1]
#         def d_f(c_set, col, th):
#             for c in c_set:
#                 p1, p2 = PF(ff[c[0]]), PF(ff[c[1]])
#                 cv2.line(img, (p1[0]+dx, p1[1]+dy), (p2[0]+dx, p2[1]+dy), col, th, cv2.LINE_AA)
#         d_f(mp_fmc.FACEMESH_TESSELATION, (60,60,60), 1)
#         d_f(mp_fmc.FACEMESH_FACE_OVAL, (255,255,255), 2)
#         d_f(mp_fmc.FACEMESH_LIPS, (255,255,255), 2)
#         d_f(mp_fmc.FACEMESH_RIGHT_EYE, (0,0,255), 2)
#         d_f(mp_fmc.FACEMESH_LEFT_EYE, (0,255,0), 2)
#         try: d_f(mp_fmc.FACEMESH_RIGHT_IRIS, (0,255,255), 2); d_f(mp_fmc.FACEMESH_LEFT_IRIS, (255,255,0), 2)
#         except: d_f(mp_fmc.FACEMESH_IRISES, (0,255,255), 2)
#         for idx, col in [(468, (0,255,255)), (473, (255,255,0))]:
#             p = PF(ff[idx]); cv2.circle(img, (p[0]+dx, p[1]+dy), 3, col, -1)
#         vw.write(img)
#     vw.release()
#     print(f"\n--- SUCCESS ---")
#     subprocess.run(['explorer', '/select,', os.path.normpath(out_name)])

# ===================== 4. RENDERER WITH SPEED CONTROL =====================

# def run_pipeline(eng, gloss_str, exp_str):
#     out_name = os.path.join(BASE_PATH, 'FINAL_SIGN_VIDEO_2.mp4')
#     with open(JSON_PATH) as f:
#         raw_map = json.load(f)
#         gloss_map = {k.upper(): v for k, v in raw_map.items()}
    
#     raw_words = re.sub(r'[^\w\s]', '', gloss_str).split()
#     raw_exps = exp_str.split()
    
#     final_words, final_exps = [], []
#     min_len = min(len(raw_words), len(raw_exps))
    
#     for i in range(min_len):
#         word, expr = raw_words[i].upper(), raw_exps[i].lower()
#         if word in gloss_map:
#             final_words.append(word)
#             final_exps.append(expr)
    
#     if not final_words:
#         print("--- NO MATCHING SIGNS FOUND IN DATASET ---")
#         return

#     # --- SPEED SELECTION PROMPT ---
#     print("-" * 30)
#     print(f"ACTUAL RENDERING: {' '.join(final_words)}")
#     speed_choice = input("Choose video speed - [N]ormal (60fps) or [S]low (30fps): ").lower()
    
#     # Logic: Normal is 60fps. Slow is 30fps (doubles visual duration).
#     target_fps = 30 if speed_choice == 's' else 60
#     print(f"Target Playback Speed: {target_fps} FPS")
#     print("-" * 30)

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

#         face_src = get_face_npy(ex)
#         if full_body:
#             b_br = bridge(full_body[-1][-1], body_data[0]); full_body.append(b_br)
#             f_a, f_b = full_face[-1][-1], (face_src[0] if face_src is not None else full_face[0][0])
#             f_bridge = [(1-t/(len(b_br)-1))*f_a + (t/(len(b_br)-1))*f_b for t in range(len(b_br))]
#             full_face.append(np.array(f_bridge))

#         full_body.append(body_data)
#         full_face.append(sync_face_to_duration(face_src, len(body_data)) if face_src is not None else np.repeat([full_face[-1][-1] if full_face else get_face_npy('static')[0]], len(body_data), axis=0))

#     final_b, final_f = np.concatenate(full_body), np.concatenate(full_face)
    
#     # Initialize VideoWriter with chosen FPS
#     vw = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (W, H))
    
#     ref = final_b[0]
#     def dist(a,b): return np.linalg.norm(np.array(a)-np.array(b))
#     L_UP_L, L_LO_L = dist((ref[44],ref[45]),(ref[52],ref[53])), dist((ref[52],ref[53]),(ref[60],ref[61]))
#     L_UP_R, L_LO_R = dist((ref[48],ref[49]),(ref[56],ref[57])), dist((ref[56],ref[57]),(ref[64],ref[65]))
#     F_SC = SCALE * FACE_SCALE_MULT
#     def PF(pt): return (int(pt[0]*F_SC)+OFFSET_X, int((pt[1]+CAMERA_Y_SHIFT)*F_SC)+OFFSET_Y)

#     print("--- Rendering Video ---")
#     for i in tqdm(range(len(final_b))):
#         img = np.zeros((H, W, 3), np.uint8)
#         fb, ff = final_b[i], final_f[i]
#         for a,b in [(11,12),(11,23),(12,24),(23,24)]: cv2.line(img, P(STATIC_BODY[a]), P(STATIC_BODY[b]), (255,255,255), 3)
        
#         # Skeleton/Hand/Face logic remains exactly as provided
#         dir_le = (np.array([fb[52]-fb[44], fb[53]-fb[45]])) / (np.linalg.norm([fb[52]-fb[44], fb[53]-fb[45]])+1e-6)
#         dir_re = (np.array([fb[56]-fb[48], fb[57]-fb[49]])) / (np.linalg.norm([fb[56]-fb[48], fb[57]-fb[49]])+1e-6)
#         j13, j14 = (STATIC_BODY[12][0]+dir_le[0]*L_UP_L, STATIC_BODY[12][1]+dir_le[1]*L_UP_L), (STATIC_BODY[11][0]+dir_re[0]*L_UP_R, STATIC_BODY[11][1]+dir_re[1]*L_UP_R)
#         dir_lw = (np.array([fb[60]-fb[52], fb[61]-fb[53]])) / (np.linalg.norm([fb[60]-fb[52], fb[61]-fb[53]])+1e-6)
#         dir_rw = (np.array([fb[64]-fb[56], fb[65]-fb[57]])) / (np.linalg.norm([fb[64]-fb[56], fb[65]-fb[57]])+1e-6)
#         j15, j16 = (j13[0]+dir_lw[0]*L_LO_L, j13[1]+dir_lw[1]*L_LO_L), (j14[0]+dir_rw[0]*L_LO_R, j14[1]+dir_rw[1]*L_LO_R)
#         for s,e in [(P(STATIC_BODY[12]),P(j13)),(P(j13),P(j15)),(P(STATIC_BODY[11]),P(j14)),(P(j14),P(j16))]: cv2.line(img, s, e, (255,255,255), 2)
#         for side, w_idx, s_idx in [('l',j15,132),('r',j16,195)]:
#             wr_p, root_p = P(w_idx), P((fb[s_idx], fb[s_idx+1]))
#             pts = {k: (wr_p[0] + P((fb[s_idx+k*3], fb[s_idx+k*3+1]))[0] - root_p[0], wr_p[1] + P((fb[s_idx+k*3], fb[s_idx+k*3+1]))[1] - root_p[1]) for k in range(21)}
#             for c in HAND_CONNECTIONS: cv2.line(img, pts[c[0]], pts[c[1]], (0,255,0) if side=='l' else (0,0,255), 2)
#         mid = ((STATIC_BODY[11][0]+STATIC_BODY[12][0])/2, (STATIC_BODY[11][1]+STATIC_BODY[12][1])/2)
#         t_n, s_n = P((mid[0], mid[1]-NECK_OFFSET)), PF((ff[1,0], ff[1,1]))
#         dx, dy = t_n[0]-s_n[0], t_n[1]-s_n[1]
#         def d_f(c_set, col, th):
#             for c in c_set:
#                 p1, p2 = PF(ff[c[0]]), PF(ff[c[1]])
#                 cv2.line(img, (p1[0]+dx, p1[1]+dy), (p2[0]+dx, p2[1]+dy), col, th, cv2.LINE_AA)
#         d_f(mp_fmc.FACEMESH_TESSELATION, (60,60,60), 1)
#         d_f(mp_fmc.FACEMESH_FACE_OVAL, (255,255,255), 2)
#         d_f(mp_fmc.FACEMESH_LIPS, (255,255,255), 2)
#         d_f(mp_fmc.FACEMESH_RIGHT_EYE, (0,0,255), 2)
#         d_f(mp_fmc.FACEMESH_LEFT_EYE, (0,255,0), 2)
#         try: d_f(mp_fmc.FACEMESH_RIGHT_IRIS, (0,255,255), 2); d_f(mp_fmc.FACEMESH_LEFT_IRIS, (255,255,0), 2)
#         except: d_f(mp_fmc.FACEMESH_IRISES, (0,255,255), 2)
#         for idx, col in [(468, (0,255,255)), (473, (255,255,0))]:
#             p = PF(ff[idx]); cv2.circle(img, (p[0]+dx, p[1]+dy), 3, col, -1)
#         vw.write(img)

#     vw.release()
#     print(f"\n--- SUCCESS: Video Rendered at {target_fps} FPS ---")
#     subprocess.run(['explorer', '/select,', os.path.normpath(out_name)])

# ===================== 4. RENDERER WITH SLOW-MO LOGIC =====================

def run_pipeline(eng, gloss_str, exp_str):
    out_name = os.path.join(BASE_PATH, 'FINAL_SIGN_VIDEO_1.mp4')
    with open(JSON_PATH) as f:
        raw_map = json.load(f)
        gloss_map = {k.upper(): v for k, v in raw_map.items()}
    
    raw_words = re.sub(r'[^\w\s]', '', gloss_str).split()
    raw_exps = exp_str.split()
    
    # --- Paired Filtering ---
    final_words, final_exps = [], []
    min_len = min(len(raw_words), len(raw_exps))
    for i in range(min_len):
        word, expr = raw_words[i].upper(), raw_exps[i].lower()
        if word in gloss_map:
            final_words.append(word)
            final_exps.append(expr)
    
    if not final_words:
        print("--- NO MATCHING SIGNS FOUND ---")
        return

    # --- SPEED CAPTURE ---
    print("-" * 30)
    print(f"RENDERING: {' '.join(final_words)}")
    speed_choice = input("Enter Speed Mode - [N]ormal (3s length) or [S]low (6s length): ").lower()
    
    # Critical Fix: We always render the full frame set, 
    # but change the metadata FPS to stretch the time.
    render_fps = 60
    playback_fps = 30 if speed_choice == 's' else 60
    print(f"Rendering frames at {render_fps}fps logic. Playback set to {playback_fps}fps.")
    print("-" * 30)

    # Frame Synthesis
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

        face_src = get_face_npy(ex)
        if full_body:
            b_br = bridge(full_body[-1][-1], body_data[0]); full_body.append(b_br)
            f_a, f_b = full_face[-1][-1], (face_src[0] if face_src is not None else full_face[0][0])
            f_bridge = [(1-t/(len(b_br)-1))*f_a + (t/(len(b_br)-1))*f_b for t in range(len(b_br))]
            full_face.append(np.array(f_bridge))

        full_body.append(body_data)
        full_face.append(sync_face_to_duration(face_src, len(body_data)) if face_src is not None else np.repeat([full_face[-1][-1] if full_face else get_face_npy('static')[0]], len(body_data), axis=0))

    final_b, final_f = np.concatenate(full_body), np.concatenate(full_face)
    
    # Initialize VideoWriter with Playback FPS
    vw = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), playback_fps, (W, H))
    
    ref = final_b[0]
    def dist(a,b): return np.linalg.norm(np.array(a)-np.array(b))
    L_UP_L, L_LO_L = dist((ref[44],ref[45]),(ref[52],ref[53])), dist((ref[52],ref[53]),(ref[60],ref[61]))
    L_UP_R, L_LO_R = dist((ref[48],ref[49]),(ref[56],ref[57])), dist((ref[56],ref[57]),(ref[64],ref[65]))
    F_SC = SCALE * FACE_SCALE_MULT
    def PF(pt): return (int(pt[0]*F_SC)+OFFSET_X, int((pt[1]+CAMERA_Y_SHIFT)*F_SC)+OFFSET_Y)

    print("--- Generating Frames ---")
    for i in tqdm(range(len(final_b))):
        img = np.zeros((H, W, 3), np.uint8)
        fb, ff = final_b[i], final_f[i]
        for a,b in [(11,12),(11,23),(12,24),(23,24)]: cv2.line(img, P(STATIC_BODY[a]), P(STATIC_BODY[b]), (255,255,255), 3)
        dir_le = (np.array([fb[52]-fb[44], fb[53]-fb[45]])) / (np.linalg.norm([fb[52]-fb[44], fb[53]-fb[45]])+1e-6)
        dir_re = (np.array([fb[56]-fb[48], fb[57]-fb[49]])) / (np.linalg.norm([fb[56]-fb[48], fb[57]-fb[49]])+1e-6)
        j13, j14 = (STATIC_BODY[12][0]+dir_le[0]*L_UP_L, STATIC_BODY[12][1]+dir_le[1]*L_UP_L), (STATIC_BODY[11][0]+dir_re[0]*L_UP_R, STATIC_BODY[11][1]+dir_re[1]*L_UP_R)
        dir_lw = (np.array([fb[60]-fb[52], fb[61]-fb[53]])) / (np.linalg.norm([fb[60]-fb[52], fb[61]-fb[53]])+1e-6)
        dir_rw = (np.array([fb[64]-fb[56], fb[65]-fb[57]])) / (np.linalg.norm([fb[64]-fb[56], fb[65]-fb[57]])+1e-6)
        j15, j16 = (j13[0]+dir_lw[0]*L_LO_L, j13[1]+dir_lw[1]*L_LO_L), (j14[0]+dir_rw[0]*L_LO_R, j14[1]+dir_rw[1]*L_LO_R)
        for s,e in [(P(STATIC_BODY[12]),P(j13)),(P(j13),P(j15)),(P(STATIC_BODY[11]),P(j14)),(P(j14),P(j16))]: cv2.line(img, s, e, (255,255,255), 2)
        for side, w_idx, s_idx in [('l',j15,132),('r',j16,195)]:
            wr_p, root_p = P(w_idx), P((fb[s_idx], fb[s_idx+1]))
            pts = {k: (wr_p[0] + P((fb[s_idx+k*3], fb[s_idx+k*3+1]))[0] - root_p[0], wr_p[1] + P((fb[s_idx+k*3], fb[s_idx+k*3+1]))[1] - root_p[1]) for k in range(21)}
            for c in HAND_CONNECTIONS: cv2.line(img, pts[c[0]], pts[c[1]], (0,255,0) if side=='l' else (0,0,255), 2)
        mid = ((STATIC_BODY[11][0]+STATIC_BODY[12][0])/2, (STATIC_BODY[11][1]+STATIC_BODY[12][1])/2)
        t_n, s_n = P((mid[0], mid[1]-NECK_OFFSET)), PF((ff[1,0], ff[1,1]))
        dx, dy = t_n[0]-s_n[0], t_n[1]-s_n[1]
        def d_f(c_set, col, th):
            for c in c_set:
                p1, p2 = PF(ff[c[0]]), PF(ff[c[1]])
                cv2.line(img, (p1[0]+dx, p1[1]+dy), (p2[0]+dx, p2[1]+dy), col, th, cv2.LINE_AA)
        d_f(mp_fmc.FACEMESH_TESSELATION, (60,60,60), 1)
        d_f(mp_fmc.FACEMESH_FACE_OVAL, (255,255,255), 2)
        d_f(mp_fmc.FACEMESH_LIPS, (255,255,255), 2)
        d_f(mp_fmc.FACEMESH_RIGHT_EYE, (0,0,255), 2)
        d_f(mp_fmc.FACEMESH_LEFT_EYE, (0,255,0), 2)
        try: d_f(mp_fmc.FACEMESH_RIGHT_IRIS, (0,255,255), 2); d_f(mp_fmc.FACEMESH_LEFT_IRIS, (255,255,0), 2)
        except: d_f(mp_fmc.FACEMESH_IRISES, (0,255,255), 2)
        for idx, col in [(468, (0,255,255)), (473, (255,255,0))]:
            p = PF(ff[idx]); cv2.circle(img, (p[0]+dx, p[1]+dy), 3, col, -1)
        vw.write(img)
    vw.release()
    print(f"\n--- SUCCESS: Final Video Length Modified ---")
    # subprocess.run(['explorer', '/select,', os.path.normpath(out_name)])
    subprocess.run(['open', '-R', os.path.normpath(out_name)])

if __name__ == "__main__":
    eng = get_input_text()
    if eng:
        g, ex = get_isl_from_ollama(eng)
        run_pipeline(eng, g, ex)