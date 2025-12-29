import os
import mediapipe as mp

# ===================== PATH CONFIGURATION =====================
BASE_PATH = r"/Users/sj/Documents/sound2sign"
FEATURES_DIR = os.path.join(BASE_PATH, 'processed_npy_flattened')
JSON_PATH = os.path.join(BASE_PATH, 'gloss_mapping_f8.json')
FACE_NPY_DIR = os.path.join(BASE_PATH, 'facial_data_npy_full')
OLLAMA_URL = "http://localhost:11434/api/generate"
OUTPUT_VIDEO_PATH = os.path.join(BASE_PATH, 'output_videos/FINAL_SIGN_VIDEO_7.mp4')

# ===================== RENDERING CONSTANTS =====================
W, H = 800, 800
SCALE = 350
FACE_SCALE_MULT = 1.5 
OFFSET_X, OFFSET_Y = 400, 400
CAMERA_Y_SHIFT, NECK_OFFSET = -0.25, 0.45
F_TRANSITION_LEN = 8

STATIC_BODY = {11: (-0.45, 0.0), 12: (0.45, 0.0), 23: (-0.35, 2.038), 24: (0.35, 2.038)}
HAND_CONNECTIONS = mp.solutions.holistic.HAND_CONNECTIONS
FACEMESH_CONNS = mp.solutions.face_mesh_connections