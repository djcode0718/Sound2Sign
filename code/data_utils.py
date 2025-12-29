import os
import random
import numpy as np
from config import FEATURES_DIR, FACE_NPY_DIR, F_TRANSITION_LEN

def bridge(a, b):
    d = np.linalg.norm(a[195:197] - b[195:197])
    n = int(np.clip(d * 35, 8, 20))
    out = [((1-(0.5-0.5*np.cos(np.pi*(i/(n-1)))))*a + (0.5-0.5*np.cos(np.pi*(i/(n-1))))*b) for i in range(n)]
    return np.array(out)

def get_face_npy(expression):
    """
    Tries to load requested expression. 
    If missing, defaults to 'static'.
    If 'static' is missing, returns zeroed array.
    """
    expr_clean = expression.strip().lower()
    folder_path = os.path.join(FACE_NPY_DIR, expr_clean)
    
    # 1. Try to find the requested expression
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        if files: 
            return np.load(os.path.join(folder_path, random.choice(files)))
    
    # 2. Fallback to 'static' if expression not found
    if expr_clean != 'static':
        print(f"--- Notice: '{expr_clean}' not found, falling back to 'static' ---")
        return get_face_npy('static')
    
    # 3. Absolute Fallback (if even static folder is empty/missing)
    print("--- Warning: No facial data found at all. Using dummy zeros. ---")
    return np.zeros((30, 478, 3)) 

def sync_face_to_duration(face_data, target_len):
    # Standard synchronization logic from original code
    out = []
    while len(out) < target_len:
        to_add = min(len(face_data), target_len - len(out))
        out.extend(face_data[:to_add])
        if len(out) < target_len:
            a, b = face_data[-1], face_data[0]
            for t in range(min(F_TRANSITION_LEN, target_len - len(out))):
                out.append((1-(t/F_TRANSITION_LEN))*a + (t/F_TRANSITION_LEN)*b)
    return np.array(out[:target_len])