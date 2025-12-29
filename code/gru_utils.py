# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import os

# class GRUTransitionManager:
#     def __init__(self, model_path):
#         """Initializes and loads the trained GRU model."""
#         if not os.path.exists(model_path):
#             print(f"ERROR: Model not found at {model_path}")
#             self.model = None
#         else:
#             self.model = load_model(model_path)
        
#         self.max_len = 150
#         self.feature_size = 258

#     def generate_bridge(self, start_sequence, target_frame, bridge_length=12):
#         """
#         Uses the GRU to predict smooth transition frames.
#         - start_sequence: Last few frames of previous Sign (shape: [N, 258])
#         - target_frame: First frame of next Sign (shape: [258,])
#         """
#         if self.model is None:
#             # Fallback to simple interpolation if model is missing
#             return np.array([((1-t/bridge_length)*start_sequence[-1] + (t/bridge_length)*target_frame) 
#                              for t in range(bridge_length)])

#         current_seq = list(start_sequence)
#         predicted_bridge = []

#         for i in range(bridge_length):
#             # 1. Prepare input sequence (pad/slice to match MAX_LEN)
#             input_data = np.array(current_seq)
#             if len(input_data) > self.max_len:
#                 input_data = input_data[-self.max_len:]
            
#             pad_width = ((0, self.max_len - len(input_data)), (0, 0))
#             padded_input = np.pad(input_data, pad_width, mode='constant').reshape(1, self.max_len, self.feature_size)

#             # 2. Predict next frame
#             prediction = self.model.predict(padded_input, verbose=0)
#             # Use the prediction corresponding to the last real timestep
#             next_frame_pred = prediction[0, len(current_seq)-1, :]
            
#             # 3. Blending: Gradually shift from GRU prediction to the exact target coordinate
#             alpha = (i + 1) / bridge_length
#             final_frame = (1 - alpha) * next_frame_pred + (alpha * target_frame)
            
#             predicted_bridge.append(final_frame)
#             current_seq.append(final_frame)

#         return np.array(predicted_bridge)