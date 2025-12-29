import os
import speech_recognition as sr
from faster_whisper import WhisperModel

class WhisperTranscriber:
    def __init__(self, model_size="base"):
        print("--- Initializing Faster-Whisper ---")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe_audio_file(self, audio_file_path: str) -> str:
        if not os.path.exists(audio_file_path):
            return ""
        segments, _ = self.model.transcribe(audio_file_path, beam_size=5)
        return "".join(segment.text for segment in segments).strip()

transcriber = WhisperTranscriber(model_size="base")

def get_input_text():
    mode = input("\n[V]oice Input or [T]ype English? ").lower()
    if mode == 'v':
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print("--- Calibrating Mic... ---")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print("--- Speak Now ---")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            temp_path = "live_audio.wav"
            with open(temp_path, "wb") as f: 
                f.write(audio.get_wav_data())
             
            print("--- Transcribing (Faster-Whisper) ---")
            text = transcriber.transcribe_audio_file(temp_path)
            
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