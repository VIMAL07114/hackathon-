import sounddevice as sd
import numpy as np
import joblib
import os

model_path = "voice_model.pkl"

if not os.path.exists(model_path):
    print("Model file not found ❌")
    exit()

model = joblib.load(model_path)

duration = 3
sample_rate = 44100

print("🎤 Listening... Speak now!")

audio = sd.rec(int(duration * sample_rate),
               samplerate=sample_rate,
               channels=1,
               dtype='float32')

sd.wait()

audio = audio.flatten()

feature = np.mean(np.abs(audio))

prediction = model.predict([[feature]])

if prediction[0] == 1:
    print("🚨 Emergency Voice Detected!")
else:
    print("✅ Normal Voice")