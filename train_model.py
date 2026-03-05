import numpy as np
import os
import librosa
from sklearn.ensemble import RandomForestClassifier
import joblib

X = []
y = []

dataset = os.getcwd()

for folder in ["emergency", "normal"]:

    folder_path = os.path.join(dataset, folder)

    if not os.path.exists(folder_path):
        print(f"{folder} folder not found ❌")
        continue

    for file in os.listdir(folder_path):

        if file.lower().endswith((".wav", ".mp3")):

            filepath = os.path.join(folder_path, file)

            try:
                audio, sr = librosa.load(filepath, sr=None)

                feature = np.mean(np.abs(audio))

                X.append([feature])

                if folder == "emergency":
                    y.append(1)
                else:
                    y.append(0)

                print(f"Loaded: {file}")

            except Exception as e:
                print(f"Skipped {file} : {e}")

if len(X) == 0:
    print("No audio files found ❌")
    exit()

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "voice_model.pkl")

print("✅ Model trained successfully!")