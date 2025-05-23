import librosa
import numpy as np
import json
from moviepy.editor import VideoFileClip
import os

# === CONFIG ===
VIDEO_PATH = "videos/VID_20250517_184414279.mp4"
AUDIO_PATH = "output/audio.wav"
WAVEFORM_JSON = "output/waveform.json"
SPIKES_JSON = "output/spikes.json"
DOWNSAMPLE_POINTS = 200000  # for display
THRESHOLD = 0.5

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# === STEP 1: Extract audio ===
print("🔊 Extracting audio...")
video = VideoFileClip(VIDEO_PATH)
video.audio.write_audiofile(AUDIO_PATH, logger=None)

# === STEP 2: Load audio ===
print("📈 Loading audio into memory...")
y, sr = librosa.load(AUDIO_PATH, sr=None)
duration = librosa.get_duration(y=y, sr=sr)

# === STEP 3: Downsample for plotting ===
print("📉 Downsampling waveform...")
factor = max(len(y) // DOWNSAMPLE_POINTS, 1)
y_ds = y[::factor]
time_ds = np.linspace(0, duration, num=len(y_ds))

# === STEP 4: Spike detection ===
print("⚠️ Detecting spikes...")
spikes = np.where(np.abs(y) > THRESHOLD)[0]
spike_times = spikes / sr

# === STEP 5: Save output ===
print("💾 Saving preprocessed data...")
with open(WAVEFORM_JSON, "w") as f:
    json.dump({"waveform": y_ds.tolist(), "time": time_ds.tolist()}, f)

with open(SPIKES_JSON, "w") as f:
    json.dump({"spike_times": spike_times.tolist()}, f)

print("✅ Done!")
