import os
import json
import numpy as np
import subprocess

# SETTINGS
VIDEO_FILE = "assets/VID_20250517_184414279.mp4"
OUTPUT_DIR = "mini_clips"
TEMP_CLIPS_DIR = os.path.join(OUTPUT_DIR, "temp")
OUTPUT_REEL = os.path.join(OUTPUT_DIR, "highlight_reel_ffmpeg_mini.mp4")
PRE_TIME = 0.01
POST_TIME = 0.45

# Collapse spikes
def collapse_spikes(spike_times, window=2.0):
    if len(spike_times) == 0:
        return np.array([])
    sorted_spikes = np.sort(spike_times)
    collapsed = [sorted_spikes[0]]
    for t in sorted_spikes[1:]:
        if t - collapsed[-1] > window:
            collapsed.append(t)
    return np.array(collapsed)

# Load spikes
with open("output/spikes.json", "r") as f:
    spike_times = np.array(json.load(f)["spike_times"])

collapsed_spikes = collapse_spikes(spike_times)

with open("output/collapsed_spikes.json", "w") as f:
    json.dump({"collapsed_spikes": collapsed_spikes.tolist()}, f)
breakpoint()
# Prepare directories
os.makedirs(TEMP_CLIPS_DIR, exist_ok=True)

# Step 1: Extract clips using ffmpeg
clip_paths = []
for i, t in enumerate(collapsed_spikes):
    start = max(0, t - PRE_TIME)
    duration = PRE_TIME + POST_TIME
    output_path = os.path.join(TEMP_CLIPS_DIR, f"clip_{i+1:03d}.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", VIDEO_FILE,
        "-t", str(duration),
        "-c", "copy",  # no re-encoding = fast + preserves aspect
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    clip_paths.append(output_path)
breakpoint()
# Step 2: Concatenate all clips
# Write list file for ffmpeg
concat_list_path = os.path.join(TEMP_CLIPS_DIR, "concat_list.txt")
with open(concat_list_path, "w") as f:
    for clip in clip_paths:
        abs_path = os.path.abspath(clip).replace('\\', '/')  # Use forward slashes
        f.write(f"file '{abs_path}'\n")

# Step 3: Run ffmpeg concat
cmd_concat = [
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_list_path,
    "-c:v", "libx264",
    "-c:a", "aac",
    "-preset", "ultrafast",     # faster encoding
    "-crf", "23",               # good quality (lower = better)
    OUTPUT_REEL
]

subprocess.run(cmd_concat)

print(f"✅ Highlight reel saved to {OUTPUT_REEL}")
