import os
import numpy as np
import subprocess
import json

# --- SETTINGS ---
VIDEO_FILE = "assets/VID_20250517_184414279.mp4"
OUTPUT_DIR = "mini_clips"
TEMP_CLIPS_DIR = os.path.join(OUTPUT_DIR, "temp")
OUTPUT_REEL = os.path.join(OUTPUT_DIR, "highlight_reel_spectro_match.mp4")
PRE_TIME = 0.1  # seconds before match
POST_TIME = 1  # seconds after match
COLLAPSE_WINDOW = 2.0  # seconds to collapse nearby matches

# --- INPUT: from matching step ---
# Make sure this file exists from your matching script
MATCH_FILE = "output/match_times.json"

with open(MATCH_FILE, "r") as f:
    match_times = np.array(json.load(f)["match_times"])  # float timestamps

# --- Collapse nearby matches ---
def collapse_matches(times, window=2.0):
    if len(times) == 0:
        return np.array([])
    times = np.sort(times)
    collapsed = [times[0]]
    for t in times[1:]:
        if t - collapsed[-1] > window:
            collapsed.append(t)
    return np.array(collapsed)

collapsed_times = collapse_matches(match_times, window=COLLAPSE_WINDOW)

# --- Prepare Directories ---
os.makedirs(TEMP_CLIPS_DIR, exist_ok=True)

# --- Step 1: Extract clips with ffmpeg ---
clip_paths = []
for i, t in enumerate(collapsed_times):
    start = max(0, t - PRE_TIME)
    duration = PRE_TIME + POST_TIME
    output_path = os.path.join(TEMP_CLIPS_DIR, f"clip_{i+1:03d}.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", VIDEO_FILE,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    clip_paths.append(output_path)

# --- Step 2: Write concat list ---
concat_list_path = os.path.join(TEMP_CLIPS_DIR, "concat_list.txt")
with open(concat_list_path, "w") as f:
    for clip in clip_paths:
        abs_path = os.path.abspath(clip).replace("\\", "/")
        f.write(f"file '{abs_path}'\n")

# --- Step 3: Concatenate clips ---
cmd_concat = [
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_list_path,
    "-c:v", "libx264",
    "-c:a", "aac",
    "-preset", "ultrafast",
    "-crf", "23",
    OUTPUT_REEL
]
subprocess.run(cmd_concat)

print(f"✅ Highlight reel saved to {OUTPUT_REEL}")
