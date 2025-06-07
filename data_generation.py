import os
import json
import numpy as np
import librosa
import cv2
import random
from sklearn.model_selection import train_test_split

# align shot
def align_shot_peak(patch):
    """Center the shot sound at 1/3 of the window"""
    energy = np.mean(patch, axis=0)
    peak_idx = np.argmax(energy)
    target_idx = 42  # 128/3 ≈ 42 frames (1/3 point)
    shift = target_idx - peak_idx
    return np.roll(patch, shift, axis=1)

# positive sample selection
def extract_positive_samples(full_dB, shot_timestamps, sr=44100, hop_length=256):
    positive_patches = []
    shot_frames = [int(t * sr / hop_length) for t in shot_timestamps]
    
    for frame in shot_frames:
        start = max(0, frame - 50)  # 50 frames before shot
        end = start + 128
        
        # Handle end-of-audio case
        if end > full_dB.shape[1]:
            start = full_dB.shape[1] - 128
            end = full_dB.shape[1]
        
        patch = full_dB[:, start:end]
        
        # Only align if patch contains clear peak
        if np.ptp(patch) > 15:  # Minimum dB range threshold
            patch = align_shot_peak(patch)
        
        positive_patches.append(patch)
    
    return positive_patches

# negative sample selection
def extract_negative_samples(full_dB, shot_timestamps, num_samples):
    """Extract background noise samples avoiding shot regions"""
    negative_patches = []
    shot_frames = [int(t * 44100 / 256) for t in shot_timestamps]
    
    # Create exclusion zones (100 frames before and after shots)
    exclusion_zones = []
    for frame in shot_frames:
        exclusion_zones.extend(range(max(0, frame - 100), min(full_dB.shape[1], frame + 100)))
    
    # Generate candidate positions
    valid_positions = [i for i in range(0, full_dB.shape[1] - 128) 
                      if i not in exclusion_zones]
    
    # Randomly sample positions
    selected_positions = random.sample(valid_positions, min(num_samples, len(valid_positions)))
    
    for start in selected_positions:
        patch = full_dB[:, start:start+128]
        negative_patches.append(patch)
    
    return negative_patches

# creating more positive samples
def augment_positive(patch):
    augmented = []
    
    # Time stretching (using image resizing)
    for scale in [0.9, 1.0, 1.1]:
        # Scale along time axis only
        scaled = cv2.resize(patch, (int(128*scale), 128), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Pad/crop to 128x128
        if scaled.shape[1] < 128:
            padded = np.zeros((128, 128))
            padded[:, :scaled.shape[1]] = scaled
            augmented.append(padded)
        else:
            augmented.append(scaled[:, :128])
    
    # Frequency masking
    for _ in range(2):
        masked = patch.copy()
        # Focus masking on mid-frequencies (where bat sounds occur)
        freq_start = random.randint(40, 80)  # 2-6kHz range
        masked[freq_start:freq_start+6, :] = 0
        augmented.append(masked)
    
        # Time masking (random segments)
    for _ in range(2):
        masked = patch.copy()
        # Avoid masking the critical shot moment (frame 42)
        safe_start = random.choice([0, 60])  # Either beginning or end
        time_start = safe_start + random.randint(0, 20)
        masked[:, time_start:time_start+8] = 0
        augmented.append(masked)

    return augmented

def create_directory_structure(base_path="dataset"):
    """Create standardized directory structure"""
    paths = {
        'train': os.path.join(base_path, "train"),
        'val': os.path.join(base_path, "val"),
        'test': os.path.join(base_path, "test")
    }
    
    for split in paths.values():
        os.makedirs(os.path.join(split, "positive"), exist_ok=True)
        os.makedirs(os.path.join(split, "negative"), exist_ok=True)
    
    return paths

def save_as_image(patches, label, output_dir):
    """Save spectrogram patches as PNG images"""
    for i, patch in enumerate(patches):
        # Normalize to 0-255 range
        norm_patch = 255 * ((patch - patch.min()) / (patch.max() - patch.min() + 1e-8))
        img = norm_patch.astype(np.uint8)
        
        # Flip vertically (matplotlib vs. image coordinate convention)
        img = np.flipud(img)
        
        # Save as PNG
        cv2.imwrite(os.path.join(output_dir, f"{label}_{i}.png"), img)

# 1. Load precomputed spectrogram
full_dB = np.load('output/full_spectrogram.npy')  # Precomputed from your code

# 2. Load verified shot timestamps
with open('output/CNN_spikes.json') as f:
    shot_timestamps = json.load(f)['collapsed_spikes']

# 3. Extract base samples
positives = extract_positive_samples(full_dB, shot_timestamps)
negatives = extract_negative_samples(full_dB, shot_timestamps, len(positives)*6)  # 6:1 ratio

# 4. Augment positive samples
augmented_positives = []
for patch in positives:
    augmented_positives.extend(augment_positive(patch))

base_paths = create_directory_structure()

# Split dataset
print("Splitting dataset...")
# Split positives
pos_train, pos_temp = train_test_split(augmented_positives, test_size=0.15 + 0.15, 
                                        random_state=42)
pos_val, pos_test = train_test_split(pos_temp, test_size=0.15/(0.15+0.15), 
                                    random_state=42)

# Split negatives
neg_train, neg_temp = train_test_split(negatives, test_size=0.15 + 0.15, 
                                        random_state=42)
neg_val, neg_test = train_test_split(neg_temp, test_size=0.15/(0.15+0.15), 
                                    random_state=42)

# Save datasets
print("Saving datasets...")
# Training set
save_as_image(pos_train, "positive", base_paths['train'] + "/positive")
save_as_image(neg_train, "negative", base_paths['train'] + "/negative")

# Validation set
save_as_image(pos_val, "positive", base_paths['val'] + "/positive")
save_as_image(neg_val, "negative", base_paths['val'] + "/negative")

# Test set
save_as_image(pos_test, "positive", base_paths['test'] + "/positive")
save_as_image(neg_test, "negative", base_paths['test'] + "/negative")

print("Dataset generation complete!")
print(f"Final counts:")
print(f"  - Training: {len(pos_train)} pos, {len(neg_train)} neg")
print(f"  - Validation: {len(pos_val)} pos, {len(neg_val)} neg")
print(f"  - Testing: {len(pos_test)} pos, {len(neg_test)} neg")



