import torch
import numpy as np
import librosa
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import json
from train import CricketNet
from tqdm import tqdm

# Configuration (match training parameters)
SR = 44100
HOP_LENGTH = 256
N_MELS = 128
WINDOW_SIZE = 128  # 128x128 spectrogram window
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.7  # Confidence threshold for detection
MIN_EVENT_GAP = 0.5  # Minimum gap between distinct events (seconds)
PEAK_DETECTION_WINDOW = 15  # Frames to look for max probability

# # Load trained model
# def load_model(model_path):
#     model = CricketNet()  # Your model class
#     model.load_state_dict(torch.load(model_path))
#     model.to(DEVICE)
#     model.eval()
#     return model

def load_model_for_inference(model_path, device="cuda"):
    model = CricketNet()
    best_ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# Generate full spectrogram
def audio_to_spectrogram(audio_path):
    y, _ = librosa.load(audio_path, sr=SR)
    S = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP_LENGTH, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=np.max)

# Sliding window prediction
def predict_spectrogram(model, full_dB):
    predictions = []
    n_frames = full_dB.shape[1]
    
    # Precompute time per frame
    frame_duration = HOP_LENGTH / SR
    
    for start_frame in tqdm(range(0, n_frames - WINDOW_SIZE + 1)):
        # Extract window
        patch = full_dB[:, start_frame:start_frame + WINDOW_SIZE]
        
        # Pad if needed at end
        if patch.shape[1] < WINDOW_SIZE:
            patch = np.pad(patch, ((0, 0), (0, WINDOW_SIZE - patch.shape[1])))
        
        # Convert to tensor (match training preprocessing)
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)  # [0,1]
        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        patch_tensor = (patch_tensor - 0.5) / 0.5  # Normalize to [-1,1]
        patch_tensor = patch_tensor.to(DEVICE)
        
        # Predict
        with torch.no_grad():
            output = model(patch_tensor)
            proba = torch.sigmoid(output).item()
        
        # Time position: center of the window + alignment offset
        time_pos = (start_frame + 42) * frame_duration
        predictions.append((time_pos, proba))
    
    return np.array(predictions)

# Collapse predictions into distinct events
def detect_events(predictions, min_gap=MIN_EVENT_GAP):
    times, probs = predictions[:, 0], predictions[:, 1]
    
    # 1. Find peaks in probability curve
    peaks, _ = find_peaks(probs, height=THRESHOLD, 
                          distance=int(MIN_EVENT_GAP/(HOP_LENGTH/SR)))
    
    # 2. Refine event times using window around peaks
    events = []
    for peak_idx in peaks:
        # Look for max probability in neighborhood
        start = max(0, peak_idx - PEAK_DETECTION_WINDOW)
        end = min(len(probs), peak_idx + PEAK_DETECTION_WINDOW)
        local_max_idx = start + np.argmax(probs[start:end])
        
        events.append(times[local_max_idx])
    
    return events

# Compare with ground truth
def evaluate_events(detected_events, ground_truth, tolerance=0.3):
    true_positives = 0
    false_positives = []
    matched_gt = []
    
    for event in detected_events:
        # Find closest ground truth within tolerance
        closest = min(ground_truth, key=lambda x: abs(x - event))
        if abs(closest - event) <= tolerance:
            true_positives += 1
            matched_gt.append(closest)
        else:
            false_positives.append(event)
    
    # Find missed ground truth
    false_negatives = [gt for gt in ground_truth if gt not in matched_gt]
    
    return true_positives, false_positives, false_negatives

# Visualization
def plot_detection_results(full_dB, predictions, events, ground_truth):
    times, probs = predictions[:, 0], predictions[:, 1]
    
    plt.figure(figsize=(15, 10))
    
    # Spectrogram
    plt.subplot(2, 1, 1)
    librosa.display.specshow(full_dB, sr=SR, hop_length=HOP_LENGTH, 
                            x_axis='time', y_axis='mel', cmap='viridis')
    plt.title('Spectrogram with Detections')
    
    # Mark ground truth
    for gt in ground_truth:
        plt.axvline(x=gt, color='lime', linestyle='--', alpha=0.7, linewidth=2)
    
    # Mark detected events
    for event in events:
        plt.axvline(x=event, color='red', linewidth=1.5)
    
    # Probability curve
    plt.subplot(2, 1, 2)
    plt.plot(times, probs, label='Prediction Confidence')
    plt.axhline(y=THRESHOLD, color='r', linestyle='--', label='Threshold')
    
    # Mark detected events
    for event in events:
        plt.axvline(x=event, color='red', linestyle='-', alpha=0.7)
    
    plt.title('Detection Probability Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig('detection_results.png')
    plt.show()

# Main processing pipeline
def detect_shots(audio_path, model_path, ground_truth_timestamps):
    # 1. Load model and audio
    model = load_model_for_inference(model_path, device='cpu')
    print("model loaded")
    full_dB = audio_to_spectrogram(audio_path)
    print("converted to spectrogram")
    
    # 2. Run sliding window prediction
    predictions = predict_spectrogram(model, full_dB)
    print("got predictions")

    # 3. Collapse predictions into distinct events
    detected_events = detect_events(predictions)
    print("detected events")
    
    # 4. Evaluate against ground truth
    tp, fp, fn = evaluate_events(detected_events, ground_truth_timestamps)
    print("evaluated")
    
    print("\nDetection Results:")
    print(f"Detected events: {len(detected_events)}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {len(fp)}")
    print(f"False Negatives: {len(fn)}")
    print(f"Precision: {tp/(tp+len(fp)):.2f}")
    print(f"Recall: {tp/(tp+len(fn)):.2f}")
    
    # 5. Visualize results
    plot_detection_results(full_dB, predictions, detected_events, ground_truth_timestamps)
    
    return detected_events

# Example usage
if __name__ == "__main__":
    # Replace with your actual paths and data
    audio_path = "output/audio.wav"
    model_path = "C:/Users/HP/OneDrive/Documents/reel-maker/checkpoints/checkpoint_5.pth"
    with open("output/TP.json") as f:
        ground_truth = json.load(f)["true_positives"]
    
    detected_events = detect_shots(audio_path, model_path, ground_truth)
    
    # Save detected events
    with open("detected_events.json", "w") as f:
        json.dump({"detected_events": detected_events}, f)