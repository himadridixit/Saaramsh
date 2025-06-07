import numpy as np
import json

def collapse_spikes(spike_times, window=2.0):
    if len(spike_times) == 0:
        return np.array([])

    sorted_spikes = np.sort(spike_times)
    collapsed = [sorted_spikes[0]]
    
    for t in sorted_spikes[1:]:
        if t - collapsed[-1] > window:
            collapsed.append(t)
    
    return np.array(collapsed)

def evaluate_matches(true_positives, detected_times, tolerance=0.2):
    """
    Evaluate how many detected times match ground‐truth true positives,
    then print a simple confusion matrix (TP, FP, FN, TN=0).

    Args:
        true_positives: 1D numpy array (sorted) of ground‐truth timestamps (in seconds)
        detected_times: 1D numpy array (sorted) of detected timestamps (in seconds)
        tolerance: Time window (in seconds) to consider a match

    Prints:
        Confusion Matrix:
          TP: X | FP: Y
          FN: Z | TN: 0
    """
    # Make copies so we can remove matched items:
    tp_matches = []
    fp_detections = []
    fn_true_positives = true_positives.copy().tolist()  # will remove matched ones
    detected_times = collapse_spikes(detected_times, window=5)

    i = 0
    j = 0
    n_true = len(true_positives)
    n_detected = len(detected_times)

    # Two‐pointer sweep over sorted lists:
    while i < n_true and j < n_detected:
        t_true = true_positives[i]
        t_det  = detected_times[j]

        if abs(t_det - t_true) <= tolerance:
            # Found a match → True Positive
            tp_matches.append((t_true, t_det))
            fn_true_positives.remove(t_true)  # mark this true as “found”
            i += 1
            j += 1
        elif t_det < t_true - tolerance:
            # Detected‐time came too early → no matching true for this detection → FP
            fp_detections.append(t_det)
            j += 1
        else:
            # t_det > t_true + tolerance → move True pointer forward (this true is unmatched so far)
            i += 1

    # Any remaining detections are extra false positives:
    fp_detections.extend(detected_times[j:])

    # Any remaining true_positives that were never matched are false negatives:
    # (fn_true_positives was initially all true_positives and had each matched true removed above)
    fn_list = fn_true_positives

    # Now compute counts:
    tp = len(tp_matches)
    fp = len(fp_detections)
    fn = len(fn_list)
    tn = 0  # we ignore TN in this setting, so set it to 0

    # Print confusion matrix in requested format:
    print("  Confusion Matrix:")
    print(f"    TP: {tp} | FP: {fp}")
    print(f"    FN: {fn} | TN: {tn}")



with open("output/TP.json", "r") as f:
    true_positives = np.array(json.load(f)["true_positives"])
with open("output/detected_events.json", "r") as f:
    detected_times = np.array(json.load(f)["detected_events"])

print("CNN")
evaluate_matches(true_positives, detected_times, tolerance=0.5)
