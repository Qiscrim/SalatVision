# ============================================================
# evaluate.py — SalatVision Posture Classification Evaluator
#
# Usage:
#   python evaluate.py --session sessions/Fajr_20260519_123456.json
#                      --ground  ground_truth.csv
#
# Output:
#   - Confusion matrix (printed + saved as PNG)
#   - Per-class precision, recall, F1
#   - Overall accuracy
# ============================================================

import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

POSTURES = ["QIYAM", "RUKU", "SUJUD", "TASHAHHUD"]


# ── Load data ────────────────────────────────────────────────

def load_session(path):
    """Load frame_log from session JSON. Returns list of {time, predicted}."""
    with open(path) as f:
        data = json.load(f)
    frame_log = data.get("frame_log", [])
    if not frame_log:
        raise ValueError(
            "No frame_log found in session file.\n"
            "Make sure you added per-frame logging to main.py first."
        )
    return frame_log


def load_ground_truth(path):
    """Load ground_truth.csv. Returns list of {time_sec, true_posture}."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            posture = row["true_posture"].strip().upper()
            if posture in POSTURES:
                rows.append({
                    "time_sec": float(row["time_sec"]),
                    "true_posture": posture
                })
    if not rows:
        raise ValueError("ground_truth.csv is empty or has no valid postures.")
    return rows


# ── Align predictions to ground truth ────────────────────────

def align(frame_log, ground_truth):
    """
    For each ground truth timestamp, find the closest predicted label
    within a 1-second tolerance window.

    Returns two lists: y_true, y_pred (same length, only matched frames).
    """
    y_true = []
    y_pred = []

    for gt in ground_truth:
        t = gt["time_sec"]
        true_label = gt["true_posture"]

        # Find all predicted frames within ±1 second of this ground truth point
        candidates = [
            f for f in frame_log
            if abs(f["time"] - t) <= 1.0
            and f["predicted"] in POSTURES  # skip TRANSITION/UNKNOWN
        ]

        if not candidates:
            continue  # no prediction near this timestamp — skip

        # Take the closest one
        best = min(candidates, key=lambda f: abs(f["time"] - t))
        y_true.append(true_label)
        y_pred.append(best["predicted"])

    return y_true, y_pred


# ── Confusion matrix ─────────────────────────────────────────

def build_confusion_matrix(y_true, y_pred):
    """Returns a 4x4 numpy array. Rows = true, Cols = predicted."""
    cm = np.zeros((4, 4), dtype=int)
    label_to_idx = {p: i for i, p in enumerate(POSTURES)}
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t]][label_to_idx[p]] += 1
    return cm


def compute_metrics(cm):
    """Per-class precision, recall, F1 from confusion matrix."""
    metrics = {}
    for i, label in enumerate(POSTURES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        metrics[label] = {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "support":   int(cm[i, :].sum())
        }
    overall_accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    return metrics, round(overall_accuracy, 3)


# ── Print results ─────────────────────────────────────────────

def print_results(cm, metrics, accuracy):
    print("\n" + "="*56)
    print("         SALATVISION — EVALUATION RESULTS")
    print("="*56)

    # Confusion matrix
    print("\nConfusion Matrix (rows = True, cols = Predicted):\n")
    col_w = 12
    header = " " * 12 + "".join(p.center(col_w) for p in POSTURES)
    print(header)
    print("-" * (12 + col_w * 4))
    for i, label in enumerate(POSTURES):
        row = label.ljust(12) + "".join(str(cm[i, j]).center(col_w) for j in range(4))
        print(row)

    # Per-class metrics
    print("\nPer-Class Metrics:\n")
    print(f"{'Posture':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("-" * 50)
    for label, m in metrics.items():
        print(f"{label:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>8.3f} {m['support']:>9}")

    print(f"\nOverall Accuracy: {accuracy:.1%}  ({int(accuracy * cm.sum())}/{cm.sum()} frames)")
    print("="*56)


# ── Plot confusion matrix ─────────────────────────────────────

def plot_confusion_matrix(cm, accuracy, output_path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalise for colour intensity (row-wise)
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div by zero
    cm_norm = cm_norm / row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Proportion (row-normalised)")

    # Axis labels
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(POSTURES, fontsize=11)
    ax.set_yticklabels(POSTURES, fontsize=11)
    ax.set_xlabel("Predicted Posture", fontsize=12, labelpad=10)
    ax.set_ylabel("True Posture", fontsize=12, labelpad=10)
    ax.set_title(
        f"SalatVision — Posture Confusion Matrix\nAccuracy: {accuracy:.1%}",
        fontsize=13, pad=14
    )

    # Cell annotations: count + percentage
    thresh = 0.5
    for i in range(4):
        for j in range(4):
            count = cm[i, j]
            pct   = cm_norm[i, j]
            color = "white" if pct > thresh else "black"
            label = f"{count}\n({pct:.0%})" if count > 0 else "0"
            ax.text(j, i, label, ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nConfusion matrix saved → {output_path}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SalatVision Evaluator")
    parser.add_argument("--session", required=True,
                        help="Path to session JSON (e.g. sessions/Fajr_xxx.json)")
    parser.add_argument("--ground",  required=True,
                        help="Path to ground_truth.csv")
    parser.add_argument("--output",  default="confusion_matrix.png",
                        help="Output path for confusion matrix image")
    args = parser.parse_args()

    print(f"Loading session:      {args.session}")
    print(f"Loading ground truth: {args.ground}")

    frame_log    = load_session(args.session)
    ground_truth = load_ground_truth(args.ground)

    print(f"\nFrames in session log:  {len(frame_log)}")
    print(f"Ground truth entries:   {len(ground_truth)}")

    y_true, y_pred = align(frame_log, ground_truth)
    print(f"Matched pairs:          {len(y_true)}")

    if len(y_true) == 0:
        print("\nERROR: No frames matched. Check that timestamps in ground_truth.csv")
        print("       align with the session log time values.")
        return

    cm = build_confusion_matrix(y_true, y_pred)
    metrics, accuracy = compute_metrics(cm)

    print_results(cm, metrics, accuracy)
    plot_confusion_matrix(cm, accuracy, args.output)


if __name__ == "__main__":
    main()