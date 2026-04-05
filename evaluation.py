# ============================================
# evaluation.py
# This file checks how well our model performed.
# It calculates:
#   - Accuracy (% of correct predictions)
#   - Confusion Matrix (TP, TN, FP, FN)
#   - Shows sample predictions
#   - Plots accuracy graph
# ============================================

import numpy as np
import matplotlib
matplotlib.use("Agg")   # use non-interactive backend (saves to file instead of showing window)
import matplotlib.pyplot as plt


def compute_accuracy(y_true, y_pred):
    """
    Calculate accuracy = number of correct predictions / total predictions

    Example: y_true = [1, 0, 1, 1], y_pred = [1, 0, 0, 1]
    Correct = 3 out of 4 = 0.75 (75%)
    """
    correct = 0
    total = len(y_true)

    for i in range(total):
        if y_true[i] == y_pred[i]:
            correct = correct + 1

    accuracy = correct / total
    return accuracy


def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix values:

    TP (True Positive)  = Model said Positive, and it WAS Positive
    TN (True Negative)  = Model said Negative, and it WAS Negative
    FP (False Positive) = Model said Positive, but it was actually Negative
    FN (False Negative) = Model said Negative, but it was actually Positive
    """
    tp = 0   # True Positive
    tn = 0   # True Negative
    fp = 0   # False Positive
    fn = 0   # False Negative

    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp = tp + 1
        elif y_pred[i] == 0 and y_true[i] == 0:
            tn = tn + 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            fp = fp + 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            fn = fn + 1

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def print_confusion_matrix(cm):
    """Print the confusion matrix in a nice format."""
    print("\n+--------------------------------------+")
    print("|         CONFUSION  MATRIX            |")
    print("+--------------------------------------+")
    print("|                 Predicted            |")
    print("|              Neg(0)    Pos(1)        |")
    print(f"|  Actual Neg  {cm['TN']:>5}     {cm['FP']:>5}         |")
    print(f"|  Actual Pos  {cm['FN']:>5}     {cm['TP']:>5}         |")
    print("+--------------------------------------+")


def show_sample_predictions(reviews, y_true, y_pred, n=10):
    """
    Print a few sample predictions so we can see how the model works.
    Shows the review text, the true label, and what the model predicted.
    """
    label_names = {0: "Negative", 1: "Positive"}

    print(f"\n{'='*70}")
    print(f"{'SAMPLE PREDICTIONS':^70}")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Review (short)':<36} {'True':<10} {'Predicted':<10}")
    print(f"{'-'*70}")

    # Pick random samples to show
    total = min(n, len(reviews))
    indices = np.random.choice(len(reviews), size=total, replace=False)

    for idx in indices:
        # Shorten long reviews for display
        if len(reviews[idx]) > 33:
            short_review = reviews[idx][:33] + "..."
        else:
            short_review = reviews[idx]

        true_label = label_names[y_true[idx]]
        pred_label = label_names[y_pred[idx]]
        print(f"{idx:<4} {short_review:<36} {true_label:<10} {pred_label:<10}")

    print(f"{'='*70}\n")


def plot_accuracy(accuracy_list, save_path="accuracy_plot.png"):
    """
    Plot a graph: X-axis = Epoch number, Y-axis = Accuracy
    Saves the graph as an image file.
    """
    epochs = list(range(1, len(accuracy_list) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy_list, marker="o", linewidth=2,
             color="blue", markerfacecolor="orange", markersize=7)
    plt.title("Training Accuracy vs Epoch", fontsize=15, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Accuracy plot saved to '{save_path}'")
