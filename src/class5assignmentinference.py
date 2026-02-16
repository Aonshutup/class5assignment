import os

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics_from_cm(cm: np.ndarray):
    """Calculate common binary‑classification metrics from a 2x2 confusion matrix."""
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def plot_confusion_matrices(cm: np.ndarray, accuracy: float, output_dir: str):
    """Create and save raw and normalized confusion matrix plots."""
    os.makedirs(output_dir, exist_ok=True)

    class_names = ["No", "Yes"]

    # Standard confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix - Class 5 Assignment Model", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    tn, fp, fn, tp = cm.ravel()
    stats_text = (
        f"True Negatives: {tn}\n"
        f"False Positives: {fp}\n"
        f"False Negatives: {fn}\n"
        f"True Positives: {tp}\n\n"
        f"Accuracy: {accuracy:.4f}"
    )
    plt.text(
        2.5,
        0.5,
        stats_text,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        verticalalignment="center",
    )

    raw_path = os.path.join(output_dir, "confusion_matrix_class5.jpg")
    plt.tight_layout()
    plt.savefig(raw_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Normalized confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Percentage"},
    )
    plt.title(
        "Normalized Confusion Matrix - Class 5 Assignment Model",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    norm_path = os.path.join(output_dir, "confusion_matrix_class5_normalized.jpg")
    plt.tight_layout()
    plt.savefig(norm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix plots saved to:\n  {raw_path}\n  {norm_path}")


def main():
    print("=" * 70)
    print("Class 5 Assignment - Inference & Evaluation")
    print("=" * 70)

    data_path = "data/class5.csv"
    model_path = "examples/class5_assignment_model.h5"
    output_dir = "data/outputclass5"

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"\nLoading trained model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Please run class5assigmenttraining.py first."
        )
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")

    # Load data
    print(f"\nLoading evaluation data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    target_col = "ProdTaken"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    print(f"\nTarget distribution:\n{y.value_counts()}")

    # One‑hot encode categorical features
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"\nCategorical columns: {categorical_cols}")

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"Number of features after encoding: {X_encoded.shape[1]}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Predictions
    print("\nMaking predictions...")
    y_pred_proba = model.predict(X_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    try:
        auc_score = roc_auc_score(y, y_pred_proba)
    except Exception:
        auc_score = float("nan")

    print(f"\nAccuracy: {accuracy:.4f}")
    if not np.isnan(auc_score):
        print(f"AUC: {auc_score:.4f}")
    else:
        print("AUC: could not be computed")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, digits=4))

    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrices
    plot_confusion_matrices(cm, accuracy, output_dir)

    # Aggregate metrics and save evaluation report similar to examples/evaluation_metrics.txt
    m = calculate_metrics_from_cm(cm)

    metrics_path = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Model Evaluation Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Data: {data_path}\n")
        f.write(f"Number of samples: {len(y)}\n\n")
        f.write(f"Accuracy:     {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)\n")
        f.write(f"Precision:    {m['precision']:.4f} ({m['precision']*100:.2f}%)\n")
        f.write(f"Recall:       {m['recall']:.4f} ({m['recall']*100:.2f}%)\n")
        f.write(f"F1-Score:     {m['f1_score']:.4f}\n")
        f.write(f"Specificity:  {m['specificity']:.4f} ({m['specificity']*100:.2f}%)\n\n")
        f.write("Confusion Matrix Breakdown:\n")
        f.write(f"True Positives:  {m['tp']}\n")
        f.write(f"True Negatives:  {m['tn']}\n")
        f.write(f"False Positives: {m['fp']}\n")
        f.write(f"False Negatives: {m['fn']}\n\n")
        f.write("=" * 70 + "\n")
        f.write("Confusion Matrix:\n")
        f.write("=" * 70 + "\n")
        f.write(str(cm) + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("Classification Report:\n")
        f.write("=" * 70 + "\n")
        f.write(classification_report(y, y_pred, digits=4))

    print(f"\nEvaluation metrics saved to: {metrics_path}")
    print("\nInference & evaluation complete.")


if __name__ == "__main__":
    main()

