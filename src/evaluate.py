import json
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test, report_path="docs/classification_report.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_test_t)

    # Classification vs regression
    is_classification = len(set(y_test)) <= 2

    if is_classification:
        if logits.shape[-1] == 1:
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
        else:
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        cm = confusion_matrix(y_test, preds)

        # Save classification report
        report = classification_report(y_test, preds)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)

        # Save confusion matrix heatmap
        os.makedirs("docs", exist_ok=True)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("docs/confusion_matrix.png", dpi=150)
        plt.close()

        print("Sample predictions:", preds[:10])
        print("True labels:", y_test[:10])
        print("Confusion Matrix:\n", cm)

        return {
            "task": "classification",
            "accuracy": float(acc),
            "f1_weighted": float(f1),
            "confusion_matrix": cm.tolist()
        }

    else:
        preds = logits.squeeze(-1).cpu().numpy()
        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))

        print("Sample predictions:", preds[:10])
        print("True values:", y_test[:10])

        return {
            "task": "regression",
            "rmse": rmse,
            "mse": float(mse)
        }

def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)