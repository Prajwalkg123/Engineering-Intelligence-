import argparse
from src.data_loader import load_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model, save_metrics
import matplotlib.pyplot as plt
import os

def save_loss_curve(losses, out_path="docs/training_curve.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(losses)+1), losses, color="#1f77b4", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def run_pipeline(args):
    # Load data
    X_train, X_test, y_train, y_test, input_dim, output_dim = load_data(args.data_path)

    # Build model
    model = build_model(input_dim, output_dim, hidden_dim=args.hidden_dim)

    # Train model
    model, losses = train_model(model, X_train, y_train, epochs=args.epochs, lr=args.learning_rate)
    save_loss_curve(losses, out_path="docs/training_curve.png")

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics, path="docs/evaluation.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engineering Intelligence ML Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    run_pipeline(args)