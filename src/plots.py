import os
import matplotlib.pyplot as plt
import numpy as np

def save_loss_curve(losses, out_path="docs/training_curve.png"):
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  plt.figure(figsize=(6,4))
  plt.plot(np.arange(1, len(losses)+1), losses, color="#1f77b4", linewidth=2)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training loss per epoch")
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close()