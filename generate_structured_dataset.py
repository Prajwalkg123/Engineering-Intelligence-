import pandas as pd
import numpy as np
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Parameters
n_samples = 200
np.random.seed(42)

# Features
f1 = np.random.rand(n_samples) * 10
f2 = np.random.rand(n_samples) * 10
f3 = np.random.rand(n_samples) * 10

# Rule-based target: if f1 + f2 > 10 → class 1, else 0
target = ((f1 + f2) > 10).astype(int)

# Build DataFrame
df = pd.DataFrame({
    "feature1": f1,
    "feature2": f2,
    "feature3": f3,
    "target": target
})

# Save to CSV
df.to_csv("data/structured_data.csv", index=False)
print("✅ Structured dataset saved to data/structured_data.csv")