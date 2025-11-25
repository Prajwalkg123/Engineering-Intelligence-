import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str):
    # Step 1: Read CSV into DataFrame
    df = pd.read_csv(path)
    assert "target" in df.columns, "Dataset must contain a 'target' column."

    # Step 2: Separate features and target
    X_raw = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].values

    # Step 3: Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Step 4: Infer dimensions
    input_dim = X.shape[1]
    output_dim = int(np.max(y)) + 1 if y.dtype.kind in "iu" else 1

    # Step 5: Train/test split (safe fallback if dataset is tiny)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
    except ValueError:
        # fallback: no stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    return X_train, X_test, y_train, y_test, input_dim, output_dim