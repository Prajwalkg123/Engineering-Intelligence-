import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def train_model(model, X_train, y_train, epochs=50, lr=1e-3, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Convert to tensors
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Loss function
    criterion = nn.CrossEntropyLoss() if model.output_dim > 1 else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in dl:
            optimizer.zero_grad()
            logits = model(xb)
            if model.output_dim == 1:
                yb = yb.float().unsqueeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dl)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    return model, epoch_losses