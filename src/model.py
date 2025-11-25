import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout=0.2):
        super(MLP, self).__init__()
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def build_model(input_dim, output_dim, hidden_dim=64, dropout=0.2):
    """
    Build a feedforward neural network.
    - input_dim: number of features
    - output_dim: number of classes (classification) or 1 (regression)
    - hidden_dim: hidden layer size
    - dropout: dropout probability
    """
    return MLP(input_dim, output_dim, hidden_dim, dropout)