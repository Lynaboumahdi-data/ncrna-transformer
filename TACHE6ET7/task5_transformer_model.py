import torch
import torch.nn as nn

class TransformerRNA(nn.Module):
    def __init__(self, input_dim=100, num_heads=4, num_layers=2, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.projection = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.projection(x).unsqueeze(1)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)
