import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from task5_transformer_model import TransformerRNA

# 1️⃣ Charger embeddings
X = np.load("data/embeddings_task4.npy")
y = np.random.randint(0, 3, size=len(X))  # labels techniques

dataset = TensorDataset(
    torch.tensor(X, dtype=torch.float32),
    torch.tensor(y, dtype=torch.long)
)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ Hyperparamètres
num_heads_list = [2, 4]
num_layers_list = [1, 2]
hidden_dim_list = [128, 256]
learning_rates = [1e-3, 5e-4]

results = []

# 3️⃣ Grid Search
for heads in num_heads_list:
    for layers in num_layers_list:
        for dim in hidden_dim_list:
            for lr in learning_rates:

                model = TransformerRNA(
                    input_dim=100,
                    num_heads=heads,
                    num_layers=layers,
                    hidden_dim=dim
                ).to(device)

                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                model.train()
                total_loss = 0

                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)

                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(loader)
                print(f"Heads={heads}, Layers={layers}, Dim={dim}, LR={lr} -> Loss={avg_loss:.4f}")

                results.append([heads, layers, dim, lr, avg_loss])

# 4️⃣ Sauvegarde
df = pd.DataFrame(
    results,
    columns=["num_heads", "num_layers", "hidden_dim", "learning_rate", "loss"]
)

df.to_csv("hyperparameter_results.csv", index=False)

with open("results.txt", "w") as f:
    for r in results:
        f.write(str(r) + "\n")
