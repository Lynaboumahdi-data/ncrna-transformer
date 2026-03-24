import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hyperparameter_results.csv")

# Graphique Loss vs Hidden Dim
plt.figure()
for dim in df["hidden_dim"].unique():
    subset = df[df["hidden_dim"] == dim]
    plt.plot(subset["learning_rate"], subset["loss"], label=f"dim={dim}")

plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Impact du learning rate et hidden_dim")
plt.legend()
plt.savefig("loss_vs_lr.png")
plt.show()
