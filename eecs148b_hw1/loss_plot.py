import pandas as pd
import matplotlib.pyplot as plt

# Load files
full_df = pd.read_csv("/home/anahita/Spring_2026/CS148b/hw1/eecs148b_hw1/runs/full_train/metrics.csv")
no_ln_df = pd.read_csv("/home/anahita/Spring_2026/CS148b/hw1/eecs148b_hw1/runs/no_pe/metrics.csv")

# Keep training rows only
full_df = full_df[full_df["split"] == "train"]
no_ln_df = no_ln_df[no_ln_df["split"] == "train"]

# Plot
plt.figure(figsize=(8,5))

plt.plot(full_df["step"], full_df["loss"], label="Sinusoidal PE")
plt.plot(no_ln_df["step"], no_ln_df["loss"], label="NoPE")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save figure
plt.savefig("loss_comparison_no_pe.png", dpi=300, bbox_inches="tight")

plt.show()


plt.figure(figsize=(8,5))
full_df["smooth"] = full_df["loss"].rolling(10).mean()
no_ln_df["smooth"] = no_ln_df["loss"].rolling(10).mean()

plt.plot(full_df["step"], full_df["smooth"], label="Sinusoidal PE")
plt.plot(no_ln_df["step"], no_ln_df["smooth"], label="NoPE")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()
# Save figure
plt.savefig("loss_comparison_smooth_no_pe.png", dpi=300, bbox_inches="tight")
plt.show()
