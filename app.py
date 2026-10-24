
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "DataCoSupplyChainDataset.csv"

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

# Create output folder
Path("outputs").mkdir(exist_ok=True)

# Calculate risk-label percentage
risk = (
    df.groupby("Shipping Mode")["Late_delivery_risk"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

# Create professional chart
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    risk.index,
    risk.values,
    width=0.6
)

# Add percentage labels
for bar, value in zip(bars, risk.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.1f}%",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

ax.set_title(
    "Delivery Risk by Shipping Mode",
    fontsize=16,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Shipping Mode", fontsize=12)
ax.set_ylabel("Orders with Risk Label (%)", fontsize=12)
ax.set_ylim(0, 105)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

plt.xticks(rotation=0)
plt.tight_layout()

# Save chart
plt.savefig(
    "outputs/delivery_risk_by_shipping.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Professional chart saved successfully!")
