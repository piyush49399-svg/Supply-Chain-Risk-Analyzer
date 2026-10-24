
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = r"F:\Supply-Chain-Risk-Analyze\models\delivery_risk_model.joblib"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Extract model components
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Calculate feature importance
feature_names = preprocessor.get_feature_names_out()
importance_scores = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_scores
})

# Select top 15 features
top_features = importance_df.nlargest(15, "Importance")

# Save the results
Path("outputs").mkdir(exist_ok=True)

top_features.to_csv(
    "outputs/top_features.csv",
    index=False
)

# Create chart
top_features = top_features.sort_values("Importance")

plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.title("Top 15 Features Influencing Model Predictions")
plt.xlabel("Feature Importance")
plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nTop 15 features:")
print(top_features.sort_values(
    "Importance", ascending=False
).to_string(index=False))

print("\nSaved chart to outputs/feature_importance.png")
