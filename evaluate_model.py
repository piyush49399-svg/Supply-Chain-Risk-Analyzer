
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.dummy import DummyClassifier

# Load dataset and trained model
FILE_PATH = r"F:\Supply-Chain-Risk-Analyze\data\DataCoSupplyChainDataset.csv"
MODEL_PATH = r"F:\Supply-Chain-Risk-Analyze\models\delivery_risk_model.joblib"

df = pd.read_csv(FILE_PATH, encoding="latin1")
model = joblib.load(MODEL_PATH)

numeric_features = [
    "Benefit per order",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Product Price",
    "Latitude",
    "Longitude"
]

categorical_features = [
    "Type",
    "Category Name",
    "Customer Segment",
    "Market",
    "Order Region",
    "Order State",
    "Shipping Mode",
    "Department Name"
]

features = [
    col for col in numeric_features + categorical_features
    if col in df.columns
]

X = df[features]
y = df["Late_delivery_risk"]

valid = y.notna()
X = X.loc[valid]
y = y.loc[valid].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Evaluate saved model
predictions = model.predict(X_test)

# Majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Random Forest": [
        accuracy_score(y_test, predictions),
        precision_score(y_test, predictions),
        recall_score(y_test, predictions),
        f1_score(y_test, predictions)
    ],
    "Majority Baseline": [
        accuracy_score(y_test, baseline_predictions),
        precision_score(
            y_test, baseline_predictions, zero_division=0
        ),
        recall_score(
            y_test, baseline_predictions, zero_division=0
        ),
        f1_score(
            y_test, baseline_predictions, zero_division=0
        )
    ]
})

print("\nMODEL VS BASELINE")
print(results.round(4).to_string(index=False))

# Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Risk 0", "Risk 1"],
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

results.to_csv("model_comparison.csv", index=False)

print("\nSaved confusion_matrix.png")
print("Saved model_comparison.csv")
