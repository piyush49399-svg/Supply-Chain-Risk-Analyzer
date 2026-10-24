
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

# Load dataset
FILE_PATH = r"F:\Supply-Chain-Risk-Analyze\data\DataCoSupplyChainDataset.csv"
df = pd.read_csv(FILE_PATH, encoding="latin1")

# Use a smaller, order-time feature set
numeric_features = [
    "Order Item Product Price",
    "Order Item Quantity",
    "Order Item Discount",
    "Order Item Discount Rate"
]

categorical_features = [
    "Type",
    "Customer Segment",
    "Market",
    "Order Region",
    "Shipping Mode",
    "Department Name"
]

features = numeric_features + categorical_features

missing_features = [c for c in features if c not in df.columns]

if not features:
    raise ValueError("No matching features found in the dataset")

print("Features used for training:", features)
X = df[features]
y = df["Late_delivery_risk"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_features = [c for c in numeric_features if c in features]
categorical_features = [c for c in categorical_features if c in features]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("Training early-risk model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nEARLY-RISK MODEL RESULTS")
print(classification_report(y_test, predictions, digits=4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))


import joblib
from pathlib import Path

# Save the trained model
model_folder = Path("models")
model_folder.mkdir(exist_ok=True)

model_path = model_folder / "early_risk_model.joblib"
joblib.dump(model, model_path)

print("\nFeatures used:")
print(features)

print(f"\nModel saved successfully to: {model_path.resolve()}")
