
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
FILE_PATH = r"F:\Supply-Chain-Risk-Analyze\data\DataCoSupplyChainDataset.csv"

df = pd.read_csv(FILE_PATH, encoding="latin1")

print("Dataset loaded:", df.shape)

# --------------------------------------------------
# 2. SELECT TARGET AND FEATURES
# --------------------------------------------------
TARGET = "Late_delivery_risk"

# Use features intended to be available at order time.
# Exclude actual delivery outcomes and direct risk/status labels.
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

features = numeric_features + categorical_features

# Keep only columns that exist in the dataset
features = [col for col in features if col in df.columns]

X = df[features].copy()
y = df[TARGET].copy()

# Treat missing target labels as invalid
valid = y.notna()
X = X.loc[valid]
y = y.loc[valid].astype(int)

print("Selected features:", features)
print("Rows used:", len(X))
print("Target distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------
numeric_features = [
    col for col in numeric_features if col in features
]
categorical_features = [
    col for col in categorical_features if col in features
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 5. RANDOM FOREST MODEL
# --------------------------------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# --------------------------------------------------
# 6. EVALUATION
# --------------------------------------------------
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# --------------------------------------------------
# 7. SAVE MODEL
# --------------------------------------------------
output_folder = Path("models")
output_folder.mkdir(exist_ok=True)

model_path = output_folder / "delivery_risk_model.joblib"
joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path.resolve()}")
print("Training and evaluation completed!")
