
import pandas as pd

# Load your Kaggle dataset
FILE_PATH = r"F:\Supply-Chain-Risk-Analyze\data\DataCoSupplyChainDataset.csv"

df = pd.read_csv(FILE_PATH, encoding="latin1")

print("=" * 55)
print("SUPPLY CHAIN DATA VALIDATION")
print("=" * 55)

# 1. Calculate actual delay
df["Actual_Delay_Days"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

# 2. Classify delivery using actual and scheduled days
df["Calculated_Status"] = df["Actual_Delay_Days"].apply(
    lambda days: (
        "Late" if days > 0
        else "On Time" if days == 0
        else "Early"
    )
)

print("\n1. CALCULATED DELIVERY CATEGORIES")
print(df["Calculated_Status"].value_counts())

print("\n2. ORIGINAL DELIVERY STATUS")
print(df["Delivery Status"].value_counts())

print("\n3. EXISTING RISK LABEL")
print(df["Late_delivery_risk"].value_counts())

# 4. Compare calculated delay with risk label
print("\n4. ACTUAL DELAY VS RISK LABEL")
print(pd.crosstab(
    df["Calculated_Status"],
    df["Late_delivery_risk"],
    normalize="index"
).mul(100).round(2))

# 5. Compare calculated status with original status
print("\n5. CALCULATED STATUS VS ORIGINAL STATUS")
print(pd.crosstab(
    df["Calculated_Status"],
    df["Delivery Status"]
))

# 6. Check for missing values in key columns
important_columns = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Delivery Status",
    "Late_delivery_risk"
]

print("\n6. MISSING VALUES IN IMPORTANT COLUMNS")
print(df[important_columns].isnull().sum())

# 7. Check whether the risk label matches actual delay
actual_late = (df["Actual_Delay_Days"] > 0).astype(int)
risk_label = df["Late_delivery_risk"]

valid_rows = actual_late.notna() & risk_label.notna()

agreement = (
    actual_late[valid_rows] == risk_label[valid_rows]
).mean() * 100

print("\n7. AGREEMENT BETWEEN ACTUAL DELAY AND RISK LABEL")
print(f"Agreement: {agreement:.2f}%")

print("\nValidation completed successfully!")
