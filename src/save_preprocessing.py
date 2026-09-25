import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)


# ============================================================
# SAME PREPROCESSING USED DURING MODEL TRAINING
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(
    subset=["TotalCharges"]
)

df = df.drop(
    "customerID",
    axis=1
)

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]


# ============================================================
# ONE-HOT ENCODING
# ============================================================

X = pd.get_dummies(
    X,
    drop_first=True
)


# ============================================================
# SAME TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# FIT SCALER ON ORIGINAL TRAINING VALUES
# ============================================================

numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

scaler = StandardScaler()

scaler.fit(
    X_train[numerical_features]
)


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = X_train.columns.tolist()


# ============================================================
# SAVE PREPROCESSING ARTIFACT
# ============================================================

preprocessing = {
    "scaler": scaler,
    "numerical_features": numerical_features,
    "feature_names": feature_names
}

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    preprocessing,
    "model/preprocessing.pkl"
)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING ARTIFACT SAVED CORRECTLY")
print("=" * 60)

print("\nNumerical features:")
print(numerical_features)

print("\nNumber of model features:")
print(len(feature_names))

print("\nFeature names:")
print(feature_names)

print("\nSaved file:")
print("model/preprocessing.pkl")

print("\nPreprocessing setup completed successfully.")
