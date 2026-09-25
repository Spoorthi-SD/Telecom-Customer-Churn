import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------

# Convert TotalCharges from text to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Remove customer ID because it does not help prediction
df = df.drop("customerID", axis=1)

# ---------------------------------------------------------
# Encode Target Variable
# ---------------------------------------------------------

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# ---------------------------------------------------------
# Separate Features and Target
# ---------------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ---------------------------------------------------------
# Convert Categorical Variables
# ---------------------------------------------------------

X = pd.get_dummies(
    X,
    drop_first=True
)

# ---------------------------------------------------------
# Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# Scale Numerical Features
# ---------------------------------------------------------

numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

scaler = StandardScaler()

X_train[numerical_features] = scaler.fit_transform(
    X_train[numerical_features]
)

X_test[numerical_features] = scaler.transform(
    X_test[numerical_features]
)

# ---------------------------------------------------------
# Save Processed Data
# ---------------------------------------------------------

os.makedirs("outputs", exist_ok=True)

X_train.to_csv("outputs/X_train.csv", index=False)
X_test.to_csv("outputs/X_test.csv", index=False)

y_train.to_csv("outputs/y_train.csv", index=False)
y_test.to_csv("outputs/y_test.csv", index=False)

# ---------------------------------------------------------
# Display Information
# ---------------------------------------------------------

print("\nFeature Engineering Completed")

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nTraining Features Shape:")
print(X_train.shape)

print("\nTesting Features Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)

print("\nChurn Distribution in Training Data:")
print(y_train.value_counts())

print("\nChurn Distribution in Testing Data:")
print(y_test.value_counts())

print("\nTotal Features After Encoding:")
print(len(X_train.columns))

print("\nFeature Names:")
print(X_train.columns.tolist())

print("\nProcessed files saved successfully.")