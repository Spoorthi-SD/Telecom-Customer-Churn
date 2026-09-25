import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# Load Processed Data
# ---------------------------------------------------------

X_train = pd.read_csv("outputs/X_train.csv")
X_test = pd.read_csv("outputs/X_test.csv")

y_train = pd.read_csv("outputs/y_train.csv").squeeze()
y_test = pd.read_csv("outputs/y_test.csv").squeeze()


# ---------------------------------------------------------
# Train Final Logistic Regression Model
# ---------------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# Create Model Directory
# ---------------------------------------------------------

os.makedirs("model", exist_ok=True)


# ---------------------------------------------------------
# Save Model
# ---------------------------------------------------------

joblib.dump(
    model,
    "model/churn_model.pkl"
)


# ---------------------------------------------------------
# Save Feature Names
# ---------------------------------------------------------

feature_names = X_train.columns.tolist()

joblib.dump(
    feature_names,
    "model/feature_names.pkl"
)


# ---------------------------------------------------------
# Display Information
# ---------------------------------------------------------

print("\nFinal Model Training Completed")

print("\nModel:")
print("Logistic Regression")

print("\nNumber of Features:")
print(len(feature_names))

print("\nTraining Samples:")
print(len(X_train))

print("\nTesting Samples:")
print(len(X_test))

print("\nModel saved successfully:")
print("model/churn_model.pkl")

print("\nFeature names saved successfully:")
print("model/feature_names.pkl")