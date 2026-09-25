import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os


# ---------------------------------------------------------
# Load Data and Model
# ---------------------------------------------------------

X_train = pd.read_csv("outputs/X_train.csv")
X_test = pd.read_csv("outputs/X_test.csv")

model = joblib.load("model/churn_model.pkl")


# ---------------------------------------------------------
# Convert Data to Numeric
# ---------------------------------------------------------

X_train = X_train.astype(float)
X_test = X_test.astype(float)

X_train_values = X_train.to_numpy(dtype=float)
X_test_values = X_test.to_numpy(dtype=float)


# ---------------------------------------------------------
# Create SHAP Explainer
# ---------------------------------------------------------

print("\nCreating SHAP explainer...")

explainer = shap.LinearExplainer(
    model,
    X_train_values
)

shap_values = explainer(
    X_test_values
)


# ---------------------------------------------------------
# Create Output Directory
# ---------------------------------------------------------

os.makedirs("outputs", exist_ok=True)


# ---------------------------------------------------------
# Create SHAP DataFrame
# ---------------------------------------------------------

shap_values_array = np.asarray(
    shap_values.values,
    dtype=float
)

shap_values_df = pd.DataFrame(
    shap_values_array,
    columns=X_test.columns
)


# ---------------------------------------------------------
# SHAP Summary Plot
# ---------------------------------------------------------

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values_array,
    X_test_values,
    feature_names=X_test.columns,
    show=False
)

plt.title("SHAP Feature Impact on Customer Churn")
plt.tight_layout()

plt.savefig(
    "outputs/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# SHAP Bar Plot
# ---------------------------------------------------------

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values_array,
    X_test_values,
    feature_names=X_test.columns,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance")
plt.tight_layout()

plt.savefig(
    "outputs/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Calculate Mean Absolute SHAP Values
# ---------------------------------------------------------

mean_shap = np.mean(
    np.abs(shap_values_array),
    axis=0
)

shap_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_Absolute_SHAP": mean_shap
})

shap_importance = shap_importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)


# ---------------------------------------------------------
# Save SHAP Importance
# ---------------------------------------------------------

shap_importance.to_csv(
    "outputs/shap_feature_importance.csv",
    index=False
)


# ---------------------------------------------------------
# Display Top Features
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TOP SHAP FEATURES")
print("=" * 60)

print(
    shap_importance.head(15).to_string(index=False)
)


# ---------------------------------------------------------
# Final Output
# ---------------------------------------------------------

print("\nSHAP explainability completed successfully.")

print("\nGenerated files:")
print("outputs/shap_summary.png")
print("outputs/shap_feature_importance.png")
print("outputs/shap_feature_importance.csv")