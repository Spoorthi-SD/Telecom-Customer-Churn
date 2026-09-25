import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

# Load test data and model
X_test = pd.read_csv("outputs/X_test.csv")
model = joblib.load("model/churn_model.pkl")

# Make sure data is numeric
X_test = X_test.astype(float)
X_test_values = X_test.to_numpy(dtype=float)

print("\nCreating individual SHAP explanation...")

# Create SHAP explainer
explainer = shap.LinearExplainer(
    model,
    X_test_values
)

# Select one customer from test data
customer_index = 0
customer = X_test_values[customer_index:customer_index + 1]

# Calculate SHAP values
customer_shap = explainer(customer)

customer_shap_values = np.asarray(
    customer_shap.values,
    dtype=float
)[0]

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Create feature contribution dataframe
individual_explanation = pd.DataFrame({
    "Feature": X_test.columns,
    "Feature_Value": customer[0],
    "SHAP_Value": customer_shap_values
})

# Sort by absolute SHAP impact
individual_explanation["Absolute_SHAP"] = (
    individual_explanation["SHAP_Value"].abs()
)

individual_explanation = individual_explanation.sort_values(
    "Absolute_SHAP",
    ascending=False
)

# Save explanation
individual_explanation.to_csv(
    "outputs/shap_individual_explanation.csv",
    index=False
)

# Print prediction probability
probability = model.predict_proba(customer)[0][1]
prediction = model.predict(customer)[0]

print("\n" + "=" * 60)
print("INDIVIDUAL CUSTOMER SHAP EXPLANATION")
print("=" * 60)

print(f"\nCustomer index: {customer_index}")
print(f"Churn probability: {probability:.2%}")

if prediction == 1:
    print("Prediction: CHURN")
else:
    print("Prediction: NO CHURN")

print("\nTop factors influencing this prediction:")
print(
    individual_explanation[
        ["Feature", "Feature_Value", "SHAP_Value"]
    ].head(10).to_string(index=False)
)

# Create waterfall plot
plt.figure(figsize=(10, 7))

shap.plots.waterfall(
    customer_shap[0],
    max_display=10,
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/shap_individual_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nIndividual SHAP explanation completed successfully.")

print("\nGenerated files:")
print("outputs/shap_individual_explanation.csv")
print("outputs/shap_individual_waterfall.png")