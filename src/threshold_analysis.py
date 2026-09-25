import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import os

# Load test data and model
X_test = pd.read_csv("outputs/X_test.csv")
y_test = pd.read_csv("outputs/y_test.csv").squeeze()

model = joblib.load("model/churn_model.pkl")

# Get churn probabilities
y_probability = model.predict_proba(X_test)[:, 1]

# Thresholds to evaluate
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

results = []

for threshold in thresholds:

    # Convert probabilities into predictions
    y_pred = (y_probability >= threshold).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })


# Create dataframe
threshold_df = pd.DataFrame(results)

# Save results
os.makedirs("outputs", exist_ok=True)

threshold_df.to_csv(
    "outputs/threshold_analysis.csv",
    index=False
)


# Print results
print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

print(
    threshold_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1-Score": "{:.4f}".format
        }
    )
)


# Plot threshold performance
plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1-Score"],
    marker="o",
    label="F1-Score"
)

plt.xlabel("Probability Threshold")
plt.ylabel("Score")
plt.title("Threshold Analysis for Customer Churn")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/threshold_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nThreshold analysis completed successfully.")

print("\nGenerated files:")
print("outputs/threshold_analysis.csv")
print("outputs/threshold_analysis.png")