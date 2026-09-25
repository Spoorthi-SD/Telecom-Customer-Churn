import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


# ---------------------------------------------------------
# Load Data and Model
# ---------------------------------------------------------

X_test = pd.read_csv("outputs/X_test.csv")
y_test = pd.read_csv("outputs/y_test.csv").squeeze()

model = joblib.load("model/churn_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"]
    )
)


# ---------------------------------------------------------
# ROC-AUC
# ---------------------------------------------------------

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print(f"ROC-AUC Score: {roc_auc:.4f}")


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/roc_curve.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# Feature Coefficients
# ---------------------------------------------------------

coefficients = model.coef_[0]

coefficient_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Importance": abs(coefficients)
})

coefficient_df = coefficient_df.sort_values(
    "Absolute_Importance",
    ascending=False
)

coefficient_df.to_csv(
    "outputs/feature_coefficients.csv",
    index=False
)


# ---------------------------------------------------------
# Top Feature Coefficients Chart
# ---------------------------------------------------------

top_features = coefficient_df.head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Coefficient",
    y="Feature"
)

plt.title("Top Features Influencing Churn Prediction")
plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# Final Output
# ---------------------------------------------------------

print("\nExplainability analysis completed successfully.")

print("\nGenerated files:")

for file in os.listdir("outputs"):
    if (
        file.endswith(".png")
        or file.endswith(".csv")
    ):
        print(file)