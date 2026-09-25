import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder if it does not exist
os.makedirs("outputs", exist_ok=True)

# Load dataset
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Set plotting style
sns.set_style("whitegrid")


# ---------------------------------------------------------
# 1. Churn Distribution
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("outputs/churn_distribution.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# 2. Churn Percentage
# ---------------------------------------------------------

churn_percentage = df["Churn"].value_counts(normalize=True) * 100

plt.figure(figsize=(7, 5))

sns.barplot(
    x=churn_percentage.index,
    y=churn_percentage.values
)

plt.title("Churn Percentage")
plt.xlabel("Churn")
plt.ylabel("Percentage (%)")

plt.tight_layout()
plt.savefig("outputs/churn_percentage.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# 3. Churn by Contract Type
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("outputs/churn_by_contract.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# 4. Churn by Internet Service
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("outputs/churn_by_internet.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# 5. Tenure Distribution by Churn
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=20,
    kde=True,
    element="step"
)

plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("outputs/tenure_by_churn.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# 6. Monthly Charges by Churn
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.savefig("outputs/monthly_charges_by_churn.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# 7. Total Charges by Churn
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="TotalCharges"
)

plt.title("Total Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Total Charges")

plt.tight_layout()
plt.savefig("outputs/total_charges_by_churn.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# 8. Churn by Payment Method
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)

plt.title("Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("outputs/churn_by_payment_method.png", dpi=300)
plt.close()


print("\nEDA completed successfully.")

print("\nGenerated files:")

for file in os.listdir("outputs"):
    if file.endswith(".png"):
        print(file)