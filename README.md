# Telecom Customer Churn Prediction

## Internship
- Organization: CODTECH IT Solutions
- Task: Telecom Customer Churn Prediction
- Intern ID: CITS9278

## Project Overview

This project uses Machine Learning to predict whether a telecom customer is likely to churn based on customer profile, service usage, contract, and billing information.

A Streamlit dashboard is developed to provide:
- Customer churn analysis
- Model performance comparison
- Explainable AI insights
- Threshold analysis
- Individual customer churn prediction
- Churn probability and prediction explanation

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer information such as:
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming Services
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges
- Churn

## Machine Learning Models

The following models were trained and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Logistic Regression was selected as the final model based on the evaluation results.

## Explainable AI

SHAP (SHapley Additive exPlanations) was used to understand the model predictions.

The project includes:
- Global SHAP feature importance
- Individual prediction explanation
- SHAP waterfall explanation

Model coefficients are also analyzed to understand the direction and strength of model associations.

## Threshold Analysis

Different classification thresholds were evaluated using:
- Precision
- Recall
- F1 Score

This helps understand how changing the prediction threshold affects churn classification.

## Streamlit Dashboard

The application contains:

### Dashboard

- Overview
- Churn Analysis
- Model Performance
- Explainable AI
- Threshold Analysis

### Customer Prediction

Users can enter customer details and obtain:
- Churn probability
- Churn prediction
- Individual prediction explanation
- Factors increasing or reducing the predicted churn risk

## Project Structure

Telecom_Customer_Churn/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── model/
│   ├── churn_model.pkl
│   ├── feature_names.pkl
│   └── preprocessing.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── feature_coefficients.csv
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── shap_summary.png
│   ├── shap_feature_importance.png
│   ├── shap_feature_importance.csv
│   ├── shap_individual_explanation.csv
│   ├── shap_individual_waterfall.png
│   ├── threshold_analysis.csv
│   └── threshold_analysis.png
│
├── src/
│   ├── train_model.py
│   ├── explainability.py
│   ├── shap_explainability.py
│   ├── shap_individual.py
│   ├── threshold_analysis.py
│   └── save_preprocessing.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

## Installation

Clone the repository and install the required dependencies:

pip install -r requirements.txt

## Run the Application

Start the Streamlit application using:

streamlit run app.py

The application will open in the browser.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib

## Conclusion

This project demonstrates an end-to-end machine learning workflow for telecom customer churn prediction, from data preprocessing and model comparison to explainable predictions and an interactive Streamlit dashboard.
