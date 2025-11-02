# Credit Risk Prediction & Automated Loan Approval System

An end-to-end **Machine Learning application** that predicts **Credit Risk Scores** and automates **Loan Approval Decisions** using financial and behavioral data.  
Built with **Streamlit**, this project integrates data preprocessing, model training, explainability (SHAP), and an interactive user interface.

---

## Project Overview

This project demonstrates how **machine learning** can assist financial institutions in evaluating loan applications efficiently and accurately.  
It includes two main predictive models **(Regression)** and (Classification)**  

- Predict the **credit risk score** (0–100) of applicants based on their financial indicators.
- Automate **loan approval** decisions using the predicted risk score and key credit metrics.
- Deploy both models in a **streamlit web app** with explanations using **feature importance**.

| Category | Details |
|-----------|--------------|
| **Dataset** | [Kaggle - Financial Risk for Loan Approval](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval/data) |
| **Deployment** | [Streamlit](https://risk-scoring-and-automated-loan-approval.streamlit.app/) |

** Key Features **
- **Exploratory Data Analysis (EDA)** to identify **data patterns**, understand **applicant profiles**, and **portfolio exposure** from approved loans.
- **Data Preprocessing Pipelines**, used **feature selection**, **RobustScaler** for numerical features, **One-Hot-Emcoding** for categorical features.
- **Risk Score Prediction** using a trained **XGBoost Regressor**. 
- **Automated Loan Decision** using a **LightGBM Classifier**.
- **Interactive Streamlit UI** for predictions.  
- **Model explainability** with feature importance insights.  

