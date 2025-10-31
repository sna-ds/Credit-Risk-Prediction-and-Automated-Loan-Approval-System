import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

def show_loan_approval():
    # --- Load model dan preprocessing ---
    model = joblib.load("models/lgbm_classifier.pkl")
    scaler = joblib.load("models/clf_scaler.pkl")
    encoder = joblib.load("models/encoder.pkl")

    selected_features = [
        'RiskScore', 'DebtToIncomeRatio', 'BankruptcyHistory', 'CreditScore', 'NetWorth',
        'MonthlyIncome', 'LoanAmount', 'InterestRate', 'PreviousLoanDefaults', 'CreditCardUtilizationRate'
    ]

    st.title("Automated Loan Approval System")
    st.markdown("""
    Input applicant details below to predict whether the loan will be **approved** or **denied** 
    based on financial and risk indicators.
    
    Choose between:
    - **Single Applicant** (manual input)
    - **Batch Upload** (upload CSV file)
    """)