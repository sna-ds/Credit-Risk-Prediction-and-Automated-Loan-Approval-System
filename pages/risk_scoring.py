import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

def show_risk_scoring():
    st.title("Credit Risk Score Prediction")
    st.write("""
    This module predicts an applicant’s **Risk Score**, representing their likelihood of loan default or financial instability.  
    The score ranges from **0 (Low Risk)** to **100 (High Risk)**, based on the applicant’s financial profile.
    """)

    # Load pre-trained model and preprocessors
    model = joblib.load('models/xgb_regressor.pkl')
    scaler = joblib.load('models/reg_scaler.pkl')

    cols = [
        'EmploymentStatus_Unemployed', 'EmploymentStatus_Self-Employed', 'EmploymentStatus_Employed',
        'MonthlyIncome', 'NetWorth', 'DebtToIncomeRatio', 'CreditScore', 'CreditCardUtilizationRate',
        'PreviousLoanDefaults', 'BankruptcyHistory', 'LengthOfCreditHistory', 'LoanAmount',
        'LoanDurationYears', 'InterestRate'
    ]

    def yes_no_encode(value):
        return 1 if value == "Yes" else 0

    with st.form("risk_form"):
        st.subheader("Enter Applicant Information")

        EmploymentStatus = st.selectbox(
            "What is the applicant’s current employment status?",
            ["Employed", "Self-Employed", "Unemployed"]
        )
        MonthlyIncome = st.number_input("Monthly Income ($)", 1_250.0, 500_000.0, 5_000.0, 500.0)
        NetWorth = st.number_input("Net Worth ($)", 1_000.0, 2_603_208.0, 50_000.0, 1_000.0)
        CreditScore = st.number_input("Credit Score", 300, 750, 600)
        LengthOfCreditHistory = st.number_input("Length of Credit History (years)", 1, 30, 10, 1)
        CreditCardUtilizationRate = st.number_input("Credit Card Utilization Rate", 0.01, 1.00, 0.30, 0.01)
        BankruptcyHistory = st.selectbox("Has declared bankruptcy?", ["No", "Yes"])
        PreviousLoanDefaults = st.selectbox("Has previous loan defaults?", ["No", "Yes"])
        DebtToIncomeRatio = st.number_input("Debt-to-Income Ratio", 0.00, 1.00, 0.50, 0.01)
        LoanAmount = st.number_input("Loan Amount ($)", 1_000.0, 1_000_000.0, 10_000.0, 1_000.0)
        LoanDurationYears = st.number_input("Loan Duration (years)", 1, 10, 5, 1)
        InterestRate = st.number_input("Interest Rate (%)", 0.00, 1.00, 0.40, 0.01)

        submitted = st.form_submit_button("🔮 Predict Risk Score")

    if submitted:
        with st.spinner("Calculating risk score..."):
            # Manual one-hot encoding for EmploymentStatus
            EmploymentStatus_Unemployed = 1 if EmploymentStatus == "Unemployed" else 0
            EmploymentStatus_Self_Employed = 1 if EmploymentStatus == "Self-Employed" else 0
            EmploymentStatus_Employed = 1 if EmploymentStatus == "Employed" else 0

            # Create DataFrame for input
            input_data = pd.DataFrame([{
                'EmploymentStatus_Unemployed': EmploymentStatus_Unemployed,
                'EmploymentStatus_Self-Employed': EmploymentStatus_Self_Employed,
                'EmploymentStatus_Employed': EmploymentStatus_Employed,
                'MonthlyIncome': MonthlyIncome,
                'NetWorth': NetWorth,
                'DebtToIncomeRatio': DebtToIncomeRatio,
                'CreditScore': CreditScore,
                'CreditCardUtilizationRate': CreditCardUtilizationRate,
                'PreviousLoanDefaults': yes_no_encode(PreviousLoanDefaults),
                'BankruptcyHistory': yes_no_encode(BankruptcyHistory),
                'LengthOfCreditHistory': LengthOfCreditHistory,
                'LoanAmount': LoanAmount,
                'LoanDurationYears': LoanDurationYears,
                'InterestRate': InterestRate
            }])

            # Apply scaling
            df_scaled = pd.DataFrame(scaler.transform(input_data), columns=cols)

            # Predict risk score
            risk_score = model.predict(df_scaled)[0]
            risk_score = np.clip(risk_score, 0, 100)  # Keep within 0–100

            # Display results
            st.subheader(f"Predicted Risk Score: **{risk_score:.2f} / 100**")

            # Feature importance interpretation (safe for XGBoost)
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": df_scaled.columns,
                "Importance": importance
            }).sort_values("Importance", ascending=False)

            top_features = importance_df.head(2)["Feature"].tolist()

            # Interpretation
            if risk_score < 40:
                reason = f"🟢 The applicant’s low risk score is mainly supported by strong {top_features[0].lower()} and stable {top_features[1].lower()}."
            elif 40 <= risk_score < 70:
                reason = f"🟠 The applicant’s moderate risk score is primarily influenced by {top_features[0].lower()} and {top_features[1].lower()}."
            else:
                reason = f"🔴 The applicant’s high risk score is mainly driven by high {top_features[0].lower()} and {top_features[1].lower()}."

            st.info(reason)

           