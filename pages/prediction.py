import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

def show_prediction():
    st.title("Credit Risk & Loan Approval Prediction")

    tab1, tab2 = st.tabs(["📈 Risk Scoring", "✅ Loan Approval Decision"])

    # TAB 1: RISK SCORING PREDICTION
    with tab1:
        st.subheader("Credit Risk Score Prediction")
        st.write("""
        Input applicant details below to predict applicant’s **Risk Score**, representing their likelihood of loan default or financial instability.  
        The score ranges from **0 (Low Risk)** to **100 (High Risk)**, based on the applicant’s financial profile.
        """)

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
            EmploymentStatus = st.selectbox(
                "Employment Status", ["Employed", "Self-Employed", "Unemployed"]
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

            submitted_risk = st.form_submit_button("🔮 Predict Risk Score")

        if submitted_risk:
            with st.spinner("Calculating risk score..."):
                # Manual encoding
                EmploymentStatus_Unemployed = 1 if EmploymentStatus == "Unemployed" else 0
                EmploymentStatus_Self_Employed = 1 if EmploymentStatus == "Self-Employed" else 0
                EmploymentStatus_Employed = 1 if EmploymentStatus == "Employed" else 0

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

                df_scaled = pd.DataFrame(scaler.transform(input_data), columns=cols)
                risk_score = np.clip(model.predict(df_scaled)[0], 0, 100)

                st.subheader(f"Predicted Risk Score: **{risk_score:.2f} / 100**")

                # Feature importance interpretation
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": df_scaled.columns,
                    "Importance": importance
                }).sort_values("Importance", ascending=False)
                top_features = importance_df.head(2)["Feature"].tolist()

                if risk_score < 40:
                    reason = f"🟢 Low Risk — strong {top_features[0].lower()} and solid {top_features[1].lower()}."
                elif 40 <= risk_score < 70:
                    reason = f"🟠 Medium Risk — mainly influenced by {top_features[0].lower()} and {top_features[1].lower()}."
                else:
                    reason = f"🔴 High Risk — driven by high {top_features[0].lower()} and {top_features[1].lower()}."

                st.info(reason)

    # TAB 2: LOAN APPROVAL DECISION
    with tab2:
        st.subheader("Automated Loan Approval Decision")
        st.markdown("""
        Input applicant details below to predict whether the loan will be **approved** or **denied**, 
        based on financial and risk indicators.
        """)

        model = joblib.load('models/lgbm_classifier.pkl')
        scaler = joblib.load('models/clf_scaler.pkl')

        cols = [
            'RiskScore', 'DebtToIncomeRatio', 'BankruptcyHistory', 'CreditScore', 'NetWorth',
            'MonthlyIncome', 'LoanAmount', 'InterestRate', 'PreviousLoanDefaults', 'CreditCardUtilizationRate'
        ]

        def yes_no_encode(value):
            return 1 if value == "Yes" else 0

        with st.form("loan_form"):
            BankruptcyHistory = st.selectbox("Has declared bankruptcy?", ["No", "Yes"])
            PreviousLoanDefaults = st.selectbox("Has previous loan defaults?", ["No", "Yes"])
            CreditCardUtilizationRate = st.number_input("Credit Utilization Rate", 0.01, 1.00, 0.30, 0.01)
            CreditScore = st.number_input("Credit Score", 300, 750, 600)
            DebtToIncomeRatio = st.number_input("Debt-to-Income Ratio", 0.00, 1.00, 0.50, 0.01)
            NetWorth = st.number_input("Net Worth ($)", 1_000.0, 2_603_208.0, 50_000.0, 1_000.0)
            MonthlyIncome = st.number_input("Monthly Income ($)", 1_250.0, 500_000.0, 5_000.0, 500.0)
            LoanAmount = st.number_input("Loan Amount ($)", 1_000.0, 1_000_000.0, 10_000.0, 1_000.0)
            InterestRate = st.number_input("Interest Rate (%)", 0.00, 1.00, 0.40, 0.01)
            RiskScore = st.number_input("Applicant’s Risk Score", 0.0, 100.0, 30.0, 0.01)

            submitted_loan = st.form_submit_button("🔮 Predict Loan Approval")

        if submitted_loan:
            with st.spinner("Analyzing loan application..."):
                input_data = pd.DataFrame([{
                    'RiskScore': RiskScore,
                    'DebtToIncomeRatio': DebtToIncomeRatio,
                    'BankruptcyHistory': yes_no_encode(BankruptcyHistory),
                    'CreditScore': CreditScore,
                    'NetWorth': NetWorth,
                    'MonthlyIncome': MonthlyIncome,
                    'LoanAmount': LoanAmount,
                    'InterestRate': InterestRate,
                    'PreviousLoanDefaults': yes_no_encode(PreviousLoanDefaults),
                    'CreditCardUtilizationRate': CreditCardUtilizationRate
                }])

                df_scaled = pd.DataFrame(scaler.transform(input_data), columns=cols)
                pred = model.predict(df_scaled)[0]
                prob = model.predict_proba(df_scaled)[0][1]

                result_text = "✅ Approved" if pred == 1 else "❌ Denied"
                st.subheader(f"Prediction Result: {result_text}")
                st.metric("Approval Probability", f"{prob:.2%}")

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df_scaled)
                shap_values_used = shap_values[1] if isinstance(shap_values, list) else shap_values

                shap_df = pd.DataFrame({
                    "Feature": df_scaled.columns,
                    "SHAP Value": shap_values_used[0]
                }).sort_values("SHAP Value", ascending=False, key=abs)

                top_features = shap_df.head(3)["Feature"].tolist()
                main_factor = shap_df.iloc[0]["Feature"]

                if pred == 1:
                    reason = (
                        f"The application was approved mainly because of strong {main_factor.lower()}, "
                        f"along with positive effects from {top_features[1].lower()} and {top_features[2].lower()}."
                    )
                else:
                    reason = (
                        f"The application was denied mainly due to a high {main_factor.lower()}, "
                        f"as well as negative impact from {top_features[1].lower()} and {top_features[2].lower()}."
                    )

                st.info(reason)
