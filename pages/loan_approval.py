import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

def show_loan_approval():
    st.title("Automated Loan Approval Decision")
    st.markdown("""
    Input applicant details below to predict whether the loan will be **approved** or **denied**, 
    based on financial and risk indicators.
    """)

    # Load with joblib
    model = joblib.load('models/lgbm_classifier.pkl')
    scaler_dict = joblib.load('models/clf_scaler.pkl')


    cols = ['RiskScore', 'DebtToIncomeRatio', 'BankruptcyHistory', 'CreditScore', 'NetWorth',
    'MonthlyIncome', 'LoanAmount', 'InterestRate', 'PreviousLoanDefaults', 'CreditCardUtilizationRate']

    def yes_no_encode(value):
        return 1 if value == "Yes" else 0

    # Use form for cleaner UX
    with st.form("loan_form"):
        st.subheader("Enter Applicant Information")

        BankruptcyHistory = st.selectbox("Has the applicant ever declared bankruptcy?", ["No", "Yes"])
        PreviousLoanDefaults = st.selectbox("Has the applicant ever defaulted on a previous loan?", ["No", "Yes"])
        CreditCardUtilizationRate = st.number_input("What percentage of the applicant’s credit limit is currently used?", 0.01, 1.00, 0.30, 0.01)
        CreditScore = st.number_input("What is the applicant’s current credit score (from credit bureau)?", 300, 750, 600)
        DebtToIncomeRatio = st.number_input("What is the ratio of total debt payments to monthly income (Debt to Income Ratio)?", 0.00, 1.00, 0.50, 0.01)
        NetWorth = st.number_input("What is the applicant’s total ($) net worth (assets minus liabilities)?", 1_000.0, 2_603_208.0, 50_000.0, 1_000.0)
        MonthlyIncome = st.number_input("What is the applicant’s average monthly income ($)?", 1_250.0, 500_000.0, 5_000.0, 500.0)
        LoanAmount = st.number_input("How much money is the applicant requesting for this loan ($)?", 1_000.0, 1_000_000.0, 10_000.0, 1_000.0)
        InterestRate = st.number_input("What is the interest rate (%) offered for this loan?", 0.00, 1.00, 0.40, 0.01)
        RiskScore = st.number_input("What is the applicant’s current credit risk score?", 0, 100.00, 30.00, 0.01)

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        with st.spinner("Running prediction..."):
            # Create input dataframe
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

            # Scale numeric values
            df_scaled = pd.DataFrame(scaler_dict.transform(input_data), columns=cols)

            # Predict
            pred = model.predict(df_scaled)[0]
            prob = model.predict_proba(df_scaled)[0][1]

            result_text = "✅ Approved" if pred == 1 else "❌ Denied"
            st.subheader(f"Prediction Result: {result_text}")
            st.metric("Approval Probability", f"{prob:.2%}")

            # SHAP Explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_scaled)

            if isinstance(shap_values, list):
                shap_values_used = shap_values[1]  # Class 1 (Approved)
            else:
                shap_values_used = shap_values     # Single output

            # Build SHAP DataFrame for this applicant
            shap_df = pd.DataFrame({
                "Feature": df_scaled.columns,
                "SHAP Value": shap_values_used[0]
            }).sort_values("SHAP Value", ascending=False, key=abs)

            # Extract top features driving the prediction
            top_features = shap_df.head(3)["Feature"].tolist()
            main_factor = shap_df.iloc[0]["Feature"]

            # Interpretation
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

