import streamlit as st
import pandas as pd
import altair as alt

def show_analysis():
    st.title("Financial Risk Analysis for Loan Approval")

    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/sna-ds/Credit-Risk-Prediction-and-Automated-Loan-Approval-System/main/dataset/Loan.csv"
        df = pd.read_csv(url)
        df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'], errors='coerce')
        df = df.dropna(subset=['ApplicationDate'])

        # Feature engineering
        df['LoanStatus'] = df['LoanApproved'].map({0: 'Denied', 1: 'Approved'})
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0,25,35,45,55,65,120], labels=['<25', '25–34', '35–44', '45–54', '55–64', '65+'], right=False)
        df['CreditScoreGroup'] = pd.cut(df['CreditScore'], bins=[340, 430, 520, 600, 670, 720], labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], right=False)
        df['DTIGroup'] = pd.cut(df['DebtToIncomeRatio'], bins=[0, 0.2, 0.35, 0.5, 1.0], labels=['Low (<20%)', 'Moderate (20–35%)', 'High (35–50%)', 'Critical (>50%)'], right=False)
        df['MonthlyIncomeGroup'] = pd.cut(df['MonthlyIncome'], bins=[0, 3000, 7000, 12000, 20000, 30000], labels=['Low (<3K)', 'Lower Middle (3–7K)', 'Middle (7–12K)', 'Upper Middle (12–20K)', 'High (>20K)'], right=False)
        return df

    df = load_data()

    total_applicants = len(df)
    approved_rate = (df['LoanApproved'] == 1).mean() * 100
    denied = (df['LoanApproved'] == 0).mean() * 100
    default_rate = (df['PreviousLoanDefaults'] == 1).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applicants", total_applicants)
    c2.metric("% Approved", f"{approved_rate:.2f}%")
    c3.metric("% Denied", f"{denied:.2f}%")
    c4.metric("% Default History", f"{default_rate:.2f}%")

    color_scale = alt.Scale(domain=["Denied", "Approved"], range=["#FF5E0E", "#4169E1"])

    st.markdown("<h3 style='color:#ef8b33;'>📈 Loan Applications Over Time</h3>", unsafe_allow_html=True)
    app_year = (
        df.groupby([df['ApplicationDate'].dt.year.rename('Year'), 'LoanStatus'])
        .size()
        .reset_index(name='Applicants')
    )

    chart_app = (
        alt.Chart(app_year)
        .mark_line(point=True)
        .encode(
            x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Applicants:Q', title='Number of Applicants'),
            color=alt.Color('LoanStatus:N', scale=color_scale, title='Loan Status')
        )
        .properties(height=400)
    )
    st.altair_chart(chart_app, use_container_width=True)

    # Applicant Profile Distribution 
    st.markdown("<h3 style='color:#ef8b33;'>Applicant Profile</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Age Distribution
    with col1:
        st.markdown("**Age Distribution**")
        chart_age = (
            alt.Chart(df)
            .mark_bar(opacity=0.85, color="#4169E1")
            .encode(
                x=alt.X('AgeGroup:N', title='Age', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('count()', title='Applicants')
            )
            .properties(height=400)
        )
        st.altair_chart(chart_age, use_container_width=True)


    # Employment Status
    with col2:
        st.markdown("**Employment Status Distribution**")
        chart_emp = (
            alt.Chart(df)
            .mark_bar(opacity=0.8, color="#4169E1")
            .encode(
                x=alt.X('EmploymentStatus:N', title='Employment Status', sort='-y', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('count()', title='Applicants')
            )
            .properties(height=400)
        )
        st.altair_chart(chart_emp, use_container_width=True)

    # Credit, DTI, Monthly Income
    col3, col4, col5 = st.columns(3)

    # Credit Score
    with col3:
        st.markdown("**Credit Score Distribution**")
        chart_credit = (
            alt.Chart(df)
            .mark_bar(opacity=0.8, color="#4169E1")
            .encode(
                x=alt.X('CreditScoreGroup:N', title='Credit Score', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('count()', title='Applicants')
            )
            .properties(height=400)
        )
        st.altair_chart(chart_credit, use_container_width=True)

    # DTI
    with col4:
        st.markdown("**Debt-to-Income Ratio Distribution**")
        chart_dti = (
            alt.Chart(df)
            .mark_bar(opacity=0.8, color="#4169E1")
            .encode(
                x=alt.X('DTIGroup:N', title='Ratio', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('count()', title='Applicants')
            )
            .properties(height=400)
        )
        st.altair_chart(chart_dti, use_container_width=True)

    # Monthly Income
    with col5:
        chart_mon = (
            alt.Chart(df)
            .mark_bar(opacity=0.8, color="#4169E1")
            .encode(
                x=alt.X('MonthlyIncomeGroup:N', title='Income Range', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('count()', title='Applicants')
            )
            .properties(height=400)
        )
        st.altair_chart(chart_mon, use_container_width=True)

    # Approved Loan Exposure
    st.markdown("<h3 style='color:#ef8b33;'>Approved Loan Exposure</h3>", unsafe_allow_html=True)
    approved = df[df['LoanApproved'] == 1].copy()
    total_amount = approved['LoanAmount'].sum()
    avg_risk = approved['RiskScore'].mean()

    col1, col2 = st.columns(2)
    col1.metric("Total Amount", f"${total_amount:,.0f}")
    col2.metric("Average Risk", f"{avg_risk:.2f}")

    exposure = approved.groupby('RiskScore')['LoanAmount'].sum().reset_index()
    chart_exposure = (
        alt.Chart(exposure)
        .mark_bar(opacity=0.8, color="#4169E1")
        .encode(
            x=alt.X('RiskScore:Q', title='Risk Score'),
            y=alt.Y('LoanAmount:Q', title='Total Approved Loan ($)')
        )
        .properties(height=400)
    )
    st.altair_chart(chart_exposure, use_container_width=True)

    st.markdown("---")
    with st.expander("View Raw Data"):
        st.dataframe(df)
        st.markdown(f"**Data Dimensions:** {df.shape[0]} rows × {df.shape[1]} columns")

    st.caption("Data Source: [Financial Risk for Loan Approval — Kaggle Dataset](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?select=Loan.csv)")