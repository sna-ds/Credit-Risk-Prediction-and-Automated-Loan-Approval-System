import streamlit as st
import pandas as pd
import plotly.express as px

def show_analysis():
    # Title
    st.title("Financial Risk Analysis for Loan Approval")
    st.write("""
    This page provides an overview of the financial institution’s loan application data.  
    It aims to uncover applicant patterns, approval trends, and potential risk exposure  
    before applying predictive models.
    """)

    # Load dataset
    df = pd.read_csv("dataset/Loan.csv")
    df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'], format='%Y-%m-%d', errors='coerce')

    # Convert numeric columns to proper types
    numeric_cols = ['Age', 'CreditScore', 'DebtToIncomeRatio', 'NetWorth', 'LoanAmount', 'RiskScore']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # KPIs
    total_applicants = len(df)
    approved_rate = (df['LoanApproved'] == 1).mean() * 100
    denied = (df['LoanApproved'] == 0).mean() * 100
    default_rate = (df['PreviousLoanDefaults'] == 1).mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applicants", total_applicants)
    col2.metric("% Approved", f"{approved_rate:.2f}%")
    col3.metric("% Denied", f"{denied:.2f}%")
    col4.metric("% Default History", f"{default_rate:.2f}%")

    # Loan Applications Over Time
    st.markdown("<h3 style='color:#ef8b33;'>📈 Loan Applications Over Time</h3>", unsafe_allow_html=True)
    time_col = "ApplicationDate"
    applicants_over_time = df.groupby(time_col)['LoanApproved'].count().reset_index()
    fig1 = px.line(
        applicants_over_time,
        x=time_col,
        y='LoanApproved',
        title="Applicants Over Time",
        markers=True
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Applicant Profile Distribution
    st.markdown("<h3 style='color:#ef8b33;'>Applicant Profile Distribution</h3>", unsafe_allow_html=True)

    # Create columns for side-by-side charts
    col1, col2 = st.columns(2)

    # Age Distribution
    with col1:
        fig_age = px.histogram(
            df,
            x='Age',
            color='LoanApproved',
            title="Age Distribution",
            barmode='overlay'
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # Employment Status Distribution
    with col2:
        fig_emp = px.histogram(
            df,
            x='EmploymentStatus',
            color='LoanApproved',
            title="Employment Status Distribution"
        )
        st.plotly_chart(fig_emp, use_container_width=True)

    # Next row of numerical charts
    col3, col4, col5 = st.columns(3)

    with col3:
        fig_credit = px.histogram(
            df,
            x='CreditScore',
            color='LoanApproved',
            title="Credit Score Distribution",
            barmode='overlay'
        )
        st.plotly_chart(fig_credit, use_container_width=True)

    with col4:
        fig_debt = px.histogram(
            df,
            x='DebtToIncomeRatio',
            color='LoanApproved',
            title="Debt to Income Ratio Distribution",
            barmode='overlay'
        )
        st.plotly_chart(fig_debt, use_container_width=True)

    with col5:
        fig_networth = px.histogram(
            df,
            x='NetWorth',
            color='LoanApproved',
            title="Net Worth Distribution",
            barmode='overlay'
        )
        st.plotly_chart(fig_networth, use_container_width=True)

    # Approved Loan Exposure
    st.markdown("---")
    st.markdown("<h3 style='color:#ef8b33;'>Approved Loan Exposure</h3>", unsafe_allow_html=True)
    approved = df[df['LoanApproved'] == 1]
    total_amount = approved['LoanAmount'].sum()
    avg_risk = approved['RiskScore'].mean()

    col1, col2 = st.columns(2)
    col1.metric("Total Amount", f"${total_amount:,.0f}")
    col2.metric("Average Risk", f"{avg_risk:.2f}")

    # Bar chart for total amount by risk score
    exposure = approved.groupby('RiskScore')['LoanAmount'].sum().reset_index()
    fig = px.bar(
        exposure,
        x='RiskScore',
        y='LoanAmount',
        title="Total Approved Loan Amount by Risk Score",
        labels={'LoanAmount': 'Total Approved Loan ($)', 'RiskScore': 'Risk Score'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data Preview
    st.markdown("---")
    with st.expander("View Raw Data"):
        st.dataframe(df)
        st.markdown(f"**Data Dimensions:** {df.shape[0]} rows × {df.shape[1]} columns")

    st.caption("Data Source: [Financial Risk for Loan Approval — Kaggle Dataset](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?select=Loan.csv)")
