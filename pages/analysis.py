import streamlit as st
import pandas as pd
import plotly.express as px

def show_analysis():
    st.title("Financial Risk Analysis for Loan Approval")
    st.write("""
    This page provides an overview of the financial institution’s loan application data.  
    It aims to uncover applicant patterns, approval trends, and potential risk exposure  
    before applying predictive models.
    """)

    # Load dataset
    df = pd.read_csv("dataset/Loan.csv")
    df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'], format='%Y-%m-%d')

    # Overview
    total_applicants = len(df)
    approved_rate = (df['LoanApproved'] == 1).mean() * 100
    denied = (df['LoanApproved'] == 0).mean() * 100
    default_rate = (df['PreviousLoanDefaults'] == 1).mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applicants", total_applicants)
    col2.metric("% Approved", f"{approved_rate:.2f}%")
    col3.metric("% Denied", f"{denied:.2f}%")
    col4.metric("% Default History", f"{default_rate:.2f}%")

    # Loan Applications & Approval Rate Over Time
    st.markdown("<h3 style='color:#ef8b33;'>Loan Applications & Approval Trends Over Time</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    time_col = "ApplicationDate"

    with col1:
        st.markdown("**Applicants Over Time**")
        applicants_over_time = df.groupby(time_col)['LoanApproved'].count().reset_index()
        fig1 = px.line(applicants_over_time, x=time_col, y='LoanApproved', markers=True)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("**Approval Rate Over Time**")
        approval_rate_over_time = df.groupby(time_col)['LoanApproved'].mean().reset_index()
        fig2 = px.line(approval_rate_over_time, x=time_col, y='LoanApproved', markers=True)
        st.plotly_chart(fig2, use_container_width=True)

    # Applicant Profile Distribution
    st.markdown("<h3 style='color:#ef8b33;'>Applicant Profile Distribution</h3>", unsafe_allow_html=True)
    st.write("Explore applicants’ demographics and credit attributes.")

    # Ensure numeric columns are clean
    numeric_cols = ['Age', 'CreditScore', 'DebtToIncomeRatio', 'NetWorth']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Row 1: Age | EmploymentStatus
    col1, col2 = st.columns(2)
    with col1:
        fig_age = px.histogram(df, x='Age', color='LoanApproved', nbins=20, title="Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        fig_emp = px.histogram(df, x='EmploymentStatus', color='LoanApproved', title="Employment Status Distribution")
        st.plotly_chart(fig_emp, use_container_width=True)

    # Row 2: Credit Score | Debt-to-Income | Net Worth
    col1, col2, col3 = st.columns(3)
    with col1:
        fig_cs = px.histogram(df, x='CreditScore', color='LoanApproved', nbins=20, title="Credit Score Distribution")
        st.plotly_chart(fig_cs, use_container_width=True)

    with col2:
        fig_dti = px.histogram(df, x='DebtToIncomeRatio', color='LoanApproved', nbins=20, title="Debt-to-Income Ratio Distribution")
        st.plotly_chart(fig_dti, use_container_width=True)

    with col3:
        fig_net = px.histogram(df, x='NetWorth', color='LoanApproved', nbins=20, title="Net Worth Distribution")
        st.plotly_chart(fig_net, use_container_width=True)

    # Approved Loan Exposure
    st.markdown("<h3 style='color:#ef8b33;'>Approved Loan Exposure</h3>", unsafe_allow_html=True)

    approved = df[df['LoanApproved'] == 1]
    total_amount = approved['LoanAmount'].sum()
    avg_risk = approved['RiskScore'].mean()

    col1, col2 = st.columns(2)
    col1.metric("Total Amount", f"${total_amount:,.0f}")
    col2.metric("Average Risk", f"{avg_risk:.2f}")

    fig_risk = px.bar(
        approved.groupby('RiskScore')['LoanAmount'].sum().reset_index(),
        x='RiskScore',
        y='LoanAmount',
        title="Total Approved Loan Amount by Risk Score",
        labels={'LoanAmount': 'Total Approved Loan ($)', 'RiskScore': 'Risk Score'},
        color='RiskScore'
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # Raw Data Section
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("📄 View Raw Data"):
        st.dataframe(df)
        st.markdown(f"**Data Dimensions:** {df.shape[0]} rows × {df.shape[1]} columns")

    st.caption("Data Source: [Financial Risk for Loan Approval — Kaggle Dataset](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?select=Loan.csv)")
