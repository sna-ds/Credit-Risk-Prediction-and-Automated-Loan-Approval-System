import streamlit as st
import pandas as pd
import plotly.express as px

def show_analysis():
    st.title("Financial Risk Analysis for Loan Approval")
    
    # delete later
    st.write("This page provides an overview of the financial institution’s loan application data. It aims to uncover applicant patterns, approval trends, and potential risk exposure before applying predictive models.")
    
    # Load dataset
    df = pd.read_csv("/workspaces/Credit-Risk-Prediction-and-Automated-Loan-Approval-System/dataset/Loan.csv")
    df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'], format='%Y-%m-%d')
    
    # KPIs
    total_applicants = len(df)
    approved = (df['LoanApproved'] == 1).mean() * 100
    denied = (df['LoanApproved'] == 0).mean() * 100
    default_rate = (df['PreviousLoanDefaults'] == 1).mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applicants", total_applicants)
    col2.metric("% Approved", f"{approved:.2f}%")
    col3.metric("% Denied", f"{denied:.2f}%")
    col4.metric("% Default History", f"{default_rate:.2f}%")

    st.markdown("### Loan Applications & Approval Rate Over Time")
    time_col = "ApplicationDate"  
    df[time_col] = pd.to_datetime(df[time_col])
    applicants_over_time = df.groupby(time_col)['LoanApproved'].count().reset_index()
    approval_rate_over_time = df.groupby(time_col)['LoanApproved'].mean().reset_index()

    fig1 = px.line(applicants_over_time, x=time_col, y='LoanApproved', title="Applicants Over Time")
    fig2 = px.line(approval_rate_over_time, x=time_col, y='LoanApproved', title="Approval Rate Over Time")
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

    st.markdown("### Applicant Profile Distribution")
    st.write("Explore applicants’ demographics and credit attributes.")
    numeric_cols = ['Age', 'CreditScore', 'DebtToIncomeRatio', 'NetWorth']
    for col in numeric_cols:
        st.plotly_chart(px.histogram(df, x=col, color='LoanApproved', title=f"{col} Distribution"))

    st.markdown("### Approved Loan Exposure")
    total_amount = df[df['LoanApproved']==1]['LoanAmount'].sum()
    avg_risk = df[df['LoanApproved']==1]['RiskScore'].mean()
    col1, col2 = st.columns(2)
    col1.metric("Total Approved Loan Amount", f"${total_amount:,.0f}")
    col2.metric("Average Risk Score (Approved)", f"{avg_risk:.2f}")

    approved = df[df['LoanApproved'] == 1]

    fig = px.bar(
        approved.groupby('RiskTier')['LoanAmount'].sum().reset_index(),
        x='RiskTier',
        y='LoanAmount',
        title="Total Approved Loan Amount by Risk Tier",
        labels={'LoanAmount': 'Total Approved Loan ($)', 'RiskTier': 'Risk Tier'},
        color='RiskTier'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    with st.expander("View Raw Data"):
        st.dataframe(df)
        st.markdown(f"**Data Dimensions:** {df.shape[0]} rows × {df.shape[1]} columns")

st.write("Data Source: [Financial Risk for Loan Approval — Kaggle Dataset](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?select=Loan.csv)")
    


