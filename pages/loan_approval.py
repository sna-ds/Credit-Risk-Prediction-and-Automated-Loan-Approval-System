import streamlit as st

def show_loan_approval():
    st.title("Automated Loan Approval Decision")
    st.write("This module automates the loan approval decision process using a binary classification model. It leverages the previously predicted Risk Score and key financial features to determine whether an applicant should be approved or denied.")