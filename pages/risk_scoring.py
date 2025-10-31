import streamlit as st

def show_risk_scoring():
    st.title("Risk Score Prediction")
    st.write("This module predicts an applicant’s Risk Score, representing their likelihood of loan default or financial instability. The score ranges from 0 (Low Risk) to 100 (High Risk), based on the applicant’s financial behavior.")