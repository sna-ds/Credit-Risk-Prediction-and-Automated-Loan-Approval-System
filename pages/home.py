import streamlit as st

def show_home():
    st.title("Financial Risk and Loan Approval Prediction")
    st.markdown("""
        👋 Welcome! 
            
        This Streamlit app helps financial institutions assess **credit risk** and automate **loan approval decisions** using machine learning models.

        Built to support data-driven decision-making, this app enables users to: 
        - Predict **borrower risk scores** accurately.  
        - Simulate **loan approval outcomes** instantly.  
        - Visualize **portfolio insights** through interactive dashboards.

    """)

    st.divider()
    st.subheader("Why This Project Matters?")
    st.write("""
        Banks and lenders must balance profitability and risk.
        By leveraging predictive analytics, this tool helps to:
        - Reduce default rates through data-driven applicant screening
        - Monitor and optimize credit portfolio health over time
    """)
