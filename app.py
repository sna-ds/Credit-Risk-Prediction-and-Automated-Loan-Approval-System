import os
import streamlit as st
from streamlit_navigation_bar import st_navbar
import pages as pg

# Page config
st.set_page_config(page_title="Loan Risk Prediction", layout="wide", initial_sidebar_state="collapsed")

# Navigation setup
pages = ["Home", "Analysis", "Risk Scoring", "Loan Approval"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "images", "icons8-home.svg")

styles = {
    "nav": {
        "background-color": "#0d6efd",
        "justify-content": "left",
        "padding": "8px",
    },
    "img": {
        "padding-right": "10px",
    },
    "span": {
        "color": "white",
        "padding": "10px",
        "font-weight": "500",
    },
    "active": {
        "background-color": "white",
        "color": "#1559be",
        "font-weight": "600",
        "padding": "10px",
        "border-radius": "8px",
    }
}

options = {
    "show_menu": False,
    "show_sidebar": False,
}

page = st_navbar(
    pages,
    logo_path=logo_path,
    styles=styles,
    options=options,
)

# Page routing
if page == "Home":
    pg.home.show_home()
elif page == "Analysis":
    pg.overview.show_overview()
elif page == "Risk Scoring":
    pg.risk_scoring.show_risk_scoring()
elif page == "Loan Approval":
    pg.loan_approval.show_loan_approval()
