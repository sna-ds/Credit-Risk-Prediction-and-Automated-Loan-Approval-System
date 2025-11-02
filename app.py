import os
import streamlit as st
from streamlit_navigation_bar import st_navbar
import pages as pg

# Page config
st.set_page_config(
    page_title="Loan Risk Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Navigation setup
pages = ["Home", "Analysis", "Prediction", "Contact"]

# Path for logo
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "images", "icons8-home.svg")

styles = {
    "nav": {
        "background-color": "royalblue",
        "justify-content": "left",
    },
    "img": {
        "padding-right": "14px",
    },
    "span": {
        "color": "white",
        "padding": "14px",
    },
    "active": {
        "background-color": "white",
        "color": "royalblue",
        "font-weight": "bold",
        "padding": "14px",
        "border-radius": "6px",
    },
}

# Navbar setup
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

# Routing
if page == "Home":
    pg.home.show_home()
elif page == "Analysis":
    pg.analysis.show_analysis()
elif page == "Prediction":
    pg.prediction.show_prediction()
elif page == "Contact":
    pg.contact.show_contact()
