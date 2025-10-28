import streamlit as st

drivers_numbers = {
    "VER": 1,
    "NOR": 4,
    "BOR": 5,
    "HAD": 6,
    "GAS": 10,
    "ANT": 12,
    "ALO": 14,
    "LEC": 16,
    "STR": 18,
    "TSU": 22,
    "ALB": 23,
    "HUL": 27,
    "LAW": 30,
    "OCO": 31,
    "COL": 43,
    "HAM": 44,
    "SAI": 55,
    "RUS": 63,
    "PIA": 81,
    "BEA": 87,
}

st.set_page_config(page_title="F1 Prediction", page_icon="🏎️", layout="wide")

st.markdown("""
# F1 Timelap Prediction App
""")