import streamlit as st
import mlflow

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

model_version = {
    "silverstone": 1,
    "monza": 2,
    "arabia": 3,
    "canada": 4,
    "monaco": 5,
}

drivers = [
    "VER",
    "NOR",
    "BOR",
    "HAD",
    "GAS",
    "ANT",
    "ALO",
    "LEC",
    "STR",
    "TSU",
    "ALB",
    "HUL",
    "LAW",
    "OCO",
    "COL",
    "HAM",
    "SAI",
    "RUS",
    "PIA",
    "BEA",
]

gps = ["silverstone", "monaco", "monza", "arabia", "canada"]

mlflow.set_tracking_uri("http://localhost:5000")


st.set_page_config(page_title="F1 Prediction", page_icon="🏎️", layout="wide")

st.markdown("""
# F1 Timelap Prediction App
""")

col1, col2 = st.columns(2)

with col1.container(border=True):
    gp = st.selectbox("GP", gps)


with col2.container(border=True):
    driver = st.selectbox("Driver", drivers)
