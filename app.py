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

teams = [
    "Red Bull Racing",
    "Aston Martin",
    "Kick Sauber",
    "Mercedes",
    "Williams",
    "McLaren",
    "Racing Bulls",
    "Alpine",
    "Ferrari",
    "Haas F1 Team",
]

gps = ["silverstone", "monaco", "monza", "arabia", "canada"]

mlflow.set_tracking_uri("http://localhost:5000")


st.set_page_config(page_title="F1 Prediction", page_icon="🏎️", layout="wide")

st.markdown("""
# F1 2025 Laptime Prediction App
""")

with st.container():
    gp = st.selectbox("GP", gps)

col11, col12 = st.columns(2)

with col11.container(border=True):
    driver = st.selectbox("Driver", drivers)

with col12.container(border=True):
    team = st.selectbox("Team/Car", teams)


col21, col22 = st.columns(2)

with col21.container(border=True):
    compound = st.selectbox("Compound", ["HARD", "MEDIUM", "SOFT"])
    
with col22.container(border=True):
    tyrelife = st.number_input("Tyrelife", min_value=2.0, max_value=20.0, step=1.0)


col31, col32, col33, col34 = st.columns(4)

with col31.container(border=True):
    speedi1 = st.number_input("SpeedI1", min_value=200.0, max_value=350.0, step=1.0)

with col32.container(border=True):
    speedi2 = st.number_input("SpeedI2", min_value=200.0, max_value=350.0, step=1.0)

with col33.container(border=True):
    speedfl = st.number_input("SpeedFL", min_value=200.0, max_value=350.0, step=1.0)

with col34.container(border=True):
    speedst = st.number_input("SpeedST", min_value=200.0, max_value=350.0, step=1.0)

meanspeed = (speedi1 + speedi2 + speedfl + speedst) / 4


