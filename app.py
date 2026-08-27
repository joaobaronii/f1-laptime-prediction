import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import os
import warnings

warnings.filterwarnings('ignore')

class F1LapTimeNet(nn.Module):
    def __init__(self, input_size):
        super(F1LapTimeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

drivers_numbers = {
    "VER": 1, "NOR": 4, "BOR": 5, "HAD": 6, "GAS": 10, "ANT": 12, "ALO": 14,
    "LEC": 16, "STR": 18, "TSU": 22, "ALB": 23, "HUL": 27, "LAW": 30, "OCO": 31,
    "COL": 43, "HAM": 44, "SAI": 55, "RUS": 63, "PIA": 81, "BEA": 87,
}

drivers = list(drivers_numbers.keys())

teams = [
    "Red Bull Racing", "Aston Martin", "Kick Sauber", "Mercedes", "Williams",
    "McLaren", "Racing Bulls", "Alpine", "Ferrari", "Haas F1 Team",
]

gps = ["silverstone", "monaco", "monza", "arabia", "canada"]

pole_laps = {
    "silverstone": 84.892,
    "monza": 78.792,
    "arabia": 87.294,
    "canada": 70.899,
    "monaco": 69.954
}

st.set_page_config(page_title="F1 Prediction", page_icon="🏎️", layout="wide")

st.markdown("""
# 🏁 F1 2025 Laptime Prediction App
## Inputs            
""")

with st.container():
    gp = st.selectbox("GP", gps)
    pole_at_gp = pole_laps[gp]
    pole_minutes = int(pole_at_gp // 60)
    pole_seconds = pole_at_gp % 60
    st.info(f"Pole lap time: {pole_minutes}:{pole_seconds:06.3f}")

col11, col12 = st.columns(2)
with col11.container(border=True):
    driver = st.selectbox("Driver", drivers)
with col12.container(border=True):
    team = st.selectbox("Team/Car", teams)

col21, col22 = st.columns(2)
with col21.container(border=True):
    compound = st.selectbox("Compound", ["HARD", "MEDIUM", "SOFT"])
with col22.container(border=True):
    tyrelife = st.slider("Tyrelife (Laps)", min_value=2.0, max_value=20.0, step=1.0)

col31, col32, col33, col34 = st.columns(4)
with col31.container(border=True):
    speedi1 = st.slider("SpeedI1 (Km/h)", min_value=220.0, max_value=360.0, step=1.0)
with col32.container(border=True):
    speedi2 = st.slider("SpeedI2 (Km/h)", min_value=220.0, max_value=360.0, step=1.0)
with col33.container(border=True):
    speedfl = st.slider("SpeedFL (Km/h)", min_value=220.0, max_value=360.0, step=1.0)
with col34.container(border=True):
    speedst = st.slider("SpeedST (Km/h)", min_value=220.0, max_value=360.0, step=1.0)

meanspeed = (speedi1 + speedi2 + speedfl + speedst) / 4

if st.button("Predict"):
    driver_num = str(drivers_numbers[driver])
    
    caminho_modelo = f"models/f1_laptime_model_{gp}.pt"
    if not os.path.exists(caminho_modelo):
        caminho_modelo = f"f1_laptime_model_{gp}.pt"
    
    if not os.path.exists(caminho_modelo):
        st.error(f"Modelo para {gp.upper()} não encontrado. Rode o script de treinamento primeiro!")
        st.stop()

    try:
        with st.spinner("Loading PyTorch model..."):
            checkpoint = torch.load(caminho_modelo, weights_only=False, map_location=torch.device('cpu'))
            
            preprocessor = checkpoint['preprocessor']
            input_dim = checkpoint['input_dim']
            
            model = F1LapTimeNet(input_dim)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()

        feature_dict = {
            "DriverNumber": [driver_num],
            "Team": [team],
            "Compound": [compound],
            "TyreLife": [tyrelife],
            "SpeedI1": [speedi1],
            "SpeedI2": [speedi2],
            "SpeedFL": [speedfl],
            "SpeedST": [speedst],
            "MeanSpeed": [meanspeed]
        }    
        input_df = pd.DataFrame(feature_dict)

        with st.spinner("Predicting..."):
            X_processed = preprocessor.transform(input_df)
            
            X_tensor = torch.tensor(X_processed, dtype=torch.float32)

            with torch.no_grad():
                prediction = model(X_tensor)
                laptime_seconds = prediction.item() 

            minutes = int(laptime_seconds // 60)
            seconds = laptime_seconds % 60
            delta_seconds = laptime_seconds - pole_at_gp

            st.metric(
                "Tempo de Volta Previsto", 
                f"{minutes}:{seconds:06.3f}", 
                delta=f"{delta_seconds:+.3f}s", 
                delta_color="inverse"
            )
        
        with st.expander("Show inputs (Raw)"):
            st.dataframe(input_df)
            
    except Exception as e:
        st.error(f"Error during the prediction: {e}")