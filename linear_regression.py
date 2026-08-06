import os
import joblib
import fastf1
import pandas as pd
from feature_engine import encoding
from sklearn import linear_model, pipeline

fastf1.Cache.enable_cache("data/raw")

categorical_features = ["DriverNumber", "Team", "Compound"]
numerical_features = ["TyreLife", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "MeanSpeed"]
features = categorical_features + numerical_features
target = "LapTime"

def get_sessions(year, gp):
    print(f"Baixando dados de {gp.upper()}...")
    sessions_list = []
    
    for session_name in ["FP1", "FP2", "FP3", "Q"]:
        try:
            session = fastf1.get_session(year, gp, session_name)
            session.load(telemetry=False, weather=False, messages=False)
            quick_laps = session.laps.pick_quicklaps()
            sessions_list.append(quick_laps)
        except Exception as e:
            pass

    if not sessions_list:
        return pd.DataFrame()

    sessions = pd.concat(sessions_list).reset_index(drop=True)
    sessions["LapTime"] = sessions["LapTime"].dt.total_seconds()
    sessions["MeanSpeed"] = (
        sessions["SpeedI1"] + sessions["SpeedI2"] + sessions["SpeedFL"] + sessions["SpeedST"]
    ) / 4
    return sessions

os.makedirs("models", exist_ok=True)

gps_to_train = ["silverstone", "monaco", "monza", "arabia", "canada"]

for gp in gps_to_train:
    df = get_sessions(2025, gp)
    
    if df.empty:
        print(f"Sem dados para {gp}. Pulando...")
        continue

    nan_columns = ["LapStartDate", "PitInTime", "PitOutTime", "Position", "Deleted"]
    df = df.drop(columns=nan_columns, errors='ignore').dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    onehot = encoding.OneHotEncoder(variables=categorical_features, ignore_format=True)
    model = linear_model.RidgeCV(cv=3)
    model_pipeline = pipeline.Pipeline(steps=[("Onehot", onehot), ("Model", model)])

    print(f"Treinando modelo de {gp.upper()}...")
    model_pipeline.fit(X, y)
    
    caminho_arquivo = f"models/laptime_{gp}.joblib"
    joblib.dump(model_pipeline, caminho_arquivo)
    print(f"✅ Modelo salvo em: {caminho_arquivo}\n")