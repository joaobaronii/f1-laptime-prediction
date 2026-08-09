import os
import fastf1
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import metrics

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

gps_to_train = ["silverstone", "monaco", "monza", "arabia", "canada"]

for gp in gps_to_train:
    df = get_sessions(2025, gp)

    if df.empty:
        print("Sem dados para {gp}.")
        continue
    
    nan_columns = ["LapStartDate", "PitInTime", "PitOutTime", "Position", "Deleted"]
    df = df.drop(columns=nan_columns, errors='ignore').dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = X_processed.shape[1]
    model = F1LapTimeNet(input_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Treinando modelo de {gp.upper()}...")

    epochs = 150
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval() 
    with torch.no_grad():
        y_pred = model(X_tensor).numpy().flatten()
        y_true = y_tensor.numpy().flatten()

    r2 = metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    max_err = metrics.max_error(y_true, y_pred)

    print(f"--- Métricas de Treino para {gp.upper()} ---")
    print(f"R2 (Coef. de Determinação): {r2:.4f}")
    print(f"MAE (Erro Médio Absoluto):  {mae:.4f} s")
    print(f"MSE (Erro Quadrático Médio):{mse:.4f} s²")
    print(f"Erro Máximo:                {max_err:.4f} s")

    path = f"models/f1_laptime_model_{gp}.pt"
    torch.save({'input_dim': input_dim, 
                'state_dict': model.state_dict(),
                'preprocessor': preprocessor
                }, path)

    print(f"Modelo salvo em {path}.")