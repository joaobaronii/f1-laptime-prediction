# %%
import fastf1
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from feature_engine import encoding
from sklearn import ensemble, linear_model, metrics, model_selection, pipeline

fastf1.Cache.enable_cache("data/raw")

features = [
    "DriverNumber",
    "Team",
    "Compound",
    "TyreLife",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "MeanSpeed",
]
target = "LapTime"

def get_sessions(year, gp):
    fp1 = fastf1.get_session(year, gp, "FP1")
    fp1.load(telemetry=False, weather=False, messages=False)
    fp1_laps = fp1.laps.pick_quicklaps()

    fp2 = fastf1.get_session(year, gp, "FP2")
    fp2.load(telemetry=False, weather=False, messages=False)
    fp2_laps = fp2.laps.pick_quicklaps()

    fp3 = fastf1.get_session(year, gp, "FP3")
    fp3.load(telemetry=False, weather=False, messages=False)
    fp3_laps = fp3.laps.pick_quicklaps()

    qualy = fastf1.get_session(year, gp, "Q")
    qualy.load(telemetry=False, weather=False, messages=False)
    qualy_laps = qualy.laps.pick_quicklaps()

    sessions = pd.concat([fp1_laps, fp2_laps, fp3_laps, qualy_laps])

    sessions["LapTime"] = sessions["LapTime"].dt.total_seconds()

    sessions["MeanSpeed"] = (
        sessions["SpeedI1"]
        + sessions["SpeedI2"]
        + sessions["SpeedFL"]
        + sessions["SpeedST"]
    ) / 4

    sessions = sessions.reset_index(drop=True)

    return sessions


df = get_sessions(2025, "silverstone")

# %%
nan_columns = ["LapStartDate", "PitInTime", "PitOutTime", "Position", "Deleted"]

df = df.drop(columns=nan_columns)

df = df.dropna()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, random_state=42, test_size=0.2, train_size=0.8
)


onehot = encoding.OneHotEncoder(variables=features, ignore_format=True)

# %%

# REGRESSÃO
# model = linear_model.LinearRegression()

# RIDGE
model = linear_model.RidgeCV(cv=3)

# RANDOM FOREST
# grid_params = {
#     "n_estimators": [100, 200, 500],
#     "min_samples_leaf": [2, 5, 10, 20],
#     "max_depth": [None, 10, 20],
# }
# rf = ensemble.RandomForestRegressor(random_state=42, n_jobs=2)
# model = model_selection.GridSearchCV(rf, grid_params, cv=3, scoring="r2", verbose=2)

# GRADIENT BOOST
# grid_params = {
#     'n_estimators':[100, 200, 500],
#     'learning_rate':[0.01, 0.1, 0.2],
#     'max_depth':[5, 10, 15],
#     'subsample':[0.8, 1],

# }
# gb = ensemble.GradientBoostingRegressor(random_state=42)
# model = model_selection.GridSearchCV(gb, grid_params, cv=3, scoring='r2', verbose = 2)

model_pipeline = pipeline.Pipeline(steps=[("Onehot", onehot), ("Model", model)])

# %%

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_id="309155165905550830")

# %%
with mlflow.start_run():
    mlflow.sklearn.autolog()

    model_pipeline.fit(X_train, y_train)

    predict_train = model_pipeline.predict(X_train)

    r2_train = metrics.r2_score(y_train, predict_train)
    mae_train = metrics.mean_absolute_error(y_train, predict_train)
    mse_train = metrics.mean_squared_error(y_train, predict_train)
    max_error_train = metrics.max_error(y_train, predict_train)

    predict_test = model_pipeline.predict(X_test)

    r2_test = metrics.r2_score(y_test, predict_test)
    mae_test = metrics.mean_absolute_error(y_test, predict_test)
    mse_test = metrics.mean_squared_error(y_test, predict_test)
    max_error_test = metrics.max_error(y_test, predict_test)

    mlflow.log_metrics(
        {
            "R2_train": r2_train,
            "MAE_train": mae_train,
            "MSE_train": mse_train,
            "max_error_train": max_error_train,
            "R2_test": r2_test,
            "MAE_test": mae_test,
            "MSE_test": mse_test,
            "max_error_test": max_error_test,
        }
    )
# %%
print(f"Coeficiente de determinação (R2) treino: {r2_train}")
print(f"Erro médio absoluto(MAE) treino: {mae_train}")
print(f"Erro quadático médio(MSE) treino: {mse_train}")
print(f"Erro máximo treino: {max_error_train}")

print(f"Coeficiente de determinação (R2) teste: {r2_test}")
print(f"Erro médio absoluto(MAE) teste: {mae_test}")
print(f"Erro quadático médio(MSE) teste: {mse_test}")
print(f"Erro máximo teste: {max_error_test}")

# %%
plt.subplot(2, 2, 1)
plt.scatter(y_train, predict_train, alpha=0.6)
plt.xlabel("Tempo real (s)")
plt.ylabel("Tempo predict (s)")
plt.title("Predição tempo de volta - treino")
plt.grid(True)

plt.subplot(2, 2, 2)
erros_train = y_train - predict_train
plt.hist(erros_train)
plt.title("Distribuição dos erros - treino")
plt.xlabel("Erros (s)")
plt.ylabel("Frequência")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.scatter(y_test, predict_test, alpha=0.6)
plt.xlabel("Tempo real (s)")
plt.ylabel("Tempo predict (s)")
plt.title("Predição tempo de volta - teste")
plt.grid(True)

plt.subplot(2, 2, 4)
erros_test = y_test - predict_test
plt.hist(erros_test)
plt.title("Distribuição dos erros - teste")
plt.xlabel("Erros (s)")
plt.ylabel("Frequência")
plt.grid(True)

plt.show()
# %%
