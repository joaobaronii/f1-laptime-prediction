# 🏎️ F1 Lap Time Prediction

Aplicação de **Machine Learning** que prevê o **tempo de volta de pilotos da Fórmula 1** com base em dados de telemetria e sessões oficiais, utilizando o pacote [FastF1](https://theoehrly.github.io/Fast-F1/) e rastreamento de experimentos com [MLflow](https://mlflow.org/).

---

## 📖 Visão Geral

O projeto é dividido em duas partes principais:

1. **Treinamento do Modelo (`train.py`)**
   - Coleta e pré-processamento de dados reais de sessões de F1.
   - Treinamento e avaliação de modelos de regressão (ex: Ridge, Random Forest, Gradient Boosting).
   - Registro automático de métricas e artefatos no **MLflow**.

2. **Aplicativo de Predição (`app.py`)**
   - Interface construída em **Streamlit**.
   - Permite selecionar os inputs.
   - Carrega automaticamente o modelo correspondente (via MLflow).
   - Exibe as predições de tempo de volta com base nas variáveis de entrada.

---

## 🧠 Funcionalidades Principais

### Modelo de Machine Learning

- Extração de features de telemetria:
  - `DriverNumber`, `Team`, `Compound`, `TyreLife`, `SpeedI1`, `SpeedI2`, `SpeedFL`, `SpeedST`, `MeanSpeed`
- Alvo: `LapTime` 
- Pipeline com:
  - `OneHotEncoder` para variáveis categóricas
  - Modelo de regressão (Ridge, Regressão Linear, Gradient Boosting ou Random Forest)
- Avaliação com métricas:
  - R², MAE, MSE, e erro máximo
- Registro completo no MLflow (métricas, parâmetros, artefatos, gráficos)

---

## 🖥️ Aplicação Streamlit

A interface gráfica permite ao usuário:

- Selecionar o **Grande Prêmio (GP)** (`silverstone`, `monza`, `arabia`, `canada`, `monaco`)
- Escolher o **piloto** (lista completa de 2025 usando abreviações oficiais)
- Inserir parâmetros adicionais (ex: tipo de pneu, vida útil, velocidades, etc.)
- Visualizar a **predição de tempo de volta**
- Exibir **métricas e gráficos** de desempenho do modelo