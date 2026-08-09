# 🏎️ F1 Lap Time Prediction

Aplicação de **Machine Learning** que prevê o **tempo de volta de pilotos da Fórmula 1** com base em dados de telemetria e sessões oficiais, utilizando o pacote [FastF1](https://theoehrly.github.io/Fast-F1/).

## 📖 Visão Geral

O projeto é composto por:

1. **Treinamento de Modelos**
   - `linear_regression.py`: Treina modelos Ridge Regression por GP
   - `neural_network.py`: Treina redes neurais (PyTorch) por GP
   - Coleta e pré-processamento de dados reais de sessões de F1 (2025)
   - Features: `DriverNumber`, `Team`, `Compound`, `TyreLife`, `SpeedI1`, `SpeedI2`, `SpeedFL`, `SpeedST`, `MeanSpeed`
   - Alvo: `LapTime` (segundos)

2. **Aplicativo de Predição (`app.py`)**
   - Interface construída em **Streamlit**
   - Carrega modelos treinados (joblib para Ridge, .pt para PyTorch)
   - Permite selecionar GP, piloto, equipe, composto, vida do pneu, velocidades
   - Exibe predição de tempo de volta e delta em relação ao tempo da pole position

## 🧠 Funcionalidades Principais

### Modelos de Machine Learning

- **Ridge Regression** (`linear_regression.py`): Pipeline com OneHotEncoder + RidgeCV
- **Neural Network** (`neural_network.py`): MLP com 2 hidden layers (64, 32), Dropout, ReLU
- Pré-processamento:
  - OneHotEncoder para variáveis categóricas (DriverNumber, Team, Compound)
  - StandardScaler para variáveis numéricas (apenas NN)
- Dados de treino: Sessões FP1, FP2, FP3, Q de 2025 (quicklaps apenas)

### Aplicação Streamlit

A interface gráfica permite ao usuário:

- Selecionar o **Grande Prêmio** (`silverstone`, `monaco`, `monza`, `arabia`, `canada`)
- Escolher o **piloto** (lista completa de 2025 usando abreviações oficiais)
- Escolher a **equipe**
- Inserir parâmetros: composto (HARD/MEDIUM/SOFT), vida do pneu (voltas), velocidades setoriais
- Visualizar a **predição de tempo de volta** e delta em relação ao **tempo da pole position**

## 🚀 Como Executar

### Pré-requisitos

```bash
# Instalar dependências
pip install fastf1 streamlit pandas joblib scikit-learn feature-engine torch
```

### Treinar Modelos

```bash
# Ridge Regression (salva em models/laptime_{gp}.joblib)
python linear_regression.py

# Neural Network (salva em models/f1_laptime_model_{gp}.pt)
python neural_network.py
```

### Rodar Aplicação

```bash
streamlit run app.py
```

## 📁 Estrutura do Projeto

```
ModeloTempoVolta/
├── app.py                    # Aplicação Streamlit
├── linear_regression.py      # Treino Ridge Regression
├── neural_network.py         # Treino Neural Network (PyTorch)
├── models/                   # Modelos treinados
│   ├── laptime_{gp}.joblib   # Modelos Ridge
│   └── f1_laptime_model_{gp}.pt  # Modelos NN
├── data/raw/                 # Cache FastF1
└── README.md
```

## 📊 GPs Suportados (2025)

| GP | Tempo Pole |
|---|---|
| Silverstone | 1:24.892 |
| Monza | 1:18.792 |
| Arábia Saudita | 1:27.294 |
| Canadá | 1:10.899 |
| Mônaco | 1:09.954 |

## 🔧 Tecnologias

- **FastF1** - Coleta de dados de telemetria F1
- **scikit-learn** / **feature-engine** - Pré-processamento e Ridge Regression
- **PyTorch** - Redes neurais
- **Streamlit** - Interface web