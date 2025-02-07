import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import matplotlib.pyplot as plt
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import yfinance as yf

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# 🙌 Paramètres
LOOKBACK = 30  
BATCH_SIZE = 32
EPOCHS = 10
CRYPTOS = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD"]

# 📥 Récupération des données Yahoo Finance
def load_data():
    crypto_prices = yf.download(CRYPTOS, start='2019-03-11', end='2024-11-28', interval='1d')['Close']
    log_returns = np.log(crypto_prices / crypto_prices.shift(1)).dropna()
    return log_returns

# 📊 Préparation des données
def prepare_lstm_data(crypto_name):
    log_returns = load_data()
    df = pd.DataFrame({"realized_volatility": log_returns[crypto_name].rolling(window=30).std()}).dropna()

    train_size = int(len(df) * 0.8)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    def create_sequences(data):
        X, y = [], []
        for i in range(LOOKBACK, len(data)):
            X.append(data[i - LOOKBACK:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(scaled_data[:train_size])
    X_test, y_test = create_sequences(scaled_data[train_size - LOOKBACK:])

    return X_train, y_train, X_test, y_test, scaler, df

# 🔮 Prédictions avec remise à l'échelle
def predict_lstm(crypto_name):
    pred_file = f"results/lstm_garch_cvi_predictions_{crypto_name}.csv"
    
    # Vérifie si les prédictions existent déjà
    if os.path.exists(pred_file):
        print(f"✅ Prédictions déjà générées pour {crypto_name}, on passe.")
        return pd.read_csv(pred_file, parse_dates=["Date"], index_col="Date")

    X_train, y_train, X_test, y_test, scaler, df = prepare_lstm_data(crypto_name)
    model_path = f"models/lstm_{crypto_name}.keras"
    
    # Vérifie si le modèle existe déjà
    if not os.path.exists(model_path):
        print(f"🚀 Entraînement du modèle pour {crypto_name}...")
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        joblib.dump(scaler, f"models/scaler_lstm_{crypto_name}.pkl")
    else:
        print(f"✅ Modèle déjà entraîné pour {crypto_name}, on passe.")
        model = load_model(model_path)

    # Prédictions
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    # Remise à l'échelle des prédictions
    predictions_train = scaler.inverse_transform(predictions_train.reshape(-1, 1)).flatten()
    predictions_test = scaler.inverse_transform(predictions_test.reshape(-1, 1)).flatten()

    # Sauvegarde des prédictions
    os.makedirs("results", exist_ok=True)
    df_pred = pd.DataFrame({
        "Date": df.index[LOOKBACK:],
        "Train_Predictions": np.concatenate([predictions_train, np.full(len(df.index[LOOKBACK:]) - len(predictions_train), np.nan)]),
        "Test_Predictions": np.concatenate([np.full(len(predictions_train), np.nan), predictions_test]),
    })
    df_pred.to_csv(pred_file, index=False)
    
    return df_pred

# 📈 Génération des graphiques avec performances du modèle
def plot_lstm_results(request: Request):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    metrics_summary = []
    
    for i, crypto in enumerate(CRYPTOS):
        try:
            df_pred = predict_lstm(crypto)
            log_returns = load_data()
            realized_volatility = log_returns[crypto].rolling(window=30).std().dropna()

            # Séparation train / test
            train_size = int(len(realized_volatility) * 0.8)
            valid_train = pd.DataFrame(realized_volatility.iloc[:train_size])
            valid_test = pd.DataFrame(realized_volatility.iloc[train_size:])

            valid_train["Predictions"] = np.nan
            valid_test["Predictions"] = np.nan
            valid_train.iloc[LOOKBACK:, valid_train.columns.get_loc("Predictions")] = df_pred["Train_Predictions"].dropna().values
            valid_test.iloc[:, valid_test.columns.get_loc("Predictions")] = df_pred["Test_Predictions"].dropna().values

            # **Correction** : Alignement des données avant calcul des métriques
            aligned_realized, aligned_pred = valid_test.iloc[:, 0].align(valid_test["Predictions"], join="inner")

            # **Correction** : Calcul des métriques avec valeurs alignées
            if not aligned_realized.empty and not aligned_pred.empty:
                mse = mean_squared_error(aligned_realized, aligned_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(aligned_realized, aligned_pred)
                metrics_summary.append({"Crypto": crypto, "MSE": mse, "RMSE": rmse, "MAE": mae})

            # 📊 Tracé des graphiques
            ax = axes[i]
            ax.plot(valid_train.index, valid_train.iloc[:, 0], label='Train - Volatility', color='#ff7f0e')
            ax.plot(valid_train.index, valid_train["Predictions"], label='Train - Predictions', linestyle='--', color='#17becf')
            ax.plot(valid_test.index, valid_test.iloc[:, 0], label='Test - Volatility', color='#1f77b4')
            ax.plot(valid_test.index, valid_test["Predictions"], label='Test - Predictions', linestyle='--', color='#d62728')

            ax.set_title(f'LSTM-GARCH-CVI Prediction for {crypto}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility')
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)

        except Exception as e:
            print(f"❌ Error processing {crypto}: {e}")

    plt.tight_layout()
    plt.savefig("static/lstm_plot.png")  # 🔹 Correction du nom de l'image
    plt.close()

    # Affichage des métriques
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_html = metrics_df.to_html(classes="table table-striped", index=False)
    return templates.TemplateResponse("lstm_results.html", {"request": request, "metrics_html": metrics_html})

@router.get("/run")
def run_lstm(request: Request):
    for crypto in CRYPTOS:
        predict_lstm(crypto)  # Vérifie et génère les prédictions si besoin
    
    return plot_lstm_results(request)
