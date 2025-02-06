import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import matplotlib.pyplot as plt

from fastapi import APIRouter

router = APIRouter()

@router.get("/run")



def load_data():
    """
    Charge les rendements logarithmiques, la volatilité GARCH et le CVI.
    """
    log_returns = pd.read_csv("log_returns.csv", index_col=0, parse_dates=True)
    cvi_data = pd.read_csv("cvi.csv", index_col=0, parse_dates=True)

    garch_volatilities = {}
    for crypto in log_returns.columns:
        vol_path = f"results/garch_volatility_{crypto}.csv"
        if os.path.exists(vol_path):
            garch_vol = pd.read_csv(vol_path, index_col=0, parse_dates=True)
            garch_volatilities[crypto] = garch_vol["Volatility"]
    
    return log_returns, garch_volatilities, cvi_data

def prepare_lstm_data(crypto_name):
    # Paramètres
    LOOKBACK = 30  
    BATCH_SIZE = 32
    EPOCHS = 10
    CRYPTOS = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD"]
    """
    Prépare les données pour LSTM en suivant la structure du notebook.
    """
    log_returns, garch_volatilities, cvi_data = load_data()
    
    if crypto_name not in log_returns.columns:
        raise ValueError(f"{crypto_name} n'est pas dans les données disponibles.")
    
    if crypto_name not in garch_volatilities:
        raise ValueError(f"Pas de volatilité GARCH disponible pour {crypto_name}")

    # Fusion des données
    df = pd.DataFrame({
        "realized_volatility": log_returns[crypto_name].rolling(window=30).std(),
        "garch_volatility": garch_volatilities[crypto_name],
        "CVI": cvi_data["Price"]
    }).dropna()

    # Séparation train/test avec 80% / 20% tout en conservant la chronologie
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    # Normalisation
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Création des séquences LSTM
    def create_sequences(data, start_idx):
        X, y = [], []
        for i in range(LOOKBACK, len(data)):
            X.append(data[i - LOOKBACK:i])
            y.append(data[i, 0])  # On prédit la volatilité réalisée
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(scaled_data[:train_size], start_idx=0)
    X_test, y_test = create_sequences(scaled_data[train_size - LOOKBACK:], start_idx=train_size)

    return X_train, y_train, X_test, y_test, scaler, df

def build_lstm_model():
    """
    Crée le modèle LSTM.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 3)),  
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_lstm(crypto_name):
    """
    Entraîne un modèle LSTM et sauvegarde les résultats.
    """
    X_train, y_train, _, _, scaler, df = prepare_lstm_data(crypto_name)

    model = build_lstm_model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1, callbacks=[callback])

    os.makedirs("models", exist_ok=True)
    model.save(f"models/lstm_garch_cvi_{crypto_name}.keras")  
    joblib.dump(scaler, f"models/scaler_lstm_{crypto_name}.pkl")

    print(f"✅ Modèle LSTM-GARCH-CVI entraîné pour {crypto_name}")

def predict_lstm(crypto_name):
    """
    Prédit la volatilité future avec LSTM et stocke les prédictions train/test.
    """
    X_train, y_train, X_test, y_test, scaler, df = prepare_lstm_data(crypto_name)

    model = tf.keras.models.load_model(f"models/lstm_garch_cvi_{crypto_name}.keras")  

    # Prédictions sur train
    predictions_train = model.predict(X_train)
    predictions_rescaled_train = scaler.inverse_transform(
        np.hstack((predictions_train, np.zeros((len(predictions_train), 2))))
    )[:, 0]  

    # Prédictions sur test
    predictions_test = model.predict(X_test)
    predictions_rescaled_test = scaler.inverse_transform(
        np.hstack((predictions_test, np.zeros((len(predictions_test), 2))))
    )[:, 0]  

    os.makedirs("results", exist_ok=True)

    # Construction du DataFrame de prédictions
    train_dates = df.index[:len(predictions_train)]
    test_dates = df.index[len(predictions_train):]

    pred_df = pd.DataFrame({
        "Date": df.index[LOOKBACK:],
        "LSTM_GARCH_CVI_Train": np.concatenate([predictions_rescaled_train, np.full(len(predictions_test), np.nan)]),
        "LSTM_GARCH_CVI_Test": np.concatenate([np.full(len(predictions_train), np.nan), predictions_rescaled_test]),
    })

    pred_df.to_csv(f"results/lstm_garch_cvi_volatility_{crypto_name}.csv", index=False)

    print(f"📈 Prédictions LSTM-GARCH-CVI enregistrées pour {crypto_name}")

    return pred_df

def plot_results():
    """
    Compare la volatilité réelle et prédite avec séparation train/test, en respectant les couleurs et subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, crypto in enumerate(CRYPTOS):
        try:
            lstm_pred = pd.read_csv(f"results/lstm_garch_cvi_volatility_{crypto}.csv", index_col=0, parse_dates=True)

            log_returns, _, _ = load_data()
            realized_volatility = log_returns[crypto].rolling(window=30).std().dropna()

            # Alignement des dates pour éviter les décalages
            realized_volatility, lstm_pred_train = realized_volatility.align(lstm_pred["LSTM_GARCH_CVI_Train"], join="inner")
            realized_volatility, lstm_pred_test = realized_volatility.align(lstm_pred["LSTM_GARCH_CVI_Test"], join="inner")

            # Séparation des données Train et Test
            valid_train = pd.DataFrame(realized_volatility[:len(lstm_pred_train)])
            valid_test = pd.DataFrame(realized_volatility[len(lstm_pred_train):])

            valid_train['Predictions'] = lstm_pred_train
            valid_test['Predictions'] = lstm_pred_test

            ax = axes[i]
            ax.plot(valid_train.index, valid_train.iloc[:, 0], label='Train - Volatility', color='#ff7f0e')
            ax.plot(valid_train.index, valid_train['Predictions'], label='Train - Predictions', linestyle='--', color='#17becf')
            ax.plot(valid_test.index, valid_test.iloc[:, 0], label='Test - Volatility', color='#1f77b4')
            ax.plot(valid_test.index, valid_test['Predictions'], label='Test - Predictions', linestyle='--', color='#d62728')

            ax.set_title(f'LSTM-GARCH-CVI Prediction for {crypto}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility')
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)

        except Exception as e:
            print(f"❌ Error processing {crypto}: {e}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for crypto in CRYPTOS:
        train_lstm(crypto)
        predict_lstm(crypto)
    
    plot_results()
