import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# 📌 Paramètres optimaux des modèles GARCH
best_garch_params = {
    "ADA-USD": {"p": 4, "q": 3, "constant": True},
    "BNB-USD": {"p": 2, "q": 3, "constant": True},
    "BTC-USD": {"p": 4, "q": 4, "constant": True},
    "ETH-USD": {"p": 4, "q": 3, "constant": True},
}

def load_log_returns(file_path="log_returns.csv"):
    """ Charge les rendements logarithmiques des cryptos. """
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df.dropna()

def train_garch_with_forecast(crypto_name):
    """ Entraîne un modèle GARCH et génère des prédictions. """
    log_returns = load_log_returns()

    if crypto_name not in log_returns.columns:
        return {"error": f"{crypto_name} n'est pas disponible."}
    
    if crypto_name not in best_garch_params:
        return {"error": f"Pas de paramètres GARCH pour {crypto_name}"}

    # Sélection des paramètres optimaux
    params = best_garch_params[crypto_name]
    p, q, constant = params["p"], params["q"], params["constant"]
    crypto_returns = log_returns[crypto_name].dropna()

    # Entraînement GARCH
    model = arch_model(crypto_returns, vol="Garch", p=p, q=q, mean="Constant" if constant else "Zero")
    results = model.fit(disp="off")

    # Sauvegarde des prédictions de volatilité
    os.makedirs("results", exist_ok=True)
    volatility_df = pd.DataFrame({"Date": crypto_returns.index, "Volatility": results.conditional_volatility.values})
    volatility_df.to_csv(f"results/garch_volatility_{crypto_name}.csv", index=False)

    return {"message": f"Modèle GARCH entraîné pour {crypto_name} et prédictions sauvegardées."}

@router.get("/train-garch")
def train_all_garch():
    """ Entraîne GARCH sur toutes les cryptos et sauvegarde les prédictions. """
    for crypto in best_garch_params.keys():
        train_garch_with_forecast(crypto)
    
    return {"message": "Modèles GARCH entraînés et sauvegardés."}

@router.get("/plot-garch", response_class=HTMLResponse)
def plot_garch_volatility(request: Request):
    """ Génère un graphique comparant la volatilité réelle et prédite. """
    log_returns = load_log_returns()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    metrics_summary = []

    for i, crypto in enumerate(best_garch_params.keys()):
        try:
            print(f"📌 Traitement de {crypto}...")

            # Charger prédictions et aligner les dates
            predicted_volatility = pd.read_csv(f"results/garch_volatility_{crypto}.csv", parse_dates=["Date"])
            realized_volatility = log_returns[crypto].rolling(window=30).std().dropna()

            # Vérification initiale
            print(f"✅ {crypto} - Avant alignement : Réalisée {realized_volatility.shape}, Prédite {predicted_volatility.shape}")

            # Conversion des dates en index datetime
            predicted_volatility.set_index("Date", inplace=True)

            # Appliquer ffill() pour éviter les NaN
            realized_volatility.fillna(method="ffill", inplace=True)
            predicted_volatility.fillna(method="ffill", inplace=True)

            # Alignement des dates
            realized_volatility, predicted_volatility_aligned = realized_volatility.align(predicted_volatility["Volatility"], join="inner")

            # Vérification après alignement
            print(f"✅ {crypto} - Après alignement : Réalisée {realized_volatility.shape}, Prédite {predicted_volatility_aligned.shape}")

            if realized_volatility.empty or predicted_volatility_aligned.empty:
                print(f"⚠️ {crypto} - Problème : séries vides après alignement.")
                continue

            # Calcul des métriques
            mse = mean_squared_error(realized_volatility, predicted_volatility_aligned)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(realized_volatility, predicted_volatility_aligned)
            metrics_summary.append({"Crypto": crypto, "MSE": mse, "RMSE": rmse, "MAE": mae})

            # Tracé des graphes
            ax = axes[i]
            ax.plot(realized_volatility.index, realized_volatility, label="Realized Volatility", color="blue")
            ax.plot(predicted_volatility_aligned.index, predicted_volatility_aligned, label="Predicted Volatility", color="red")
            ax.set_title(f"Volatility Comparison for {crypto}")
            ax.legend()
            ax.grid(True)

        except Exception as e:
            print(f"❌ Erreur lors du traitement de {crypto}: {e}")

    plt.tight_layout()
    plt.savefig("static/garch_plot.png")
    plt.close()

    # Création du tableau HTML pour afficher les métriques
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_html = metrics_df.to_html(classes="table table-striped", index=False)

    return templates.TemplateResponse("garch_results.html", {"request": request, "metrics_html": metrics_html})

@router.get("/run")
def run_garch(request: Request):
    """ Redirige vers la visualisation des prédictions GARCH """
    return plot_garch_volatility(request)
