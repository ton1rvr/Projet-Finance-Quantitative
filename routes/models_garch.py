import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model
from fastapi import APIRouter

router = APIRouter()

@router.get("/run")


def load_log_returns(file_path="log_returns.csv"):
    """
    Charge les rendements logarithmiques des cryptos et remplace les NaN.
    """
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df.dropna()  # Suppression des lignes avec NaN

def train_garch_with_forecast(crypto_name):
    """
    Entraîne un modèle GARCH et génère des prédictions sur toute la période.
    """
    log_returns = load_log_returns()
    # Paramètres optimaux des modèles GARCH
    best_garch_params = {
    "ADA-USD": {"p": 4, "q": 3, "constant": True},
    "BNB-USD": {"p": 2, "q": 3, "constant": True},
    "BTC-USD": {"p": 4, "q": 4, "constant": True},
    "ETH-USD": {"p": 4, "q": 3, "constant": True},
}


    if crypto_name not in log_returns.columns:
        raise ValueError(f"{crypto_name} n'est pas dans les données disponibles.")
    
    if crypto_name not in best_garch_params:
        raise ValueError(f"Pas de paramètres GARCH optimaux pour {crypto_name}")

    # Sélection des paramètres optimaux
    params = best_garch_params[crypto_name]
    p, q, constant = params["p"], params["q"], params["constant"]

    # Sélection des rendements de la crypto
    crypto_returns = log_returns[crypto_name].dropna()

    # Définition et entraînement du modèle GARCH
    model = arch_model(crypto_returns, vol="Garch", p=p, q=q, mean="Constant" if constant else "Zero")
    results = model.fit(disp="off")

    # Extraction des prédictions de volatilité
    predicted_volatility = results.conditional_volatility

    # Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(results, f"models/garch_{crypto_name}.pkl")

    # Sauvegarde des prédictions de volatilité
    os.makedirs("results", exist_ok=True)
    volatility_df = pd.DataFrame({"Date": crypto_returns.index, "Volatility": predicted_volatility.values})
    volatility_df.to_csv(f"results/garch_volatility_{crypto_name}.csv", index=False)

    return predicted_volatility

def evaluate_garch_models():
    """
    Compare les prédictions GARCH aux volatilités réalisées et génère un graphique.
    """
    log_returns = load_log_returns()

    # Initialisation des subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    metrics_summary = []

    for i, crypto in enumerate(best_garch_params.keys()):
        try:
            # Charger les prédictions GARCH
            predicted_volatility = pd.read_csv(f"results/garch_volatility_{crypto}.csv")
            predicted_volatility.set_index("Date", inplace=True)
            predicted_volatility.index = pd.to_datetime(predicted_volatility.index)

            # Calcul de la volatilité réalisée (Rendements absolus sur 30 jours)
            realized_volatility = log_returns[crypto].rolling(window=30).std().dropna()

            # Vérification et remplacement des NaN par interpolation
            realized_volatility = realized_volatility.interpolate()
            predicted_volatility["Volatility"] = predicted_volatility["Volatility"].interpolate()

            # Alignement des dates
            realized_volatility, predicted_volatility_aligned = realized_volatility.align(predicted_volatility["Volatility"], join="inner")

            # Vérification finale des NaN
            if realized_volatility.isna().sum() > 0 or predicted_volatility_aligned.isna().sum() > 0:
                print(f"⚠️ Attention : NaN détectés après alignement pour {crypto} - Correction appliquée")
                realized_volatility.fillna(method='ffill', inplace=True)
                predicted_volatility_aligned.fillna(method='ffill', inplace=True)

            # Calcul des métriques
            mse = mean_squared_error(realized_volatility, predicted_volatility_aligned)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(realized_volatility, predicted_volatility_aligned)

            metrics_summary.append({
                'Crypto': crypto,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae
            })

            # Graphique comparatif
            ax = axes[i]
            ax.plot(realized_volatility.index, realized_volatility, label="Realized Volatility", color="blue", linewidth=2)
            ax.plot(predicted_volatility_aligned.index, predicted_volatility_aligned, label="Predicted Volatility (GARCH)", color="red", linewidth=2)
            ax.set_title(f"Volatility Comparison for {crypto}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Volatility")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

        except Exception as e:
            print(f"❌ Error processing {crypto}: {e}")

    # Ajuster la disposition
    plt.tight_layout()

    # Sauvegarde des performances des modèles
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv("results/garch_performance.csv", index=False)

    # Affichage des résultats
    print("\n✅ Summary of model performance:")
    print(metrics_df)

    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    for crypto in best_garch_params.keys():
        print(f"✅ Training GARCH for {crypto}...")
        train_garch_with_forecast(crypto)

    print("✅ Evaluating GARCH models...")
    evaluate_garch_models()
