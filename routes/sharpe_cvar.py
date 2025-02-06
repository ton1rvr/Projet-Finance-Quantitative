import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from fastapi import APIRouter

router = APIRouter()

@router.get("/run")


# 🔹 Charger les données de volatilité prédite et de rendements
def load_data():
    
    log_returns = pd.read_csv("log_returns.csv", index_col=0, parse_dates=True)

    volatility_predictions = {}
    for crypto in log_returns.columns:
        vol_path = f"results/lstm_garch_cvi_volatility_{crypto}.csv"
        if os.path.exists(vol_path):
            vol_pred = pd.read_csv(vol_path, index_col=0, parse_dates=True)
            volatility_predictions[crypto] = vol_pred["LSTM_GARCH_CVI_Test"]

    return log_returns, volatility_predictions

# 🔹 Calcul des volatilités moyennes prédictives
def calculate_predicted_volatility():
    # 🔹 Paramètres globaux
    rf = 0.03 / 365  # Taux sans risque journalier
    lookback_vol = 60  # Fenêtre pour la moyenne de volatilité prédite

    log_returns, vol_predictions = load_data()

    # Vérifier les colonnes disponibles pour éviter KeyError
    predictions_lstm = pd.DataFrame(vol_predictions).iloc[-lookback_vol:].copy()
    print("🔎 Colonnes disponibles :", predictions_lstm.columns.tolist())

    # Moyenne des volatilités prédites sur la période de lookback
    volatility_predicted_ma = predictions_lstm.mean()

    return log_returns, volatility_predicted_ma

# 🔹 Optimisation du ratio de Sharpe
def optimize_sharpe_ratio():
    log_returns, volatility_predicted_ma = calculate_predicted_volatility()

    # Extraction des volatilités moyennes prédites par crypto avec vérification des noms
    vol_dict = {}
    for col in log_returns.columns:
        matching_col = [c for c in volatility_predicted_ma.index if col in c]
        if matching_col:
            vol_dict[f"{col}_vol"] = volatility_predicted_ma[matching_col[0]]
        else:
            print(f"⚠️ Attention : Aucune volatilité prédite trouvée pour {col}")

    # Moyenne des rendements des 30 derniers jours
    last_30_returns = log_returns.iloc[-30:]
    mu = last_30_returns.mean().values  # Rendements attendus

    # Construction de la matrice de covariance basée sur la volatilité prédite
    volatilities = np.array(list(vol_dict.values()))
    corr_matrix = last_30_returns.corr().values
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

    # Fonction objectif : Maximisation du ratio de Sharpe
    def sharpe_ratio(weights, mu, cov_matrix, risk_free_rate=0):
        portfolio_return = np.dot(weights, mu)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility  # On minimise

    # Contraintes : somme des poids = 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(-1, 1) for _ in range(len(mu))]  # Autoriser les positions longues et courtes

    # Initialisation aléatoire des poids
    initial_weights = np.random.uniform(-1, 1, size=len(mu))
    initial_weights /= np.sum(np.abs(initial_weights))

    # Optimisation avec SciPy
    result = minimize(sharpe_ratio, initial_weights, args=(mu, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x
    max_sharpe_ratio = -result.fun

    return optimal_weights, max_sharpe_ratio, vol_dict, mu

# 🔹 Graphique des Rendements Cumulés
def plot_cumulative_returns(optimal_weights):
    """
    Affiche le rendement cumulé du portefeuille sur les dates prédites.
    """
    log_returns, _ = load_data()

    # Utiliser uniquement les dernières valeurs correspondant aux prédictions
    last_30_returns = log_returns.iloc[-30:]

    # Calcul du rendement du portefeuille avec les poids optimisés
    portfolio_returns = last_30_returns.dot(optimal_weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # 🔹 Graphique des rendements cumulés
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_returns, color='green', linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Rendement Cumulé")
    plt.title("Performance Cumulée du Portefeuille (Sur Période Prédite)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# 🔹 Affichage des résultats
def display_results():
    optimal_weights, max_sharpe_ratio, vol_dict, mu = optimize_sharpe_ratio()

    print("\n✅ **Résultats de l'Optimisation du Portefeuille**")
    print(f"🔹 **Ratio de Sharpe Maximisé** : {max_sharpe_ratio:.4f}")
    print("📊 **Poids optimaux des cryptos** :")
    for i, col in enumerate(vol_dict.keys()):
        print(f"   - {col}: {optimal_weights[i]:.4f}")

    print("\n📉 **Volatilités Prédites Moyennes** :")
    for key, value in vol_dict.items():
        print(f"   - {key}: {value:.4f}")

    # 🔹 Graphique : Répartition des poids
    plt.figure(figsize=(10, 5))
    plt.bar(list(vol_dict.keys()), optimal_weights, color='skyblue')
    plt.xlabel("Cryptos")
    plt.ylabel("Poids optimaux")
    plt.title("Répartition Optimale du Portefeuille (Max Sharpe)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # 🔹 Affichage des rendements cumulés
    plot_cumulative_returns(optimal_weights)

# 🔹 Exécution principale
if __name__ == "__main__":
    display_results()
