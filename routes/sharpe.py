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
    var_alpha = 0.05  # Niveau de confiance pour la CVaR
    log_returns, vol_predictions = load_data()

    predictions_lstm = pd.DataFrame(vol_predictions).iloc[-lookback_vol:].copy()
    print("🔎 Colonnes disponibles :", predictions_lstm.columns.tolist())

    # Moyenne des volatilités prédites sur la période de lookback
    volatility_predicted_ma = predictions_lstm.mean()

    return log_returns, volatility_predicted_ma

# 🔹 Optimisation du portefeuille avec maximisation du ratio de Sharpe basé sur la CVaR
def optimize_portfolio_cvar():
    log_returns, volatility_predicted_ma = calculate_predicted_volatility()

    # Extraction des volatilités moyennes prédites par crypto
    vol_dict = {}
    for col in log_returns.columns:
        matching_col = [c for c in volatility_predicted_ma.index if col in c]
        if matching_col:
            vol_dict[f"{col}_vol"] = volatility_predicted_ma[matching_col[0]]
        else:
            print(f"⚠️ Attention : Aucune volatilité prédite trouvée pour {col}")

    # Moyenne des rendements des 30 derniers jours
    last_30_returns = log_returns.iloc[-30:]
    mu_cvar = last_30_returns.mean().values  

    # Construction de la matrice de covariance
    volatilities_cvar = np.array(list(vol_dict.values()))
    corr_matrix_cvar = last_30_returns.corr().values
    cov_matrix_cvar = np.outer(volatilities_cvar, volatilities_cvar) * corr_matrix_cvar  

    # 🔹 Fonction de calcul de la CVaR
    def portfolio_cvar(weights_cvar, portfolio_returns_cvar, alpha=var_alpha):
        portf_returns_cvar = np.dot(portfolio_returns_cvar, weights_cvar)
        var = np.percentile(portf_returns_cvar, alpha * 100)
        cvar = np.mean(portf_returns_cvar[portf_returns_cvar <= var])  
        return abs(cvar)  

    # 🔹 Fonction de maximisation du ratio de Sharpe basé sur la CVaR
    def sharpe_ratio_cvar(weights_cvar, mu, portfolio_returns_cvar, risk_free_rate):
        portf_return_cvar = np.dot(weights_cvar, mu) 
        portf_cvar = portfolio_cvar(weights_cvar, portfolio_returns_cvar) 
        return -(portf_return_cvar - risk_free_rate) / portf_cvar  

    # Contraintes et optimisation
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(-1,1) for _ in range(len(mu_cvar))] 

    initial_weights = np.random.uniform(-1, 1, size=len(mu_cvar))
    initial_weights /= np.sum(np.abs(initial_weights))  

    opt_result_cvar = minimize(
        sharpe_ratio_cvar,
        initial_weights,
        args=(mu_cvar, last_30_returns, rf),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights_cvar = opt_result_cvar.x
    max_sharpe_ratio_cvar = -opt_result_cvar.fun

    return optimal_weights_cvar, max_sharpe_ratio_cvar, vol_dict, mu_cvar, last_30_returns

# 🔹 Graphique des Rendements Cumulés avec optimisation CVaR
def plot_cumulative_returns_cvar(optimal_weights_cvar, last_30_returns):
    """
    Affiche le rendement cumulé du portefeuille optimisé avec la CVaR.
    """
    portfolio_cumulative_returns_cvar = (1 + np.dot(last_30_returns, optimal_weights_cvar)).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(last_30_returns.index, portfolio_cumulative_returns_cvar, label="Optimized Portfolio (CVaR)", color='red')
    plt.xlabel("Date")
    plt.ylabel("Cumulative performance")
    plt.title("Sharpe-based optimized portfolio performance based on CVaR")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# 🔹 Affichage des résultats
def display_results_cvar():
    optimal_weights_cvar, max_sharpe_ratio_cvar, vol_dict, mu_cvar, last_30_returns = optimize_portfolio_cvar()

    print("\n✅ **Résultats de l'Optimisation du Portefeuille (CVaR)**")
    print(f"🔹 **Ratio de Sharpe Maximisé (CVaR)** : {max_sharpe_ratio_cvar:.4f}")
    print("📊 **Poids optimaux des cryptos** :")
    for i, col in enumerate(vol_dict.keys()):
        print(f"   - {col}: {optimal_weights_cvar[i]:.4f}")

    print("\n📉 **Volatilités Prédites Moyennes** :")
    for key, value in vol_dict.items():
        print(f"   - {key}: {value:.4f}")

    # 🔹 Graphique : Répartition des poids (CVaR)
    plt.figure(figsize=(10, 5))
    plt.bar(list(vol_dict.keys()), optimal_weights_cvar, color='lightcoral')
    plt.xlabel("Cryptos")
    plt.ylabel("Poids optimaux")
    plt.title("Répartition Optimale du Portefeuille (Max Sharpe basé sur la CVaR)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # 🔹 Affichage des rendements cumulés (CVaR)
    plot_cumulative_returns_cvar(optimal_weights_cvar, last_30_returns)

# 🔹 Exécution principale
if __name__ == "__main__":
    display_results_cvar()
