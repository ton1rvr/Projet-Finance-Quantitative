# 📈 Crypto Volatility Forecasting & Portfolio Optimization

**Université Paris 1 – Master MOSEF**

### 🎯 Objective

Forecast the volatility of BTC, ETH, BNB, and ADA using GARCH, LSTM, and hybrid LSTM-GARCH models, then build optimized portfolios based on predicted risk.

### ⚙️ Methods

* **Models**: GARCH, LSTM, LSTM-GARCH (+ CVI input)
* **Metrics**: RMSE, MAPE
* **Portfolio optimization**:

  * Maximize Sharpe Ratio
  * Control tail risk with CVaR
  * Optimized via SLSQP

### 📊 Key Results

* **LSTM-GARCH + CVI** gives the best volatility forecasts
* Two portfolio strategies:

  * Sharpe-based (SR ≈ 1.07)
  * CVaR-adjusted (SR ≈ 0.80, lower risk)

### 🛠 Stack

Python, TensorFlow, arch, pandas, Yahoo Finance API, CVI data

### 👥 Authors

| Nom              | GitHub / Contact                                  | 
|------------------|---------------------------------------------------|
| Gaétan Dumas     | [@gaetan250](https://github.com/gaetan250)        |
| Pierre Liberge   | [@pierreliberge](https://github.com/pierreliberge)|
| Tonin Rivory     | [@ton1rvr](https://github.com/ton1rvr)            | 


### 📄 See full paper : [Working_Paper.pdf](https://github.com/ton1rvr/Projet-Finance-Quantitative/blob/main/Working_Paper.pdf)

