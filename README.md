# Projet-Finance-Quantitative

# 📌 **Documentation de l'API de Prédiction et d'Optimisation des Cryptos**

## 🚀 **Introduction**
Cette API fournit des fonctionnalités avancées de prédiction et d’optimisation des cryptos en utilisant un modèle LSTM combiné avec GARCH et CVaR. L'API permet :  
✅ La récupération des données financières depuis Yahoo Finance  
✅ La prédiction de la volatilité réalisée des cryptos via un modèle LSTM  
✅ L’optimisation d’un portefeuille de cryptos avec le ratio de Sharpe et la CVaR  

---

## 📍 **Endpoints de l'API**

### 📈 **Prédiction avec LSTM**
#### **1️⃣ Lancer l'entraînement et la prédiction**
**URL:** `/run`  
**Méthode:** `GET`  
**Description:** Ce endpoint entraîne le modèle LSTM si nécessaire et génère des prédictions de volatilité pour les cryptos sélectionnées.  

**Réponse:**  
📌 Retourne une page HTML affichant les résultats des prédictions sous forme de graphiques et de métriques de performance (MSE, RMSE, MAE).  

---

### 📊 **Optimisation du portefeuille (Sharpe et CVaR)**
#### **2️⃣ Lancer l'optimisation du portefeuille avec Sharpe**
**URL:** `/run`  
**Méthode:** `GET`  
**Description:** Ce endpoint charge les rendements log, récupère les prédictions de volatilité et optimise un portefeuille de cryptos en maximisant le ratio de Sharpe.  

**Réponse:**  
📌 Retourne les poids optimaux du portefeuille, la volatilité moyenne prédite, ainsi qu’un graphique des performances cumulées du portefeuille.  

---

#### **3️⃣ Lancer l'optimisation du portefeuille avec Sharpe basé sur la CVaR**
**URL:** `/run`  
**Méthode:** `GET`  
**Description:** Ce endpoint effectue une optimisation du portefeuille en maximisant le ratio de Sharpe, tout en utilisant la Conditional Value at Risk (CVaR) comme indicateur de risque.  

**Réponse:**  
📌 Retourne les poids optimaux, la volatilité moyenne prédite, et un graphique des rendements cumulés du portefeuille optimisé selon la CVaR.  

---

## ⚙️ **Détails Techniques**
### 🔹 **Modèle LSTM**
- Entrée : Log-returns des cryptos, fenêtre de 30 jours  
- Architecture :  
  - 2 couches LSTM  
  - Dropout régulier  
  - Dense en sortie  
- Optimiseur : Adam  
- Fonction de perte : Mean Squared Error  

### 🔹 **Optimisation du portefeuille**
- **Ratio de Sharpe :** Maximisation du rendement attendu divisé par la volatilité  
- **CVaR :** Évaluation du risque de pertes extrêmes (5% des pires cas)  
- **Contraintes :**  
  - Somme des poids = 1  
  - Autorisation des positions longues et courtes (-1 à 1)  

---

## 📌 **Exemple d’Utilisation**
1️⃣ Démarrer un serveur FastAPI :  
```bash
uvicorn main:app --reload
```
2️⃣ Accéder à la documentation interactive :  
📌 **Swagger UI** : `http://127.0.0.1:8000/docs`  
📌 **ReDoc** : `http://127.0.0.1:8000/redoc`  
