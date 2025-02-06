import pandas as pd
import yfinance as yf
import requests
import cloudscraper
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
import matplotlib
matplotlib.use('Agg')  # Désactive Tkinter pour éviter l'erreur
import matplotlib.pyplot as plt

import os

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def get_data_yfinance(cryptos, start_date, end_date):
    """
    Récupère les données historiques des cryptos via yfinance.
    """
    data = yf.download(cryptos, start=start_date, end=end_date, interval='1d')
    return data['Close'].dropna()

def get_data_investing(start_date, end_date):
    """
    Récupère les données du Crypto Volatility Index (CVI) via Investing.com API.
    """
    url = f"https://api.investing.com/api/financialdata/historical/1178491?start-date={start_date}&end-date={end_date}&time-frame=Daily&add-missing-rows=false"
    headers = {"domain-id": "www"}
    session = cloudscraper.create_scraper()
    
    response = session.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json().get("data", [])
        df = pd.DataFrame([{ 
            "Date": row["rowDate"], 
            "Price": row.get("last_close") 
        } for row in data])
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")
    else:
        print(f"API Error: HTTP {response.status_code}")
        return None

def save_data(data, filename):
    """
    Sauvegarde les données au format CSV.
    """
    data.to_csv(filename, index=True)
    print(f"Data saved in {filename}")

@router.get("/run")
def fetch_data(request: Request):
    """
    Exécute la récupération des données et affiche le head des fichiers.
    """
    cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD']
    start_date = '2019-03-11'
    end_date = '2024-11-28'

    # Récupérer et sauvegarder les données
    crypto_data = get_data_yfinance(cryptos, start_date, end_date)
    save_data(crypto_data, "crypto_prices.csv")

    cvi_data = get_data_investing(start_date, end_date)
    if cvi_data is not None:
        save_data(cvi_data, "cvi_data.csv")

    # Lire les 5 premières lignes pour affichage
    crypto_head = pd.read_csv("crypto_prices.csv").head().to_html()
    cvi_head = pd.read_csv("cvi_data.csv").head().to_html() if cvi_data is not None else "<p>Pas de données CVI disponibles.</p>"

    return templates.TemplateResponse("fetch_data.html", {
        "request": request,
        "crypto_head": crypto_head,
        "cvi_head": cvi_head
    })

@router.get("/plot")
def plot_crypto_prices():
    """
    Génère un graphique des prix des cryptos.
    """
    df = pd.read_csv("crypto_prices.csv", index_col=0, parse_dates=True)
    
    plt.figure(figsize=(10, 5))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.title("Évolution des Prix des Cryptos")
    plt.grid(True)
    
    # Sauvegarde et affichage du graphique
    plot_path = "static/crypto_prices_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return FileResponse(plot_path)
