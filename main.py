from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from routes import data_processing, models_garch, models_lstm, sharpe, sharpe_cvar

app = FastAPI()

# Définition des templates et fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Page d'accueil avec les 5 boutons cliquables
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Inclusion des routes
app.include_router(data_processing.router, prefix="/fetch-data")
app.include_router(models_garch.router, prefix="/garch")
app.include_router(models_lstm.router, prefix="/lstm-garch-cvi")
app.include_router(sharpe.router, prefix="/sharpe")
app.include_router(sharpe_cvar.router, prefix="/sharpe-cvar")

