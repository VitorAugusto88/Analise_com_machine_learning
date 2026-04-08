from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminhos absolutos baseados na localização do main.py (dentro de Src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "Frontend")
MODEL_PATH = os.path.join(BASE_DIR, "modelo.pkl")

# Servir arquivos estáticos do Frontend
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Carregar modelo
model = joblib.load(MODEL_PATH)


class CasaSimples(BaseModel):
    rm: float


medias = {
    "crim": 3.61,
    "zn": 11.36,
    "indus": 11.13,
    "chas": 0,
    "nox": 0.55,
    "rm": 6.28,
    "age": 68.5,
    "dis": 3.79,
    "rad": 9.5,
    "tax": 408,
    "ptratio": 18.45,
    "b": 356,
    "lstat": 12.65
}


@app.post("/prever")
def prever(dados: CasaSimples):
    entrada = medias.copy()
    entrada["rm"] = dados.rm
    df = pd.DataFrame([entrada])
    previsao = model.predict(df)
    return {"preco": float(previsao[0])}


@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))