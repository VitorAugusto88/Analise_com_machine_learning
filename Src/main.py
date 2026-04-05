from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base_Dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(Base_Dir, "modelo.pkl")
model = joblib.load(model_path)



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

# teste
@app.get("/")
def home():
    return {"message": "API funcionando"}