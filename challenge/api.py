import fastapi
import pandas as pd
import pickle
from pydantic import BaseModel
from typing import List
from challenge.model import DelayModel
from fastapi import HTTPException

app = fastapi.FastAPI()

VALID_OPERAS = {'American Airlines', 'Air Canada', 'Air France', 'Aeromexico',
       'Aerolineas Argentinas', 'Austral', 'Avianca', 'Alitalia',
       'British Airways', 'Copa Air', 'Delta Air', 'Gol Trans', 'Iberia',
       'K.L.M.', 'Qantas Airways', 'United Airlines', 'Grupo LATAM',
       'Sky Airline', 'Latin American Wings', 'Plus Ultra Lineas Aereas',
       'JetSmart SPA', 'Oceanair Linhas Aereas', 'Lacsa'}
VALID_TIPOS = {"I", "N"}
VALID_MONTHS = set(range(1, 13))

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class PredictRequest(BaseModel):
    flights: List[Flight]

# Load trained model
with open("challenge/model.pkl", "rb") as file:
    model = pickle.load(file)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(
    request: PredictRequest
) -> dict:
    data = pd.DataFrame(
        [flight.model_dump() for flight in request.flights])
    
    for _,flight in data.iterrows():
        if flight.OPERA not in VALID_OPERAS:
            raise HTTPException(status_code=400, detail="Invalid OPERADOR")

        if flight.TIPOVUELO not in VALID_TIPOS:
            raise HTTPException(status_code=400, detail="Invalid TIPOVUELO")

        if flight.MES not in VALID_MONTHS:
            raise HTTPException(status_code=400, detail="Invalid MES")

    features = model.preprocess(data)

    predictions = model.predict(features)

    return {
        "predict": predictions
    }