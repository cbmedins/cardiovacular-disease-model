# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 20:15:35 2023

@author: cris_
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",  # Agrega tu origen local si es necesario
    "https://prototipo-web-titulacion.vercel.app",
    "https://cardiovascular-disease-app.vercel.app", /# Agrega la URL de tu aplicación en Vercel
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Cargar el modelo entrenado y el scaler
model = load_model("modeloCnn.h5")
scaler = joblib.load("standardScalerCnn.pkl")


# Definir la estructura de entrada para las solicitudes
class HeartData(BaseModel):
    Age: float
    Sex: float
    ChestPainType: float
    RestingBP: float
    Cholesterol: float
    FastingBS: float
    RestingECG: float
    MaxHR: float
    ExerciseAngina: float
    Oldpeak: float
    ST_Slope: float


@app.post("/predict/")
def predict(data: HeartData):
    # Convertir los datos de entrada en un DataFrame
    data_dict = data.dict()
    input_data = pd.DataFrame([data_dict])

    # Escalar los datos de entrada
    input_data_scaled = scaler.transform(input_data)

    # Realizar la predicción con el modelo
    prediction = model.predict(input_data_scaled).round(0)
    #risk_percentage = prediction.item()


    # Convertir el resultado de la predicción en un valor entero (0 o 1)
    result = int(prediction[0, 0])

    return {"prediction": result}

if __name__ == '__main__':
        import uvicorn
        uvicorn.run(app, port=8000)
