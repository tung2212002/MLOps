from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import io
import boto3
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
import json

from prediction_model.predict import generate_predictions
from server.config import TRACKING_URI, API_PREFIX, CORS_ALLOW_ORIGIN, CORS_ALLOW_CREDENTIALS, CORS_ALLOW_METHODS, CORS_ALLOW_HEADERS
from prediction_model.config import config

mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(
    title="Health Data Prediction App using FastAPI - MLOps",
    description="Predicting health and demographic statistics",
    version='1.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGIN,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

Instrumentator().instrument(app).expose(app)


class CountryData(BaseModel):
    Year: int
    Status: int = 1
    Life_expectancy: float = 0.0
    Adult_Mortality: float
    Infant_deaths: float
    Alcohol: float
    Percentage_expenditure: float
    Hepatitis_B: float
    Measles: float
    BMI: float
    Under_five_deaths: float
    Polio: float
    Total_expenditure: float
    Diphtheria: float
    HIV_AIDS: float
    GDP: float
    Population: int
    Thinness_1_19_years: float
    Thinness_5_9_years: float
    Income_composition_of_resources: float
    Schooling: float


@app.get("/")
def index():
    return {"message": "Welcome to the MLOps Health Data Prediction App."}


@app.post("/prediction_api")
def predict(country_data: CountryData):
    data = country_data.model_dump()
    prediction = generate_predictions([data])
    return {"predict": json.dumps(prediction)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
