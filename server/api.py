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

from prediction_model.predict import generate_predictions, generate_predictions_batch
from config import TRACKING_URI, API_PREFIX, ENABLE_OPENAPI, CORS_ALLOW_ORIGIN, CORS_ALLOW_CREDENTIALS, CORS_ALLOW_METHODS, CORS_ALLOW_HEADERS
from prediction_model.config import config


def upload_to_s3(file_content, filename):
    s3 = boto3.client('s3')
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    if filename.endswith('.csv'):
        filename = filename[:-4]
        
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    folder_path = f"{config.FOLDER}/{current_date}"
 
    filename_with_datetime = f"{filename}_{current_datetime}.csv"
    
    s3_key = f"{folder_path}/{filename_with_datetime}"

    response = s3.put_object(Bucket=config.S3_BUCKET, Key=s3_key, Body=file_content)
  
    return s3_key 


mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(
    title="Health Data Prediction App using FastAPI - MLOps",
    description="Predicting health and demographic statistics",
    version='1.0',
    openapi_url=(
        f"{API_PREFIX}/openapi.json" if ENABLE_OPENAPI else None
    ),
    redoc_url=None,
    docs_url="/docs",
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
    Status: str
    Life_expectancy: float
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
    return {"message": "Welcome to the MLOps Health Data Prediction App"}


@app.post("/prediction_api")
def predict(country_data: CountryData):
    data = country_data.model_dump()
    prediction = generate_predictions([data])["prediction"][0]
    return {"prediction": prediction}


@app.post("/prediction_ui")
def predict_gui(Year: int, Status: str, Life_expectancy: float, Adult_Mortality: float, Infant_deaths: float,
                Alcohol: float, Percentage_expenditure: float, Hepatitis_B: float, Measles: float, BMI: float,
                Under_five_deaths: float, Polio: float, Total_expenditure: float, Diphtheria: float, HIV_AIDS: float,
                GDP: float, Population: int, Thinness_1_19_years: float, Thinness_5_9_years: float,
                Income_composition_of_resources: float, Schooling: float):

    input_data = [Year, Status, Life_expectancy, Adult_Mortality, Infant_deaths, Alcohol, Percentage_expenditure,
                  Hepatitis_B, Measles, BMI, Under_five_deaths, Polio, Total_expenditure, Diphtheria, HIV_AIDS, GDP,
                  Population, Thinness_1_19_years, Thinness_5_9_years, Income_composition_of_resources, Schooling]

    cols = ['Year', 'Status', 'Life_expectancy', 'Adult_Mortality', 'Infant_deaths', 'Alcohol', 
            'Percentage_expenditure', 'Hepatitis_B', 'Measles', 'BMI', 'Under_five_deaths', 'Polio', 
            'Total_expenditure', 'Diphtheria', 'HIV_AIDS', 'GDP', 'Population', 'Thinness_1_19_years', 
            'Thinness_5_9_years', 'Income_composition_of_resources', 'Schooling']
    
    data_dict = dict(zip(cols, input_data))
    prediction = generate_predictions([data_dict])["prediction"][0]
    return {"prediction": prediction}


@app.post("/batch_prediction")
async def batch_predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), index_col=False)
    
    # Ensure the CSV file contains the required features
    required_columns = ['Year', 'Status', 'Life_expectancy', 'Adult_Mortality', 'Infant_deaths', 'Alcohol', 
                        'Percentage_expenditure', 'Hepatitis_B', 'Measles', 'BMI', 'Under_five_deaths', 'Polio', 
                        'Total_expenditure', 'Diphtheria', 'HIV_AIDS', 'GDP', 'Population', 'Thinness_1_19_years', 
                        'Thinness_5_9_years', 'Income_composition_of_resources', 'Schooling']
    if not all(column in df.columns for column in required_columns):
        return {"error": "CSV file does not contain the required columns."}

    predictions = generate_predictions_batch(df)["prediction"]
    df['Prediction'] = predictions
    result = df.to_csv(index=False)
    
    s3_key = upload_to_s3(result.encode('utf-8'), file.filename)

    return StreamingResponse(io.BytesIO(result.encode('utf-8')), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
