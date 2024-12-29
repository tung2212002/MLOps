import pytest
import mlflow
import pandas as pd
from prediction_model.config import config
from prediction_model.predict import generate_predictions
from prediction_model.processing.data_handling import load_dataset
from pydantic import BaseModel
import random

# Set the MLflow tracking URI
mlflow.set_tracking_uri(config.TRACKING_URI)

@pytest.fixture
def random_data_point():
    random_country = generate_random_country_data()
    return random_country.model_dump()

@pytest.fixture
def single_prediction(random_data_point):
    random_data_point = [random_data_point]
    result = generate_predictions(random_data_point)
    return result

def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None

def test_single_pred_numeric_type(single_prediction):
    prediction = single_prediction
    assert isinstance(prediction, (float, int))

def test_single_pred_value_range(single_prediction):
    prediction = single_prediction
    assert 0 <= prediction <= 150

def test_single_pred_valid_result(single_prediction):
    prediction = single_prediction
    assert prediction is not None
    assert not pd.isna(prediction)
    
    
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
    
def generate_random_country_data():
    return CountryData(
        Year=random.randint(2000, 2025),
        Status=random.randint(0, 1),
        Life_expectancy=random.uniform(50.0, 90.0),
        Adult_Mortality=random.uniform(50, 400),
        Infant_deaths=random.uniform(0, 100),
        Alcohol=random.uniform(0.0, 15.0),
        Percentage_expenditure=random.uniform(0.0, 500.0),
        Hepatitis_B=random.uniform(0.0, 100.0),
        Measles=random.uniform(0, 10000),
        BMI=random.uniform(10.0, 40.0),
        Under_five_deaths=random.uniform(0, 100),
        Polio=random.uniform(0.0, 100.0),
        Total_expenditure=random.uniform(0.0, 15.0),
        Diphtheria=random.uniform(0.0, 100.0),
        HIV_AIDS=random.uniform(0.0, 10.0),
        GDP=random.uniform(100.0, 100000.0),
        Population=random.randint(1000, 100000000),
        Thinness_1_19_years=random.uniform(0.0, 15.0),
        Thinness_5_9_years=random.uniform(0.0, 15.0),
        Income_composition_of_resources=random.uniform(0.0, 1.0),
        Schooling=random.uniform(0.0, 20.0)
    )