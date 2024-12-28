import pytest
import mlflow
import pandas as pd
from prediction_model.config import config
from prediction_model.predict import generate_predictions
from prediction_model.processing.data_handling import load_dataset

# Set the MLflow tracking URI
mlflow.set_tracking_uri(config.TRACKING_URI)

@pytest.fixture
def random_data_point():
    life_expectancy_df = load_dataset(config.DATASETS_FILE)
    
    single_row = life_expectancy_df.sample(n=1, random_state=42)  # Lấy một dòng ngẫu nhiên
    return single_row

@pytest.fixture
def single_prediction(random_data_point):
    result = generate_predictions(random_data_point)
    return result

def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None

def test_single_pred_numeric_type(single_prediction):
    prediction = single_prediction.get('predictions')[0]
    assert isinstance(prediction, (float, int))

def test_single_pred_value_range(single_prediction):
    prediction = single_prediction.get('predictions')[0]
    assert 0 <= prediction <= 150

def test_single_pred_valid_result(single_prediction):
    prediction = single_prediction.get('predictions')[0]
    assert prediction is not None
    assert not pd.isna(prediction)