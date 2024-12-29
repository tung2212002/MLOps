import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from prediction_model.config import config  

# Function to generate predictions from the best model stored in MLflow
def generate_predictions(data_input):
    # Convert the input data into a DataFrame
    data = pd.DataFrame(data_input)
    
    # Get the experiment name from the config
    experiment_name = config.EXPERIMENT_NAME
    
    # Fetch the experiment and get the best run based on metrics
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs_df = mlflow.search_runs(experiment_ids=experiment_id, order_by=['metrics.r2_score DESC'])
    
    # Get the best run (based on R2 score) and load the best model
    best_run = runs_df.iloc[0]
    best_run_id = best_run['run_id']
    best_model = 'runs:/' + best_run_id + config.MODEL_NAME
    life_expectancy_model = mlflow.sklearn.load_model(best_model)

    # Make predictions using the loaded model
    prediction = life_expectancy_model.predict(data)
    
    # Return the predictions in the required format (e.g., for regression predictions)
    result = prediction.tolist()
    return result[0]

if __name__ == '__main__':
    # This would be an example of how the predictions function could be called
    sample_input = [
        {
            'Year': 2015,
            'Status_Developing': 1,  # One-hot encoded for 'Developing'
            'Life expectancy ': 65.0,
            'Adult Mortality': 263.0,
            'Infant_deaths': 62.0,
            'Alcohol': 0.01,
            'Percentage_expenditure': 71.27962362,
            'Hepatitis B': 65.0,
            'Measles': 1154.0,
            'BMI': 19.1,
            'Under_five_deaths': 83.0,
            'Polio': 6.0,
            'Total expenditure': 8.16,
            'Diphtheria': 65.0,
            'HIV/AIDS': 0.1,
            'GDP': 584.25921,
            'Population': 33736494,
            'Thinness_1_19_years': 17.2,
            'Thinness_5_9_years': 17.3,
            'Income_composition_of_resources': 0.479,
            'Schooling': 10.1
        }
    ]
    
    predictions = generate_predictions(sample_input)
    print(predictions)
