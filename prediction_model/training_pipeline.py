import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from prediction_model.processing.data_handling import load_dataset
from prediction_model.config import config

# Set MLflow URI
mlflow.set_tracking_uri(config.TRACKING_URI)

# Load the dataset
def get_data(input_file):
    data = load_dataset(input_file)
    return data

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config_params = load_config(config.CONFIG_FILE)

life_expectancy_df = get_data(config.DATASETS_FILE)
life_expectancy_df = life_expectancy_df.drop(columns = config.DROP_FEATURES)
life_expectancy_df['Status'].unique()
life_expectancy_df = pd.get_dummies(life_expectancy_df, columns = ['Status'])
life_expectancy_df = life_expectancy_df.apply(lambda x: x.fillna(x.mean()),axis=0)

# Load and preprocess the data
X = life_expectancy_df.drop(columns = [config.TARGET])
y = life_expectancy_df[[config.TARGET]]
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Define the search space for hyperparameter tuning
search_space = {
    'max_depth': hp.choice('max_depth', np.array(config_params['search_space']['max_depth'], dtype=int)),
    'learning_rate': hp.uniform('learning_rate', config_params['search_space']['learning_rate']['min'], config_params['search_space']['learning_rate']['max']),
    'n_estimators': hp.choice('n_estimators', np.array(config_params['search_space']['n_estimators'], dtype=int)),
    'subsample': hp.uniform('subsample', config_params['search_space']['subsample']['min'], config_params['search_space']['subsample']['max']),
    'colsample_bytree': hp.uniform('colsample_bytree', config_params['search_space']['colsample_bytree']['min'], config_params['search_space']['colsample_bytree']['max']),
    'gamma': hp.uniform('gamma', config_params['search_space']['gamma']['min'], config_params['search_space']['gamma']['max']),
    'reg_alpha': hp.uniform('reg_alpha', config_params['search_space']['reg_alpha']['min'], config_params['search_space']['reg_alpha']['max']),
    'reg_lambda': hp.uniform('reg_lambda', config_params['search_space']['reg_lambda']['min'], config_params['search_space']['reg_lambda']['max'])
}

# Objective function for hyperparameter tuning
def objective(params):
    # Create an XGBoost regressor with the given hyperparameters
    reg = xgb.XGBRegressor(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        eval_metric='rmse'
    )

    regression_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('regressor', reg)
    ])
    
    # Train the pipeline and log the metrics
    mlflow.xgboost.autolog()
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run(nested=True):
        # Fit the model on training data
        regression_pipeline.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = regression_pipeline.predict(X_test)

        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metrics({
            'mse': mse,
            'r2_score': r2
        })

        # Log the trained model
        mlflow.sklearn.log_model(regression_pipeline, "LifeExpectancy-prediction-model")

    # Return the loss for hyperparameter optimization (minimizing the MSE)
    return {'loss': mse, 'status': STATUS_OK}

# Hyperparameter tuning using Hyperopt
trials = Trials()
best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=5, trials=trials)

# Output the best hyperparameters
print("Best hyperparameters:", best_params)
