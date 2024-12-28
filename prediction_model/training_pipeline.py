import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# import prediction_model.pipeline as preprocessing_pipeline
import prediction_model.processing.preprocessing as pp 
from prediction_model.processing.data_handling import load_dataset
from prediction_model.config import config
from server.config import settings

# Set MLflow URI
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

# Load the dataset
def get_data(input_file):
    data = load_dataset(input_file)
    # X = data[config.FEATURES]
    # y = data[config.TARGET]  
    # return X, y
    return data

life_expectancy_df = get_data(config.DATASETS_FILE)
life_expectancy_df = life_expectancy_df.drop(columns = ['Country'])
life_expectancy_df['Status'].unique()
life_expectancy_df = pd.get_dummies(life_expectancy_df, columns = ['Status'])
life_expectancy_df = life_expectancy_df.apply(lambda x: x.fillna(x.mean()),axis=0)

# Load and preprocess the data
# X, y = get_data(config.DATASETS_FILE)
X = life_expectancy_df.drop(columns = ['Life expectancy '])
y = life_expectancy_df[['Life expectancy ']]
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# scaler_X = StandardScaler()
# X_train = scaler_X.fit_transform(X_train)
# X_test = scaler_X.transform(X_test)

# scaler_y = StandardScaler()
# y_train = scaler_y.fit_transform(y_train)
# y_test = scaler_y.transform(y_test)


# preprocessing_pipeline = Pipeline([
#     ('MeanImputation', pp.MeanImputer(variables=['Adult Mortality', 'Alcohol', 'Hepatitis B', ' BMI ', 'Total expenditure', 'GDP', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling'])),
#     ('ModeImputation', pp.ModeImputer(variables=['Status'])),
#     ('DropColumns', pp.DropColumns(variables_to_drop=['Country'])),
#     ('ScaleFeatures', MinMaxScaler())
# ])
# Apply preprocessing pipeline (ensure this pipeline is correct in your preprocessing file)
# X = preprocessing_pipeline.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the search space for hyperparameter tuning
search_space = {
    'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.choice('n_estimators', np.arange(50, 300, 50, dtype=int)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'gamma': hp.uniform('gamma', 0, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1)
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

    # Define the pipeline with preprocessing and model
    # regression_pipeline = Pipeline([
    #     ('Preprocessing', preprocessing_pipeline),  # Ensure this is a valid transformer pipeline
    #     ('Regressor', reg)
    # ])

    regression_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('regressor', reg)
    ])
    
    # Train the pipeline and log the metrics
    mlflow.xgboost.autolog()
    mlflow.set_experiment("life_expectancy_prediction")
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
        mlflow.sklearn.log_model(regression_pipeline, "life_expectancy_model")

    # Return the loss for hyperparameter optimization (minimizing the MSE)
    return {'loss': mse, 'status': STATUS_OK}

# Hyperparameter tuning using Hyperopt
trials = Trials()
best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=5, trials=trials)

# Output the best hyperparameters
print("Best hyperparameters:", best_params)
