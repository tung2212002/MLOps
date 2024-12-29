import pathlib
import os


current_directory = os.path.dirname(os.path.realpath(__file__)) #current directory of the script

PACKAGE_ROOT = os.path.dirname(current_directory) #parent directory of current directory


# PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

DATASETS_FILE = "LifeExpectancyData.csv"

TARGET = 'Life expectancy '

DROP_FEATURES = [
    'Country',
]

S3_BUCKET = "mlops-it5414-project"

FOLDER="datadrift"

TRACKING_URI="http://ec2-54-173-194-157.compute-1.amazonaws.com:5000"


EXPERIMENT_NAME="life_expectancy_prediction_model"

MODEL_NAME="/LifeExpectancy-prediction-model"

CONFIG_FILE="/app/prediction_model/config.yaml"
