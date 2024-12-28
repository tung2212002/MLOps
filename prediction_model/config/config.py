import pathlib
import os


current_directory = os.path.dirname(os.path.realpath(__file__)) #current directory of the script

PACKAGE_ROOT = os.path.dirname(current_directory) #parent directory of current directory


# PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

# TRAIN_FILE = 'train.csv'
# TEST_FILE = 'test.csv'
DATASETS_FILE = "LifeExpectancyData.csv"


# TARGET = 'Loan_Status'
TARGET = 'Life expectancy '

FEATURES = [
    "Adult Mortality",
    "Alcohol",
    " BMI ",
    "Schooling"
]

NUM_FEATURES = [
    "Adult Mortality",
    "Alcohol",
    " BMI ",
    "Schooling"
]

CAT_FEATURES = [
]

# Các cột cần bỏ (nếu có)
DROP_FEATURES = [
    'Country',
]

# Các cột cần áp dụng mã hóa (encoding)
FEATURES_TO_ENCODE = CAT_FEATURES

# Các cột cần áp dụng log transformation
LOG_FEATURES = [
    "Adult Mortality",
    "infant deaths",
    "percentage expenditure",
    "Measles ",
    "under-five deaths ",
    "GDP",
    "Population"
]

# Các cột cần xử lý domain-specific
FEATURE_TO_MODIFY = []  # Nếu cần thực hiện phép toán trên các cột
FEATURE_TO_ADD = []     # Các cột được thêm để phục vụ phép toán

S3_BUCKET = "life-expectancy-prediction-model"

FOLDER="datadrift"

TRACKING_URI="http://ec2-54-173-194-157.compute-1.amazonaws.com:5001"


EXPERIMENT_NAME="life_expectancy_prediction_model"

MODEL_NAME="/LifeExpectancy-prediction-model"

