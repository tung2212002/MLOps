from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp 
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from prediction_model.processing.preprocessing import MeanImputer, ModeImputer, DropColumns


preprocessing_pipeline = Pipeline([
    ('MeanImputation', pp.MeanImputer(variables=['Adult Mortality', 'Alcohol', 'Hepatitis B', ' BMI ', 'Total expenditure', 'GDP', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling'])),
    ('ModeImputation', pp.ModeImputer(variables=['Status'])),
    ('DropColumns', pp.DropColumns(variables_to_drop=['Country'])),
    ('ScaleFeatures', MinMaxScaler())
])
