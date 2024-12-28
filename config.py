import os

class Config:
    # Backend information
    DEBUG = os.getenv('DEBUG', 'False') == 'True'  # Biến DEBUG có giá trị True nếu được set là 'True'
    ENABLE_OPENAPI = os.getenv('ENABLE_OPENAPI', 'False') == 'True'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    WORKERS_COUNT = int(os.getenv('WORKERS_COUNT', 1))
    API_PREFIX = os.getenv('API_PREFIX', '')

    # CORS information
    CORS_ALLOW_CREDENTIALS = os.getenv('CORS_ALLOW_CREDENTIALS', 'True') == 'True'
    CORS_ALLOW_METHODS = os.getenv('CORS_ALLOW_METHODS', '*').split(',')
    CORS_ALLOW_HEADERS = os.getenv('CORS_ALLOW_HEADERS', '*').split(',')
    CORS_ALLOW_ORIGIN = os.getenv('CORS_ALLOW_ORIGIN', '*').split(',')

    # MySQL configuration
    MYSQL_USER = os.getenv('MYSQL_USER', '')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_HOST = os.getenv('MYSQL_HOST', '')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', '')

    # S3 Artifact Storage
    S3_ARTIFACT_ROOT = os.getenv('S3_ARTIFACT_ROOT', '')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', '')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    AWS_REGION = os.getenv('AWS_REGION', 'ap-southeast-1')

    # MLflow Tracking
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', '')
    DVC_REMOTE_URL = os.getenv('DVC_REMOTE_URL', '')

    # MongoDB configuration
    MONGODB_ADDRESS = os.getenv('MONGODB_ADDRESS', 'mongodb://mongo:27017')

    # Evidently service
    EVIDENTLY_SERVICE = os.getenv('EVIDENTLY_SERVICE', '')
    RUN_ID = os.getenv('RUN_ID', '')

    # Logging information
    LOG_LEVEL = int(os.getenv('LOG_LEVEL', 10))

settings = Config()