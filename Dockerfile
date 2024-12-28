FROM python:3.10-slim-buster

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Initialize git 
RUN git init

COPY prediction_model/datasets/*.dvc ./prediction_model/datasets/
COPY .dvc/ ./.dvc/
COPY requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install dvc[s3]

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_REGION
ARG MYSQL_DATABASE
ARG MYSQL_HOST
ARG MYSQL_PASSWORD
ARG MYSQL_PORT
ARG MYSQL_USER

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_REGION
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Copy all files
COPY . .

# RUN touch prediction_model/__init__.py && \
RUN    dvc remote modify mlopsremote access_key_id ${AWS_ACCESS_KEY_ID} && \
    dvc remote modify mlopsremote secret_access_key ${AWS_SECRET_ACCESS_KEY} && \
    dvc remote modify mlopsremote region ${AWS_DEFAULT_REGION} && \
    dvc pull -v --force && \
    python prediction_model/training_pipeline.py

    # entrypoint: mlflow server --backend-store-uri mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@${MYSQL_HOST}:${MYSQL_PORT}/${MYSQL_DATABASE} --default-artifact-root=s3://mlflow/ --host=0.0.0.0 --port=5000
# CMD ["mlflow", "server", "--backend-store-uri", "mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@${MYSQL_HOST}:${MYSQL_PORT}/${MYSQL_DATABASE}", "--default-artifact-root", "s3://mlflow/", "--host", "0.0.0.0", "--port", "5000"]

# CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
# && \
#     pytest -v /app/tests/test_prediction.py && \
#     pytest --junitxml=/app/tests/test-results.xml /app/tests/test_prediction.py