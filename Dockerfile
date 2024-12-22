FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app 

COPY requirements.txt .
RUN pip install --upgrade pip 

RUN chmod +x /app/tests

RUN chmod +w /app/tests

RUN chmod +x /app/prediction_model

RUN chmod +w /app/prediction_model/trained_models

RUN chmod +w /app/prediction_model/datasets

ENV PYTHONPATH="${PYTHONPATH}:/app/prediction_model"

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install dvc[s3]


ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_REGION

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_REGION


RUN dvc remote modify mlopsremote access_key_id ${AWS_ACCESS_KEY_ID}
RUN dvc remote modify mlopsremote secret_access_key ${AWS_SECRET_ACCESS_KEY}
RUN dvc remote modify mlopsremote region ${AWS_DEFAULT_REGION}

RUN dvc pull -v --force

RUN python /app/prediction_model/training_pipeline.py && \
    pytest -v /app/tests/test_prediction.py && \
    pytest --junitxml=/app/tests/test-results.xml /app/tests/test_prediction.py

EXPOSE 8005

ENTRYPOINT ["python"]
CMD ["main.py"]