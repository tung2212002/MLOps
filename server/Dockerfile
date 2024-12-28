FROM python:3.10-slim-buster

COPY . /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=120 -r requirements.txt

RUN pip install uvicorn==0.23.2

EXPOSE 8080

CMD uvicorn api:app --host 0.0.0.0 --port 8080 --workers 3 --log-level debug
