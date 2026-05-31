
FROM python:3.12.7-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_docker.txt .

RUN pip install --no-cache-dir -r requirements_docker.txt

COPY MLProject/ /app/MLProject/

EXPOSE 5001

CMD ["mlflow", "models", "serve", "-m", "/app/MLProject/mlruns/0/", "-h", "0.0.0.0", "-p", "5001", "--env-manager", "local"]