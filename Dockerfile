FROM python:3.8-slim-bullseye

WORKDIR /app

COPY . .



RUN pip install -r requirements.txt

CMD ["python", "app.py"]
