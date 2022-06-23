FROM python:3.8.13-slim-buster

COPY predictor.py /predictor.py
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "predictor.py"]
