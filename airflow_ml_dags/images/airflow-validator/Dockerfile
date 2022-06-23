FROM python:3.8.13-slim-buster

COPY validator.py /validator.py
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "validator.py"]
