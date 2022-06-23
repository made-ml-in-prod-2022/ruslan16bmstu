FROM python:3.8.13-slim-buster

COPY preprocessor.py /preprocessor.py
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "preprocessor.py"]
