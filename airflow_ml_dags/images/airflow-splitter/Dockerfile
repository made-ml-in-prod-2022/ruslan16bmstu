FROM python:3.8.13-slim-buster

COPY splitter.py /splitter.py
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "splitter.py"]
