FROM python:3.8.13-slim-buster

COPY trainer.py /trainer.py
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "trainer.py"]
