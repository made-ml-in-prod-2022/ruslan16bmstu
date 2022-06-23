FROM python:3.8.13-slim-buster

COPY faker_generator.py /faker_generator.py
COPY generator.py /generator.py
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "generator.py"]
