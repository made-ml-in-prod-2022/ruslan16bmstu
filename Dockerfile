FROM python:3.8.13-slim-buster

COPY online_inference/requirements.txt ./requirements.txt
COPY online_inference/model_api.py ./model_api.py
COPY online_inference/model_object/model.pkl /model.pkl

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR .

ENV MODEL_PATH="/model.pkl"
ENV HOST="0.0.0.0"
ENV PORT="8000"

CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8000"]
