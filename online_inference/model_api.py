from fastapi import FastAPI
from sklearn.pipeline import Pipeline
from pydantic import BaseModel, conlist
from typing import List, Union, Optional
import pandas as pd
import uvicorn
import pickle
import os


class Request(BaseModel):
    data: List[conlist(Union[float, int], min_items=13, max_items=13)]
    features: List[str]


class Response(BaseModel):
    predicts: List[int]


model: Optional[Pipeline] = None

app = FastAPI()


@app.get("/")
def root():
    return {"msg": "Heart disease classificator"}


@app.get("/predict", response_model=Response)
def make_predict(request: Request):
    df = pd.DataFrame(request.data, columns=request.features)
    targets = model.predict(df)
    return Response(predicts=[str(target) for target in targets])


@app.on_event("startup")
def start(model_path=None):
    global model

    if model_path is None:
        model_path = os.getenv("MODEL_PATH")

    if not os.path.isfile(model_path):
        raise RuntimeError(f"file {model_path} does not exists")

    model = read_model(model_path)


@app.get("/healz")
def health() -> bool:
    if model is None:
        return False
    else:
        return True


def read_model(model_path: str) -> Pipeline:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=os.getenv("PORT", 8000))
