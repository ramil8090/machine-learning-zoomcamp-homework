import pickle
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

from fastapi import FastAPI
import uvicorn

model_name = './pipeline_v2.bin'

class InputData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

class PredictResponse(BaseModel):
    probability: float

app = FastAPI(title="prediction")

with open(model_name, 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(input_data):
    result = pipeline.predict_proba(input_data)[0, 1]
    return float(result)

@app.post("/predict")
def predict(input_data: InputData) -> PredictResponse:
    prob = predict_single(input_data.model_dump())

    return PredictResponse(
        probability=prob
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9696)