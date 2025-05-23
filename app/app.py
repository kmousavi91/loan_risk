from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import load_model

app = FastAPI(title="Loan Risk Prediction API")

model, scaler, columns = load_model("models/model.joblib", "models/scaler.joblib", "models/columns.joblib")

class ApplicantData(BaseModel):
    features: list

@app.get("/")
def root():
    return {"message": "Loan Risk API is running. Use POST /predict to submit data."}

@app.post("/predict")
def predict(applicant: ApplicantData):
    try:
        input_data = np.array(applicant.features).reshape(1, -1)
        if input_data.shape[1] != len(columns):
            raise ValueError(f"Expected {len(columns)} features, got {input_data.shape[1]}")
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        return {"approved": bool(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
