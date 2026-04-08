from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from fastapi.middleware.cors import CORSMiddleware 

# Load model
model = joblib.load("adult_income_pipeline.pkl")

app = FastAPI(title="Adult Income Prediction API")

# ⭐ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # For development (allow all)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Schema
class Person(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

# API Endpoint
@app.post("/predict")
def predict(data: Person):

    df = pd.DataFrame([data.dict()])

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    return {
        "income_class": ">50K" if pred == 1 else "<=50K",
        "probability_<=50K": round(float(proba[0]), 3),
        "probability_>50K": round(float(proba[1]), 3)
    }