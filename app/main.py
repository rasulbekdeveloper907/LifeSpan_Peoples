import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Model path (Windows compatible)
# --------------------------------------------------
MODEL_PATH = Path(r"C:\Users\Rasulbek907\Desktop\ML_Engineer_Salary_Prediction\Models\xgb_model.pkl")

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="💼 ML Engineer Salary Prediction API",
    version="1.0"
)

model = None

# --------------------------------------------------
# Load model on startup
# --------------------------------------------------
@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        model = None

# --------------------------------------------------
# Input Schema (FEATURES)
# --------------------------------------------------
class SalaryInput(BaseModel):
    work_year: int
    experience_level: str
    employment_type: str
    job_title: str
    salary_currency: str
    employee_residence: str
    remote_ratio: int
    company_location: str
    company_size: str

# --------------------------------------------------
# Output Schema (REGRESSION)
# --------------------------------------------------
class SalaryPrediction(BaseModel):
    predicted_salary_usd: float

# --------------------------------------------------
# Health endpoints
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "running", "model": "XGBoost Salary Regressor"}

@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# Predict endpoint
# --------------------------------------------------
@app.post("/predict", response_model=SalaryPrediction)
def predict(data: SalaryInput):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([data.model_dump()])

    try:
        prediction = model.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("Predicted salary: %.2f USD", prediction)

    return SalaryPrediction(
        predicted_salary_usd=round(float(prediction), 2)
    )
