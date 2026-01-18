import joblib
import pandas as pd
import gradio as gr
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# Model path (Docker-safe)
# ===========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Models" / "xgb_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    model = None


# ===========================
# Predict function
# ===========================
def predict_salary(
    work_year,
    experience_level,
    employment_type,
    job_title,
    salary_currency,
    employee_residence,
    remote_ratio,
    company_location,
    company_size
):

    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([{
        "work_year": int(work_year),
        "experience_level": experience_level,
        "employment_type": employment_type,
        "job_title": job_title,
        "salary_currency": salary_currency,
        "employee_residence": employee_residence,
        "remote_ratio": int(remote_ratio),
        "company_location": company_location,
        "company_size": company_size
    }])

    predicted_salary = float(model.predict(df)[0])

    return {
        "predicted_salary_usd": round(predicted_salary, 2)
    }


# ===========================
# Gradio UI
# ===========================
demo = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Number(label="Work Year", precision=0, value=2024),
        gr.Dropdown(["EN", "MI", "SE", "EX"], label="Experience Level"),
        gr.Dropdown(["FT", "PT", "CT", "FL"], label="Employment Type"),
        gr.Textbox(label="Job Title", value="Machine Learning Engineer"),
        gr.Textbox(label="Salary Currency", value="USD"),
        gr.Textbox(label="Employee Residence", value="US"),
        gr.Slider(0, 100, step=50, label="Remote Ratio"),
        gr.Textbox(label="Company Location", value="US"),
        gr.Dropdown(["S", "M", "L"], label="Company Size")
    ],
    outputs=gr.JSON(label="Salary Prediction"),
    title="ML Engineer Salary Predictor",
    description="XGBoost Regressor | Predict annual salary (USD)"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
