import joblib
import pandas as pd
import gradio as gr
from pathlib import Path
import logging

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Model path (Docker-safe)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Models" / "GradientBoostingRegressor.joblib"

# --------------------------------------------------
# Load model
# --------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    model = None


# --------------------------------------------------
# Predict function
# --------------------------------------------------
def predict_life_span(
    name,
    birth_date,
    birth_place,
    death_date,
    death_place,
    occupation,
    awards,
    alma_mater,
    education,
    spouse,
    children,
    occupation_cluster,
    birth_year,
    death_year,
    life_span_cluster,
    edu_award_cluster,
    bio_cluster
):
    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([{
        "name": float(name),
        "birth_date": float(birth_date),
        "birth_place": float(birth_place),
        "death_date": float(death_date),
        "death_place": float(death_place),
        "occupation": float(occupation),
        "awards": float(awards),
        "alma_mater": float(alma_mater),
        "education": float(education),
        "spouse": float(spouse),
        "children": float(children),
        "occupation_cluster": float(occupation_cluster),
        "birth_year": float(birth_year),
        "death_year": float(death_year),
        "life_span_cluster": float(life_span_cluster),
        "edu_award_cluster": float(edu_award_cluster),
        "bio_cluster": float(bio_cluster)
    }])

    prediction = model.predict(df)[0]

    return {
        "predicted_life_span_years": round(float(prediction), 2)
    }


# --------------------------------------------------
# Gradio UI
# --------------------------------------------------
demo = gr.Interface(
    fn=predict_life_span,
    inputs=[
        gr.Number(label="Name (Encoded)", value=1),
        gr.Number(label="Birth Date (Encoded)", value=0),
        gr.Number(label="Birth Place (Encoded)", value=5),
        gr.Number(label="Death Date (Encoded)", value=0),
        gr.Number(label="Death Place (Encoded)", value=3),
        gr.Number(label="Occupation (Encoded)", value=2),
        gr.Number(label="Awards Count", value=1),
        gr.Number(label="Alma Mater (Encoded)", value=4),
        gr.Number(label="Education Level (Encoded)", value=3),
        gr.Number(label="Spouse (Encoded)", value=1),
        gr.Number(label="Children Count", value=2),
        gr.Number(label="Occupation Cluster", value=1),
        gr.Number(label="Birth Year", value=1950),
        gr.Number(label="Death Year", value=2020),
        gr.Number(label="Life Span Cluster", value=2),
        gr.Number(label="Education-Award Cluster", value=1),
        gr.Number(label="Bio Cluster", value=0)
    ],
    outputs=gr.JSON(label="Life Span Prediction"),
    title="🧬 Life Span Prediction App",
    description="GradientBoostingRegressor | Predict human life span (years)"
)

# --------------------------------------------------
# Launch app
# --------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
