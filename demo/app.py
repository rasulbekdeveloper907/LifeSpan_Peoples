import gradio as gr
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------
# Model yuklash
# ---------------------------
MODEL_PATH = Path("Models/GradientBoostingRegressor.joblib")
model = joblib.load(MODEL_PATH)

# ---------------------------
# Predict function
# ---------------------------
def predict_lifespan(name, birth_date, birth_place, death_date, death_place,
                     occupation, awards, alma_mater, education, spouse,
                     children, occupation_cluster, birth_year, death_year,
                     life_span_cluster, edu_award_cluster, bio_cluster):
    df = pd.DataFrame([{
        "name": name,
        "birth_date": birth_date,
        "birth_place": birth_place,
        "death_date": death_date,
        "death_place": death_place,
        "occupation": occupation,
        "awards": awards,
        "alma_mater": alma_mater,
        "education": education,
        "spouse": spouse,
        "children": children,
        "occupation_cluster": occupation_cluster,
        "birth_year": birth_year,
        "death_year": death_year,
        "life_span_cluster": life_span_cluster,
        "edu_award_cluster": edu_award_cluster,
        "bio_cluster": bio_cluster
    }])
    prediction = float(model.predict(df)[0])
    # Number output va progress uchun ikkita qiymat
    return round(prediction, 2), round(prediction, 2)

# ---------------------------
# Dashboard
# ---------------------------
with gr.Blocks(title="💀 Life Span Prediction Dashboard") as demo:

    # Header
    gr.Markdown(
        """
        # 🧬 Life Span Prediction
        Predict human life span (years) using GradientBoostingRegressor
        """
    )

    with gr.Row():
        # ---------------------------
        # Personal Info Card
        # ---------------------------
        with gr.Column(scale=1):
            with gr.Box():  # Gradio 6.3.0 da ishlaydi
                gr.Markdown("### 🧑 Personal Info")
                name = gr.Textbox(label="Name", placeholder="Enter full name")
                birth_date = gr.Textbox(label="Birth Date (YYYY-MM-DD)", placeholder="YYYY-MM-DD")
                birth_place = gr.Textbox(label="Birth Place")
                death_date = gr.Textbox(label="Death Date (YYYY-MM-DD)", placeholder="YYYY-MM-DD")
                death_place = gr.Textbox(label="Death Place")

        # ---------------------------
        # Career & Bio Card
        # ---------------------------
        with gr.Column(scale=1):
            with gr.Box():
                gr.Markdown("### 🏆 Career & Bio")
                occupation = gr.Textbox(label="Occupation")
                awards = gr.Number(label="Awards Count", value=0, precision=0)
                alma_mater = gr.Textbox(label="Alma Mater")
                education = gr.Textbox(label="Education Level")
                spouse = gr.Number(label="Spouse Count", value=0, precision=0)
                children = gr.Number(label="Children Count", value=0, precision=0)
                occupation_cluster = gr.Number(label="Occupation Cluster", value=0, precision=0)
                birth_year = gr.Number(label="Birth Year", value=0, precision=0)
                death_year = gr.Number(label="Death Year", value=0, precision=0)
                life_span_cluster = gr.Number(label="Life Span Cluster", value=0, precision=0)
                edu_award_cluster = gr.Number(label="Education & Awards Cluster", value=0, precision=0)
                bio_cluster = gr.Number(label="Bio Cluster", value=0, precision=0)

    # ---------------------------
    # Predict Button va Outputs
    # ---------------------------
    predict_btn = gr.Button("Predict 🏁", variant="primary")

    life_span_output = gr.Number(label="🧪 Predicted Life Span (years)", interactive=False)
    life_span_progress = gr.Progress(label="Life Span Gauge", minimum=0, maximum=120, value=0)

    predict_btn.click(
        fn=predict_lifespan,
        inputs=[
            name, birth_date, birth_place, death_date, death_place,
            occupation, awards, alma_mater, education, spouse,
            children, occupation_cluster, birth_year, death_year,
            life_span_cluster, edu_award_cluster, bio_cluster
        ],
        outputs=[life_span_output, life_span_progress]
    )

    # Footer
    gr.Markdown(
        """
        ---
        Built by **Your Name** | GradientBoostingRegressor | ML Dashboard
        """
    )

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
