import gradio as gr
import numpy as np
import pandas as pd
import joblib

model = joblib.load("models/lgbm_model.pkl")

CAT_COLS = [
    "course",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty",
]

FEATURE_ORDER = list(model.feature_names_in_)
# ['age', 'gender', 'course', 'study_hours', 'class_attendance',
#  'internet_access', 'sleep_hours', 'sleep_quality', 'study_method',
#  'facility_rating', 'exam_difficulty']


def predict_score(
    attendance,
    study_hours,
    sleep_hours,
    course,
    study_method,
    sleep_quality,
    facility_rating,
    exam_difficulty,
):
  
  
    raw = {
        "course": course,
        "study_hours": study_hours,
        "class_attendance": attendance,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "study_method": study_method,
        "facility_rating": facility_rating,
        "exam_difficulty": exam_difficulty,

        # default / fixed values
        "age": 21,
        "gender": "other",
        "internet_access": "yes",
    }

    df = pd.DataFrame([raw])[FEATURE_ORDER]

    for col in CAT_COLS:
        df[col] = df[col].astype("category")

    score = float(model.predict(df)[0])
    score = max(0.0, min(100.0, score))

    # ── Colour + recommendation logic ─────────────────────────
    if score >= 80:
        bar_color = "#16a34a"
        recommendation = "Excellent! Keep up the great work!"
    elif score >= 60:
        bar_color = "#3b82f6"
        recommendation = (
            "Good performance! Consider increasing study hours "
            "for even better results."
        )
    else:
        bar_color = "#dc2626"
        recommendation = (
            "Focus on improving class attendance and study hours "
            "for better results!"
        )


    radius = 80
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - score / 100)

    return f"""
    <div style="text-align: center; padding: 30px;">
        <h2 style="color: #2563eb; margin-bottom: 10px;">Your Predicted Exam Score</h2>
        <div style="position: relative; width: 200px; height: 200px; margin: 30px auto;">
            <svg width="200" height="200" style="transform: rotate(-90deg);">
                <circle cx="100" cy="100" r="{radius}" fill="none" stroke="#e5e7eb" stroke-width="20"/>
                <circle cx="100" cy="100" r="{radius}" fill="none" stroke="{bar_color}" stroke-width="20"
                        stroke-dasharray="{circumference}"
                        stroke-dashoffset="{offset}"
                        stroke-linecap="round"/>
            </svg>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
                <div style="font-size: 48px; font-weight: bold; color: {bar_color};">{score:.1f}</div>
                <div style="font-size: 12px; color: #6b7280;">out of 100</div>
                <div style="font-size: 10px; color: #9ca3af; margin-top: 5px;">PREDICTED SCORE</div>
            </div>
        </div>
        <p style="color: #6b7280; margin-top: 20px; line-height: 1.6;">
            Based on the factors you provided, this is your estimated performance.<br>
            {recommendation}
        </p>
    </div>
    """


# ─── Custom CSS (unchanged) ──────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

/* Header Styling */
.header-container {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-radius: 20px;
    margin-bottom: 30px;
}

.main-title {
    font-size: 48px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #1e293b 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size: 18px;
    color: #64748b;
    margin-bottom: 20px;
}

/* Badge Styling */
.ai-badge {
    display: inline-block;
    background: #dbeafe;
    color: #3b82f6;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 20px;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-top: 30px;
}


.feature-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.12);
}

.feature-icon {
    font-size: 36px;
    margin-bottom: 15px;
}

.feature-title {
    font-size: 18px;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 8px;
}

.feature-desc {
    font-size: 14px;
    color: #64748b;
}

/* Input Styling */
.input-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 15px;
}

/* ===== INPUT GRID ===== */
.input-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}

/* Mobile fallback */
@media (max-width: 900px) {
    .input-grid {
        grid-template-columns: 1fr;
    }
}


.rank-badge {
    display: inline-block;
    background: #f1f5f9;
    color: #64748b;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    float: right;
}

/* Button Styling */
.predict-button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    border-radius: 10px !important;
    border: none !important;
    font-size: 16px !important;
    cursor: pointer !important;
    transition: transform 0.2s !important;
}

.predict-button:hover {
    transform: scale(1.05) !important;
}

/* Output Styling */
#output {
    background: white;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    padding: 20px;
    margin-top: 20px;
}
"""

# ─── Gradio UI ────────────────────────────────────────────────
with gr.Blocks(css=custom_css, title="PredictScore.AI - Student Score Predictor") as demo:

    # Header
    gr.HTML("""
        <div class="header-container">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <div style="background: #3b82f6; width: 60px; height: 60px; border-radius: 15px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                    <span style="color: white; font-size: 32px;">🎓</span>
                </div>
                <div style="text-align: left;">
                    <h1 style="margin: 0; font-size: 32px; font-weight: 700; color: #1e293b;">PredictScore.AI</h1>
                    <p style="margin: 0; color: #64748b; font-size: 14px;">Student Score Predictor</p>
                </div>
            </div>

            <div class="ai-badge">
                AI-Powered Predictor
            </div>

            <h2 class="main-title">Predict Your <span style="color: #3b82f6;">Exam Score</span></h2>
            <p class="subtitle">Enter your study habits, attendance, and other factors to get an AI-powered<br>
            prediction of your exam performance. Understand what impacts your grades most.</p>

            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Data-Driven</div>
                    <div class="feature-desc">Trained on large student data</div>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🧠</div>
                    <div class="feature-title">ML Based</div>
                    <div class="feature-desc">LightGBM Regressor</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-title">Instant Results</div>
                    <div class="feature-desc">Get predictions fast</div>
                </div>
            </div>
        </div>
    """)

    with gr.Row():
        # ================= INPUT =================
        with gr.Column(scale=2):
            gr.HTML("<h3 style='color: #ffffff; margin-bottom: 20px;'>Enter Your Information</h3>")

            # ---- GRID START ----
            gr.HTML("<div class='input-grid'>")

            # Row 1
            attendance = gr.Slider(
                minimum=0, maximum=100, value=75, step=1,
                label="📚 Class Attendance (%)"
            )

            study_hours = gr.Slider(
                minimum=0, maximum=12, value=4, step=0.5,
                label="⏰ Study Hours (per day)"
            )

            # Row 2
            sleep_hours = gr.Slider(
                minimum=0, maximum=12, value=7, step=0.5,
                label="😴 Sleep Hours (per night)"
            )

            study_method = gr.Dropdown(
                choices=["coaching", "self-study", "mixed", "group study", "online videos"],
                value="mixed",
                label="📚 Study Method"
            )

            # Row 3
            course = gr.Dropdown(
                choices=["b.tech", "b.sc", "b.com", "bca", "bba", "ba", "diploma"],
                value="b.tech",
                label="📖 Course"
            )

            facility_rating = gr.Dropdown(
                choices=["high", "medium", "low"],
                value="medium",
                label="🏫 Facility Rating"
            )

            # Row 4
            sleep_quality = gr.Dropdown(
                choices=["good", "average", "poor"],
                value="average",
                label="😴 Sleep Quality"
            )

            exam_difficulty = gr.Dropdown(
                choices=["easy", "moderate", "hard"],
                value="moderate",
                label="📊 Exam Difficulty"
            )

            gr.HTML("</div>")
            # ---- GRID END ----

          

        # ================= OUTPUT =================
        with gr.Column(scale=1):
            gr.HTML("<h3 style='color: #ffffff; margin-bottom: 20px;'>Prediction Result</h3>")
            output = gr.HTML(elem_id="output")

            # Buttons
            with gr.Row():
                predict_btn = gr.Button("✨ Predict Score", elem_classes="predict-button")
                reset_btn = gr.Button("🔄 Reset Form", variant="secondary")




    # ── All inputs in the order predict_score() expects ─────────
    ALL_INPUTS = [
        attendance, study_hours, sleep_hours, course,
        study_method, sleep_quality, facility_rating,
        exam_difficulty,
    ]

    predict_btn.click(fn=predict_score, inputs=ALL_INPUTS, outputs=output)

    def reset_form():
        return [
            75,          # attendance
            4,           # study_hours
            7,           # sleep_hours
            "b.tech",    # course
            "mixed",     # study_method
            "average",   # sleep_quality
            "medium",    # facility_rating
            "moderate",  # exam_difficulty
            ""           # output (clear result)
        ]

    reset_btn.click(
        fn=reset_form,
        outputs=ALL_INPUTS + [output],
    )

if __name__ == "__main__":
    demo.launch(share=True)