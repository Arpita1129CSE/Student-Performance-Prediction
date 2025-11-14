# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Prediction System")

BASE = Path(__file__).parent

# Try loading artifacts
pipeline = None
model = None
scaler = None
label_encoder = None
feature_columns = None


# 2) try model + optional scaler + features + label encoder
if pipeline is None:
    if (BASE / "student_model.pkl").exists():
        model = joblib.load(BASE / "student_model.pkl")
    if (BASE / "scaler.pkl").exists():
        scaler = joblib.load(BASE / "scaler.pkl")
    if (BASE / "label_encoder.pkl").exists():
        label_encoder = joblib.load(BASE / "label_encoder.pkl")
    if (BASE / "features.pkl").exists():
        feature_columns = joblib.load(BASE / "features.pkl")

st.markdown("---")
st.subheader("📋 Enter Student Details")

# Default UI fields (match your feature names & order)
cols_left, cols_right = st.columns(2)
with cols_left:
    attendance = st.slider("Attendance (%)", 0, 100, 85)
    midterm_score = st.slider("Midterm Score", 0, 100, 70)
    final_score = st.slider("Final Score", 0, 100, 75)
    assignments_avg = st.slider("Assignments Average", 0, 100, 80)
    quizzes_avg = st.slider("Quizzes Average", 0, 100, 70)
    participation_score = st.slider("Participation Score", 0, 100, 75)
    projects_score = st.slider("Projects Score", 0, 100, 80)

with cols_right:
    total_score = st.slider("Total Score", 0, 100, 78)
    study_hours = st.slider("Study Hours per Week", 0, 40, 15)
    extracurricular = st.selectbox("Extracurricular Activities (0/1)", [0, 1])
    internet_access = st.selectbox("Internet Access at Home (0/1)", [0, 1])
    parent_education = st.selectbox("Parent Education Level (0-3)", [0,1,2,3])
    stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
    sleep_hours = st.slider("Sleep Hours per Night", 0, 12, 7)

# Build input DataFrame (columns must match model expectation)
input_df = pd.DataFrame([[
    attendance, midterm_score, final_score, assignments_avg, quizzes_avg,
    participation_score, projects_score, total_score, study_hours,
    extracurricular, internet_access, parent_education, stress_level, sleep_hours
]], columns=[
    'Attendance (%)','Midterm_Score','Final_Score','Assignments_Avg','Quizzes_Avg',
    'Participation_Score','Projects_Score','Total_Score','Study_Hours_per_Week',
    'Extracurricular_Activities','Internet_Access_at_Home','Parent_Education_Level',
    'Stress_Level (1-10)','Sleep_Hours_per_Night'
])

st.write("Input preview:")
st.dataframe(input_df)

# Predict when button pressed
if st.button("🔮 Predict Grade"):
    try:
        if pipeline is not None:
            pred = pipeline.predict(input_df)[0]
        else:
            # align columns if features.pkl exists
            if feature_columns is not None:
                input_df = input_df[feature_columns]
            # apply scaler if present
            if scaler is not None:
                X = scaler.transform(input_df)
            else:
                X = input_df.values
            pred = model.predict(X)[0]
        # if label encoder was used, inverse transform
        if label_encoder is not None:
            pred_display = label_encoder.inverse_transform([pred])[0]
        else:
            pred_display = pred
        st.success(f"🎯 Predicted Grade: **{pred_display}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

st.markdown("---")
st.caption("Make sure saved model artifacts (student_model.pkl or pipeline.pkl) are in the same folder as this file.")