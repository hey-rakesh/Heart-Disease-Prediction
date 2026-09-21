import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "knn_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "knn_scaler.pkl")
column = joblib.load(BASE_DIR / "models" / "knn_column.pkl")

st.title("Heart Stroke Prediction by Rakesh ❤️")
st.markdown("Provide the following details...")


age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("SEX",["M", "F"])

chest_pain = st.selectbox("Chest Pain Type",["ATA", "NAP", "TA", "ASY"])

resting_bp = st.number_input("Resting Blood Pressure (mmHg)",80, 200,120)

cholestrol = st.number_input("Cholesterol (mg/dL)",100,600,200)

fasting_bd = st.selectbox("Fasting Blood Sugar > 120mg/dL",[0, 1])

resting_ecg = st.selectbox("Resting ECG",["Normal", "ST", "LVH"])

max_hr = st.slider("Max Heart Rate",60,220,150)

exercise_angina = st.selectbox("Exercise-Induced Angina",["Y", "N"])

old_peak = st.slider("OLD Peak (ST Depression)",0.0,6.0,1.0)

st_slope = st.selectbox("ST SLOPE",["Up", "Flat", "Down"])



if st.button("Predict"):

    raw_Data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholestrol,
        "FastingBS": fasting_bd,
        "MaxHR": max_hr,
        "Oldpeak": old_peak,

        "Sex_" + sex: 1,

        "ChestPainType_" + chest_pain: 1,

        "RestingECG_" + resting_ecg: 1,

        "ExerciseAngina_" + exercise_angina: 1,

        "ST_Slope_" + st_slope: 1
    }


    input_dataframe = pd.DataFrame([raw_Data])

   
    for col in column:
        if col not in input_dataframe.columns:
            input_dataframe[col] = 0

 
    input_dataframe = input_dataframe[column]


    scaler_input = scaler.transform(input_dataframe)


    prediction = model.predict(scaler_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")