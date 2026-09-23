import streamlit as st
import pandas as pd
import pickle

# Load your saved model
model = pickle.load(open('heart_model.pkl', 'rb'))

st.title("Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk")

# Input fields
age = st.slider("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.slider("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])



p_value = st.number_input("Phone Number", 0, 100, 50)
name = st.text_input("patient Name")


# Encode inputs — using the SAME custom mapping from notebook
sex_enc = 10 if sex == "Male" else 20
chest_pain_enc = {"ATA": 5, "NAP": 10, "ASY": 15, "TA": 20}[chest_pain]
fasting_bs_enc = 1 if fasting_bs == "Yes" else 0
resting_ecg_enc = {"Normal": 3, "ST": 6, "LVH": 9}[resting_ecg]
exercise_angina_enc = 8 if exercise_angina == "Yes" else 4
st_slope_enc = {"Up": 1, "Flat": 2, "Down": 3}[st_slope]

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_enc],
        'ChestPainType': [chest_pain_enc],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs_enc],
        'RestingECG': [resting_ecg_enc],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina_enc],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_enc]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"  Higher risk of heart disease ({probability:.1%} confidence)")
    else:
        st.success(f"Low er risk of heart disease ({(1 - probability):.1%} confidence)")
