import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
#importing the heart_failure_model
import h5py

with h5py.File("heart_failure_model.h5", "r") as f:
    print(list(f.keys()))

model = load_model("heart_failure_model.h5")
S_Scaler = joblib.load("scaler.pkl")

st.title("Heart Failure Prediction")
st.write("This is a simple web app to predict heart failure using machine learning.")
st.sidebar.header("User Input")
st.sidebar.markdown("""
Please enter the following information:
""")
# collecting data  age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time
age = st.sidebar.slider("Age", 20, 100)
anaemia = st.sidebar.selectbox("Anaemia", [0, 1])
creatinine_phosphokinase = int(st.sidebar.text_input("Creatinine Phosphokinase", 0, 1000))
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
ejection_fraction = st.sidebar.slider("Ejection Fraction", 0, 100)
high_blood_pressure = st.sidebar.selectbox("High Blood Pressure", [0, 1])
platelets = int(st.sidebar.text_input("Platelets"))
serum_creatinine = float(st.sidebar.text_input("Serum Creatinine"))
serum_sodium = float(st.sidebar.text_input("Serum Sodium"))
sex = st.sidebar.selectbox("Sex", [0, 1])
smoking = st.sidebar.selectbox("Smoking", [0, 1])
time = st.sidebar.slider("Time", 0, 1000)

st.sidebar.button("Predict")
#integrate model and show prediction
st.write("Prediction: ")
input_data = [[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]]
# input_data = [72,0,127,1,50,1,218000,1,134,1,0,33]

input_array = np.array(input_data)
sample = input_array.reshape(1, -1)
sample_scaled = S_Scaler.transform(sample)
prediction = model.predict(sample_scaled)

# (Optional) reshape if model expects batch dimension
# If model was trained on shape (n_features,), add batch axis

st.write(1 if prediction>0.5 else 0)
st.sidebar.button("Clear")
