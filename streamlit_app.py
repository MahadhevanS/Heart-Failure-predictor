import streamlit as st
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

def cardio_vascular():
    # Inject custom CSS for styling
    st.set_page_config(layout="wide")
    st.markdown("""
    <style>
        .reportview-container .main .block-container{
            padding-top: 0rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 0rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>div>input {
            border-radius: 4px;
            border: 1px solid #ccc;
            padding: 8px;
        }
        h1, h2, h3 {
            color: #2c3e50;
            text-align: center;
        }
        .prediction-box {
            background-color: #2ff500;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            text-align: center;
        }
        .error-box {
            background-color: #ff0000;
            border: 2px solid #ef5350;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    # Correctly handle file paths
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "heart_failure_model.h5")
    scaler_path = os.path.join(current_dir, "scaler.pkl")

    # Load the model and scaler only once
    try:
        model = load_model(model_path)
        S_Scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

    # Use columns to center the input form
    center_col_padding, main_col, center_col_padding2 = st.columns([1,500,1])

    with main_col:
        st.markdown("<h2 style='text-align: left;'>Please provide the following information:</h2>", unsafe_allow_html=True)

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            age = st.slider("Age", 20, 100, 50)
            anaemia = 0 if st.selectbox("Anaemia", ["Yes", "No"])=="No" else 1

        with col2:
            platelets = int(st.text_input("Platelets", "263358"))
            serum_creatinine = float(st.text_input("Serum Creatinine", "1.1"))
        with col3:
            diabetes = 0 if st.selectbox("Diabetes", ["Yes", "No"])=="No" else 1
            ejection_fraction = st.slider("Ejection Fraction", 0, 100, 35)
        with col4:
            sex = 0 if st.selectbox("Sex", ["Male", "Female"])=="Female" else 1
            smoking = 0 if st.selectbox("Smoking", ["No", "Yes"])=="No" else 1
        with col5:
            creatinine_phosphokinase = int(st.text_input("Creatinine Phosphokinase", "0"))
            serum_sodium = float(st.text_input("Serum Sodium", "136.0"))
        with col6:
            high_blood_pressure = 0 if st.selectbox("High Blood Pressure", ["No", "Yes"])=="No" else 1
            time = st.slider("Time (days)", 4, 285, 130)
        st.markdown("---")
        # Place the predict button in the centered column
        if st.button("Predict",use_container_width=True):
            st.subheader("Prediction Result:")
            
            input_data = np.array([[
                age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                sex, smoking, time
            ]])

            sample_scaled = S_Scaler.transform(input_data)
            prediction_prob = model.predict(sample_scaled)
            
            prediction = 1 if prediction_prob[0][0] > 0.4 else 0
            
            if prediction == 1:
                st.markdown(
                    f"<div class='error-box'><h2>High Risk of Heart Failure</h2>"
                    f"<h4>Confidence: {prediction_prob[0][0]:.2f}</h4></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='prediction-box'><h2>Low Risk of Heart Failure</h2>"
                    f"<h4>Confidence: {prediction_prob[0][0]:.2f}</h4></div>",
                    unsafe_allow_html=True
                )

if __name__=="__main__":
    cardio_vascular()
