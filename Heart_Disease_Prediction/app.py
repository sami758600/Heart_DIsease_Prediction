import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("heart_disease_model.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the patient's information below to get a prediction "
    "from the trained Logistic Regression model."
)


# --------------------------------------------------
# Patient Information
# --------------------------------------------------

st.header("Patient Information")


age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=55
)


sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
)


cp = st.selectbox(
    "Chest Pain Type (cp)",
    options=[0, 1, 2, 3],
    help="0 = Typical angina, 1 = Atypical angina, "
         "2 = Non-anginal pain, 3 = Asymptomatic"
)


trestbps = st.number_input(
    "Resting Blood Pressure (trestbps)",
    min_value=50,
    max_value=250,
    value=130
)


chol = st.number_input(
    "Serum Cholesterol (chol)",
    min_value=50,
    max_value=700,
    value=240
)


fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl (fbs)",
    options=[0, 1],
    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
)


restecg = st.selectbox(
    "Resting ECG (restecg)",
    options=[0, 1, 2],
    help="0 = Normal, 1 = ST-T wave abnormality, "
         "2 = Left ventricular hypertrophy"
)


thalach = st.number_input(
    "Maximum Heart Rate (thalach)",
    min_value=50,
    max_value=250,
    value=150
)


exang = st.selectbox(
    "Exercise Induced Angina (exang)",
    options=[0, 1],
    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
)


oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)


slope = st.selectbox(
    "Slope",
    options=[0, 1, 2]
)


ca = st.selectbox(
    "Number of Major Vessels (ca)",
    options=[0, 1, 2, 3, 4]
)


thal = st.selectbox(
    "Thal",
    options=[0, 1, 2, 3]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict", use_container_width=True):

    # Create DataFrame in the same feature order
    # used during model training

    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])


    # Prediction
    prediction = model.predict(patient)[0]


    # Probability
    probability = model.predict_proba(patient)[0]


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.header("Prediction Result")


    if prediction == 1:

        st.error(
            "Model Prediction: Heart Disease (Class 1)"
        )

    else:

        st.success(
            "Model Prediction: No Heart Disease (Class 0)"
        )


    # Probability display

    st.subheader("Model Probabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Class 0",
            f"{probability[0] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Class 1",
            f"{probability[1] * 100:.2f}%"
        )


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.divider()

st.caption(
    "⚠️ This application is an educational machine-learning project. "
    "The prediction is not a medical diagnosis and should not be used "
    "as a substitute for professional medical advice."
)

