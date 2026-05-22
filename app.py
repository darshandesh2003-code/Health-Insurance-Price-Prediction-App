import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================================
# Load Trained Model
# =========================================

model = joblib.load("Gradient Boosting.pkl")

# =========================================
# App Title
# =========================================

st.title("Insurance Price Prediction")

st.write(
    "Enter the details below to predict insurance charges."
)

# =========================================
# User Inputs
# =========================================

age = st.number_input(
    "Enter Age",
    min_value=18,
    max_value=100,
    value=25
)

sex = st.selectbox(
    "Select Gender",
    ["male", "female"]
)

bmi = st.number_input(
    "Enter BMI",
    min_value=10.0,
    max_value=53.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=5,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# =========================================
# Prediction Button
# =========================================

if st.button("Predict", key="predict_btn"):

    try:

        # =========================================
        # Encoding
        # =========================================

        smoker_value = 1 if smoker == "yes" else 0

        sex_female = 1 if sex == "female" else 0
        sex_male = 1 if sex == "male" else 0

        region_value = {
            'southwest': 0,
            'northwest': 1,
            'northeast': 2,
            'southeast': 3
        }[region]

        # =========================================
        # Input DataFrame
        # =========================================

        input_data = pd.DataFrame({

            'age': [int(age)],

            'bmi': [float(bmi)],

            'children': [int(children)],

            'Smoker': [int(smoker_value)],

            'sex_female': [int(sex_female)],

            'sex_male': [int(sex_male)],

            'Region': [int(region_value)]

        })

        # =========================================
        # Prediction
        # =========================================

        prediction = model.predict(input_data)

        # Reverse Log Transformation
        final_price = np.exp(prediction[0])

        # Slight increase for 5 children
        if children == 5:
            final_price = final_price * 1.08

        # Keep realistic range
        final_price = np.clip(
            final_price,
            1000,
            70000
        )

        # =========================================
        # Display Result
        # =========================================

        st.success(
            f"Predicted Insurance Price: ₹ {final_price:,.2f}"
        )

        

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )
