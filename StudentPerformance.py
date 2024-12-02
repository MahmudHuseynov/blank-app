import streamlit as st
import pandas as pd
import joblib

# Load saved scaler and Linear Regression model
scaler = joblib.load('scaler1.pkl')
lr_model = joblib.load('lr_model1.pkl')

# Function to map grades to rubrics
def grade_to_rubric(grade):
    if grade < 4:
        return "Undefined"
    elif grade < 8:
        return "Orienting"
    elif grade < 12:
        return "Beginning"
    elif grade < 16:
        return "Proficient"
    else:
        return "Advanced"

# Streamlit application title
st.title("Student Grade Predictor")

st.sidebar.markdown("### Input Student Data")
# Input fields
G1 = st.sidebar.number_input("G1 (Grade for period 1)", min_value=0, max_value=20, value=10)
G2 = st.sidebar.number_input("G2 (Grade for period 2)", min_value=0, max_value=20, value=10)
studytime = st.sidebar.number_input("Study Time (hours)", min_value=0, max_value=4, value=2)
failures = st.sidebar.number_input("Failures (count)", min_value=0, max_value=4, value=0)
absences = st.sidebar.number_input("Absences (count)", min_value=0, max_value=93, value=5)
dalc = st.sidebar.slider("Workday Alcohol Consumption (1-5)", min_value=1, max_value=5, value=1)
walc = st.sidebar.slider("Weekend Alcohol Consumption (1-5)", min_value=1, max_value=5, value=1)

# Create DataFrame for input
input_data = pd.DataFrame([{
    'G1': G1, 
    'G2': G2, 
    'studytime': studytime, 
    'failures': failures, 
    'absences': absences, 
    'Dalc': dalc, 
    'Walc': walc
}])

# Scale input data
scaled_data = scaler.transform(input_data)

# Make prediction using Linear Regression
prediction = lr_model.predict(scaled_data)

# Convert prediction to rubric category
predicted_grade = prediction[0]
rubric_category = grade_to_rubric(predicted_grade)

# Display prediction and rubric category
st.subheader("Prediction using Linear Regression")
st.write(f"Predicted Final Grade (G3): {predicted_grade:.2f}")
st.write(f"Grade Category: **{rubric_category}**")