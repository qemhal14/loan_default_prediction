import streamlit as st
import numpy as np
import pandas as pd
import pickle
from xgboost.sklearn import XGBClassifier

with open('loan_default_predictor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict loan default based on input data
def predict_default(input_data, threshold=0.15):
    prob = model.predict_proba(input_data)[:, 1]  
    status = np.where(prob >= threshold, "Default", "Not Default")
    return prob, status

# Streamlit app layout
st.title("Loan Default Prediction App")

# Display choice for single or batch input in the main dashboard
input_choice = st.radio("Choose Input Method", ('Single Data Input', 'Batch Data Input'))

# Single Data Input Section
if input_choice == 'Single Data Input':
    st.header("Single Customer Data Input")

    person_age = st.number_input("Person Age", min_value=20, max_value=100, value=30)
    person_income = st.number_input("Person Income", min_value=0, value=50000)
    person_home_ownership = st.selectbox("Person Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    person_emp_length = st.number_input("Employment Length (in years)", min_value=0.0, value=5.0)
    loan_intent = st.selectbox("Loan Intent", ['EDUCATION', 'MEDICAL', 'PERSONAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
    loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
    loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, value=5.0)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, value=10.0)
    cb_person_default_on_file = st.selectbox("Customer Default on File", ['N', 'Y'])
    cb_person_cred_hist_length = st.number_input("Credit History Length (in years)", min_value=0, value=10)

    # Create input data as a DataFrame
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })

    # Predict button
    if st.button("Predict"):
        prob, status = predict_default(input_data)
        st.write(f"Probability of Default: {prob[0]:.2f}")
        st.write(f"Loan Status: {status[0]}")

# Batch Data Input Section
if input_choice == 'Batch Data Input':
    st.header("Batch Customer Data Input")
    file_upload = st.file_uploader("Upload CSV file", type=['csv'])

    if file_upload is not None:
        batch_data = pd.read_csv(file_upload)
        st.write("Uploaded Data", batch_data)

        if st.button("Predict for Batch"):
            prob, status = predict_default(batch_data)
            batch_data['Probability of Default'] = prob
            batch_data['Loan Status'] = status
            st.write("Prediction Results", batch_data)
