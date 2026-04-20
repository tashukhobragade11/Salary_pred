
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file 'linear_regression_model.pkl' not found. Please ensure it's in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input features (based on your X DataFrame columns)
# For simplicity, these are currently numerical inputs. 
# If you have the original mappings for categorical features, you'd integrate them here.

rating = st.slider('Rating', min_value=0.0, max_value=5.0, value=3.5, step=0.1)
company_name_encoded = st.number_input('Company Name (Encoded)', min_value=0, value=0)
job_title_encoded = st.number_input('Job Title (Encoded)', min_value=0, value=0)
salaries_reported = st.number_input('Salaries Reported', min_value=0, value=1)
location_encoded = st.number_input('Location (Encoded)', min_value=0, value=0)
employment_status_encoded = st.number_input('Employment Status (Encoded)', min_value=0, value=0)
job_roles_encoded = st.number_input('Job Roles (Encoded)', min_value=0, value=0)

# Create a DataFrame for prediction
input_data = pd.DataFrame([[rating, company_name_encoded, job_title_encoded, salaries_reported, location_encoded, employment_status_encoded, job_roles_encoded]],
                          columns=['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'])

if st.button('Predict Salary'):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f'Predicted Salary: ₹{prediction:,.2f}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
