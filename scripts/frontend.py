import streamlit as st
from .backend import Backend

def run_frontend():
    # Initialize backend
    backend = Backend()

    st.title('Medical Symptom Diagnosis App')
    st.write('Enter the patient symptoms below to get the predicted diagnosis and treatment plan.')

    # Input
    symptom_input = st.text_area('Symptoms', placeholder='Enter symptoms separated by commas')

    if st.button('Predict'):
        if symptom_input.strip() == '':
            st.warning('Please enter symptoms to predict.')
        else:
            diagnosis, treatment = backend.predict(symptom_input)
            st.success(f'**Predicted Diagnosis:** {diagnosis}')
            st.info(f'**Recommended Treatment Plan:** {treatment}')
