# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 20:12:29 2025

@author: Dell
"""

import pickle
from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu

# Loading the saved models
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'

diabetes_model = pickle.load(open(MODELS_DIR / 'Diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(MODELS_DIR / 'Heart_model.sav', 'rb'))
parkinson_model = pickle.load(open(MODELS_DIR / 'Parkinson_model.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    select = option_menu('Disease Prediction System using ML',
                         ['Diabetes Prediction',
                          'Heart Disease Prediction',
                          'Parkinsons Prediction'],
                         icons=['bandaid', 'heart-pulse', 'person-wheelchair'],
                         default_index=0)

# Diabetes prediction page
if select == 'Diabetes Prediction':
    st.title("Diabetes Prediction Using ML")
    
    # Getting the input data from the user
    # Columns for the input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', '0')
        Glucose = st.text_input('Glucose Level', '0')
    
    with col2:
        BloodPressure = st.text_input('Blood Pressure Level', '0')
        SkinThickness = st.text_input('Thickness of skin', '0')
    
    with col3:
        Insulin = st.text_input('Insulin Level', '0')
        BMI = st.text_input('BMI value', '0.0')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Level', '0.0')
        Age = st.text_input('Age of the patient', '0')

    diabetes_diagnosis = ''

    if st.button('Predict Diabetes'):
        # Convert input data to the appropriate types with validation
        try:
            input_data = [
                int(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                int(Age)
            ]
            
            
            diabetes_predict = diabetes_model.predict([input_data])
            
            if diabetes_predict[0] == 1:
                diabetes_diagnosis = "The person is Diabetic"
            else:
                diabetes_diagnosis = "The person is not Diabetic"
            
            st.success(diabetes_diagnosis)
        
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

# Heart disease prediction page
# Heart disease prediction page
if select == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction Using ML")
    
    # Getting the input data from the user
    # Columns for the input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age = st.text_input('Age of the patient', '0')
        Sex = st.selectbox('Sex', ['0', '1'])  # 0 = Female, 1 = Male
        CP = st.selectbox('Chest Pain Type (cp)', ['0', '1', '2', '3'])  # 0-3 for different types of chest pain
        Trestbps = st.text_input('Resting Blood Pressure (trestbps)', '0')
        Chol = st.text_input('Cholesterol Level (chol)', '0')
    
    with col2:
       Fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ['0', '1'])  # 0 = False, 1 = True
       Restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', ['0', '1', '2'])  # 0-2 for different results
       Thalach = st.text_input('Maximum Heart Rate Achieved (thalach)', '0')
       Exang = st.selectbox('Exercise Induced Angina (exang)', ['0', '1'])  # 0 = No, 1 = Yes
    
    with col3:
        
        Oldpeak = st.text_input('Oldpeak', '0.0')
        Slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', ['0', '1', '2'])  # 0-2 for different slopes
        Ca = st.text_input('Number of Major Vessels (ca)', '0')
        Thal = st.selectbox('Thalassemia (thal)', ['0', '1', '2', '3'])  # 0-3 for different thalassemia types

    heart_disease_diagnosis = ''

    if st.button('Predict Heart Disease'):
        # Convert input data to the appropriate types with validation
        try:
            input_data = [
                int(Age),
                int(Sex),
                int(CP),
                float(Trestbps),
                float(Chol),
                int(Fbs),
                int(Restecg),
                float(Thalach),
                int(Exang),
                float(Oldpeak),
                int(Slope),
                int(Ca),
                int(Thal)
            ]
            
            # Debugging output
            st.write("Input Data:", input_data)
            
            heart_disease_predict = heart_disease_model.predict([input_data])
            
            if heart_disease_predict[0] == 1:
                heart_disease_diagnosis = "The person has Heart Disease"
            else:
                heart_disease_diagnosis = "The person does not have Heart Disease"
            
            st.success(heart_disease_diagnosis)
        
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")


# Parkinson's prediction page
if select == 'Parkinsons Prediction':
    st.title("Parkinson Prediction Using ML")

    # Getting the input data from the user
    # Columns for the input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        MDVP_Fo = st.text_input('MDVP:Fo(Hz)', '0.0')
        MDVP_Fhi = st.text_input('MDVP:Fhi(Hz)', '0.0')
        MDVP_Flo = st.text_input('MDVP:Flo(Hz)', '0.0')
        MDVP_Jitter_Percent = st.text_input('MDVP:Jitter(%)', '0.0')
    
    with col2:
       MDVP_Shimmer = st.text_input('MDVP:Shimmer', '0.0')
       MDVP_Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', '0.0')
       Shimmer_APQ3 = st.text_input('Shimmer:APQ3', '0.0')
       Shimmer_APQ5 = st.text_input('Shimmer:APQ5', '0.0')
       
    with col3:
        MDVP_APQ = st.text_input('MDVP:APQ', '0.0')
        Shimmer_DDA = st.text_input('Shimmer:DDA', '0.0')
        NHR = st.text_input('NHR', '0.0')
        HNR = st.text_input('HNR', '0.0')
  
       
    
    # Additional inputs
    col4, col5 = st.columns(2)
    
    with col4:
        MDVP_Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', '0.0')
        MDVP_RAP = st.text_input('MDVP:RAP', '0.0')
        MDVP_PPQ = st.text_input('MDVP:PPQ', '0.0')
        Jitter_DDP = st.text_input('Jitter:DDP', '0.0')
        
    with col5:
        RPDE = st.text_input('RPDE', '0.0')
        DFA = st.text_input('DFA', '0.0')
        Spread1 = st.text_input('spread1', '0.0')
        Spread2 = st.text_input('spread2', '0.0')
        D2 = st.text_input('D2', '0.0')
        PPE = st.text_input('PPE', '0.0')

    parkinson_diagnosis = ''

    if st.button('Predict Parkinsons'):
        # Convert input data to the appropriate types with validation
        try:
            input_data = [
                float(MDVP_Fo),
                float(MDVP_Fhi),
                float(MDVP_Flo),
                float(MDVP_Jitter_Percent),
                float(MDVP_Jitter_Abs),
                float(MDVP_RAP),
                float(MDVP_PPQ),
                float(Jitter_DDP),
                float(MDVP_Shimmer),
                float(MDVP_Shimmer_dB),
                float(Shimmer_APQ3),
                float(Shimmer_APQ5),
                float(MDVP_APQ),
                float(Shimmer_DDA),
                float(NHR),
                float(HNR),
                float(RPDE),
                float(DFA),
                float(Spread1),
                float(Spread2),
                float(D2),
                float(PPE)
            ]
            
        
            st.write("Input Data:", input_data)
            
            parkinson_predict = parkinson_model.predict([input_data])
            
            if parkinson_predict[0] == 1:
                parkinson_diagnosis = "The person has Parkinson's Disease"
            else:
                parkinson_diagnosis = "The person does not have Parkinson's Disease"
            
            st.success(parkinson_diagnosis)
        
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
