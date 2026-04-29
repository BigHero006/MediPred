# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 20:12:29 2025

@author: Dell
"""

import pickle
from pathlib import Path
import streamlit as st

# Configure page
st.set_page_config(
    page_title="MediPred - Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #0066cc;
        --success-color: #00cc66;
        --danger-color: #ff3333;
        --warning-color: #ffcc00;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0066cc 0%, #00ccff 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5em;
        margin: 0;
        font-weight: bold;
    }
    
    .main-header p {
        font-size: 1.1em;
        margin: 10px 0 0 0;
        opacity: 0.9;
    }
    
    /* Input section styling */
    .input-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin-bottom: 20px;
    }
    
    .input-section-title {
        font-size: 1.3em;
        font-weight: bold;
        color: #0066cc;
        margin-bottom: 15px;
    }
    
    /* Result cards */
    .result-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
    }
    
    .result-positive {
        background-color: #ffe6e6;
        border-left: 5px solid #ff3333;
        color: #cc0000;
    }
    
    .result-negative {
        background-color: #e6ffe6;
        border-left: 5px solid #00cc66;
        color: #009900;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        font-size: 1.1em;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 102, 204, 0.3);
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 20px;
        padding: 15px;
        background: linear-gradient(135deg, #0066cc 0%, #00ccff 100%);
        color: white;
        border-radius: 8px;
        text-align: center;
    }
    
    /* Info boxes */
    .info-box {
        background: #e6f2ff;
        border-left: 4px solid #0066cc;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    .info-box h4 {
        color: #0066cc;
        margin-top: 0;
    }
    
    /* Accuracy badges */
    .accuracy-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00cc66 0%, #00aa55 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Loading the saved models
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'

diabetes_model = pickle.load(open(MODELS_DIR / 'Diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(MODELS_DIR / 'Heart_model.sav', 'rb'))
parkinson_model = pickle.load(open(MODELS_DIR / 'Parkinson_model.sav', 'rb'))

# Sidebar navigation
with st.sidebar:
    st.markdown('<div class="sidebar-header">🏥 MediPred</div>', unsafe_allow_html=True)
    st.markdown("---")
    select = st.radio(
        "Select Prediction Model:",
        ['🩺 Diabetes Prediction', '❤️ Heart Disease Prediction', '🧠 Parkinsons Prediction'],
        label_visibility="visible"
    )
    st.markdown("---")
    st.markdown("""
    ### About MediPred
    MediPred uses machine learning models trained on medical datasets to provide disease predictions.
    - **Diabetes Model**: 72% Accuracy
    - **Heart Disease Model**: 80% Accuracy  
    - **Parkinson's Model**: 95% Accuracy
    
    ⚠️ **Disclaimer**: These predictions are for educational purposes only and should not replace professional medical advice.
    """)
    
# Extract base names for comparison
select_base = select.split()[-2:] if 'Diabetes' in select else select.split()[-2:]

# Diabetes prediction page
if 'Diabetes' in select:
    # Header
    st.markdown('<div class="main-header"><h1>🩺 Diabetes Prediction System</h1><p>Enter your health metrics below for diabetes risk assessment</p></div>', unsafe_allow_html=True)
    
    # Info section
    st.markdown("""
    <div class="info-box">
    <h4>📋 About This Assessment</h4>
    This model analyzes medical indicators to predict diabetes risk based on the provided health metrics.
    <span class="accuracy-badge">72% Accuracy</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="input-section"><div class="input-section-title">📊 Enter Your Health Information</div></div>', unsafe_allow_html=True)
    
    # Getting the input data from the user with organized columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Pregnancy & Glucose**")
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
        Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=120)
    
    with col2:
        st.markdown("**Blood Pressure & Skin**")
        BloodPressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=200, value=70)
        SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
    
    with col3:
        st.markdown("**Insulin & BMI**")
        Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=800, value=80)
        BMI = st.number_input('BMI (kg/m²)', min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    
    col4, col5 = st.columns(2)
    with col4:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, step=0.1)
    with col5:
        Age = st.number_input('Age (years)', min_value=0, max_value=120, value=30)

    col_button, col_empty = st.columns([1, 3])
    with col_button:
        predict_button = st.button('🔍 Predict Diabetes', key='diabetes_predict')

    if predict_button:
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
            probability = diabetes_model.predict_proba([input_data])
            
            if diabetes_predict[0] == 1:
                st.markdown(f"""
                <div class="result-card result-positive">
                ⚠️ Diabetes Detected<br>
                Risk Level: {probability[0][1]*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                st.warning("⚠️ **Prediction: You may have diabetes symptoms. Please consult a healthcare professional for proper diagnosis.**")
            else:
                st.markdown(f"""
                <div class="result-card result-negative">
                ✅ No Diabetes Detected<br>
                Risk Level: {probability[0][1]*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ **Prediction: Your health metrics suggest low diabetes risk. Maintain healthy habits!**")
        
        except ValueError as e:
            st.error("❌ Please enter valid numeric values for all fields.")


# Heart disease prediction page
if 'Heart' in select:
    st.markdown('<div class="main-header"><h1>❤️ Heart Disease Prediction System</h1><p>Enter your cardiovascular health metrics below</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📋 About This Assessment</h4>
    This model analyzes cardiovascular indicators to predict heart disease risk.
    <span class="accuracy-badge">80% Accuracy</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="input-section"><div class="input-section-title">📊 Enter Your Cardiovascular Information</div></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics & Vital Signs**")
        Age = st.number_input('Age (years)', min_value=0, max_value=120, value=50)
        Sex = st.selectbox('Sex', ['Female (0)', 'Male (1)'], key='heart_sex')
        Sex = int(Sex.split('(')[1].split(')')[0])
        Trestbps = st.number_input('Resting Blood Pressure (mmHg)', min_value=0, max_value=250, value=120)
    
    with col2:
        st.markdown("**Blood Work & Tests**")
        CP = st.selectbox('Chest Pain Type', 
                         ['Typical Angina (0)', 'Atypical Angina (1)', 'Non-anginal Pain (2)', 'Asymptomatic (3)'], key='heart_cp')
        CP = int(CP.split('(')[1].split(')')[0])
        Chol = st.number_input('Cholesterol (mg/dL)', min_value=0, max_value=600, value=200)
        Fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', ['No (0)', 'Yes (1)'], key='heart_fbs')
        Fbs = int(Fbs.split('(')[1].split(')')[0])
    
    with col3:
        st.markdown("**ECG & Stress Test**")
        Restecg = st.selectbox('Resting ECG Results',
                              ['Normal (0)', 'ST-T Abnormality (1)', 'LV Hypertrophy (2)'], key='heart_ecg')
        Restecg = int(Restecg.split('(')[1].split(')')[0])
        Thalach = st.number_input('Max Heart Rate Achieved (bpm)', min_value=0, max_value=250, value=150)
        Exang = st.selectbox('Exercise Induced Angina', ['No (0)', 'Yes (1)'], key='heart_exang')
        Exang = int(Exang.split('(')[1].split(')')[0])
    
    col4, col5 = st.columns(2)
    with col4:
        Oldpeak = st.number_input('ST Depression (Oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        Slope = st.selectbox('ST Slope', ['Upsloping (0)', 'Flat (1)', 'Downsloping (2)'], key='heart_slope')
        Slope = int(Slope.split('(')[1].split(')')[0])
    
    with col5:
        Ca = st.number_input('Major Vessels (0-3)', min_value=0, max_value=4, value=0)
        Thal = st.selectbox('Thalassemia', 
                           ['Normal (0)', 'Fixed Defect (1)', 'Reversible Defect (2)', 'Other (3)'], key='heart_thal')
        Thal = int(Thal.split('(')[1].split(')')[0])
    
    col_button, col_empty = st.columns([1, 3])
    with col_button:
        predict_button = st.button('🔍 Predict Heart Disease', key='heart_predict')

    if predict_button:
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
            
            heart_disease_predict = heart_disease_model.predict([input_data])
            probability = heart_disease_model.predict_proba([input_data])
            
            if heart_disease_predict[0] == 1:
                st.markdown(f"""
                <div class="result-card result-positive">
                ⚠️ Heart Disease Detected<br>
                Risk Level: {probability[0][1]*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                st.warning("⚠️ **Prediction: You may have heart disease. Please consult a cardiologist immediately.**")
            else:
                st.markdown(f"""
                <div class="result-card result-negative">
                ✅ No Heart Disease Detected<br>
                Risk Level: {probability[0][1]*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ **Prediction: Your cardiovascular health looks good. Keep up regular exercise and healthy diet!**")
        
        except ValueError:
            st.error("❌ Please enter valid values for all fields.")


# Parkinson's prediction page
if 'Parkinsons' in select:
    st.markdown('<div class="main-header"><h1>🧠 Parkinson\'s Disease Prediction System</h1><p>Enter your voice and motor metrics below</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📋 About This Assessment</h4>
    This model analyzes voice characteristics and motor control metrics to predict Parkinson's disease risk.
    <span class="accuracy-badge">95% Accuracy</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="input-section"><div class="input-section-title">📊 Enter Your Vocal & Motor Metrics</div></div>', unsafe_allow_html=True)
    
    with st.expander("📖 Feature Guide", expanded=False):
        st.markdown("""
        - **MDVP Metrics**: Voice fundamental frequency and jitter measurements
        - **Shimmer**: Voice intensity and variation
        - **NHR/HNR**: Noise-to-Harmonic ratio indicators
        - **RPDE/DFA**: Non-linear dynamical complexity measures
        - **PPE**: Pitch Period Entropy for voice analysis
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**MDVP Frequency**")
        MDVP_Fo = st.number_input('MDVP Fo(Hz)', min_value=0.0, max_value=500.0, value=150.0, step=0.1)
        MDVP_Fhi = st.number_input('MDVP Fhi(Hz)', min_value=0.0, max_value=500.0, value=200.0, step=0.1)
        MDVP_Flo = st.number_input('MDVP Flo(Hz)', min_value=0.0, max_value=500.0, value=100.0, step=0.1)
        
        st.markdown("**MDVP Jitter**")
        MDVP_Jitter_Percent = st.number_input('MDVP Jitter (%)', min_value=0.0, max_value=1.0, value=0.005, step=0.0001, format="%.4f")
        MDVP_Jitter_Abs = st.number_input('MDVP Jitter (Abs)', min_value=0.0, max_value=0.1, value=0.00005, step=0.00001, format="%.5f")
    
    with col2:
        st.markdown("**MDVP Shimmer**")
        MDVP_Shimmer = st.number_input('MDVP Shimmer', min_value=0.0, max_value=1.0, value=0.03, step=0.001, format="%.3f")
        MDVP_Shimmer_dB = st.number_input('MDVP Shimmer(dB)', min_value=0.0, max_value=5.0, value=0.3, step=0.01, format="%.2f")
        Shimmer_APQ3 = st.number_input('Shimmer APQ3', min_value=0.0, max_value=0.1, value=0.015, step=0.001, format="%.3f")
        Shimmer_APQ5 = st.number_input('Shimmer APQ5', min_value=0.0, max_value=0.1, value=0.02, step=0.001, format="%.3f")
        
        st.markdown("**Shimmer Advanced**")
        MDVP_APQ = st.number_input('MDVP APQ', min_value=0.0, max_value=0.1, value=0.02, step=0.001, format="%.3f")
        Shimmer_DDA = st.number_input('Shimmer DDA', min_value=0.0, max_value=0.1, value=0.03, step=0.001, format="%.3f")
    
    with col3:
        st.markdown("**Ratios & Entropy**")
        NHR = st.number_input('NHR', min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.2f")
        HNR = st.number_input('HNR', min_value=0.0, max_value=40.0, value=25.0, step=0.5)
        MDVP_RAP = st.number_input('MDVP RAP', min_value=0.0, max_value=0.1, value=0.003, step=0.0001, format="%.4f")
        MDVP_PPQ = st.number_input('MDVP PPQ', min_value=0.0, max_value=0.1, value=0.004, step=0.0001, format="%.4f")
        Jitter_DDP = st.number_input('Jitter DDP', min_value=0.0, max_value=0.1, value=0.006, step=0.0001, format="%.4f")
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("**Non-linear Measures**")
        RPDE = st.number_input('RPDE', min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f")
        DFA = st.number_input('DFA', min_value=0.0, max_value=1.0, value=0.7, step=0.01, format="%.2f")
        Spread1 = st.number_input('Spread1', min_value=-10.0, max_value=10.0, value=-5.0, step=0.1, format="%.1f")
    
    with col5:
        st.markdown("**Advanced Metrics**")
        Spread2 = st.number_input('Spread2', min_value=0.0, max_value=1.0, value=0.2, step=0.01, format="%.2f")
        D2 = st.number_input('D2', min_value=0.0, max_value=5.0, value=2.5, step=0.1, format="%.1f")
        PPE = st.number_input('PPE', min_value=0.0, max_value=1.0, value=0.2, step=0.01, format="%.2f")
    
    col_button, col_empty = st.columns([1, 3])
    with col_button:
        predict_button = st.button('🔍 Predict Parkinsons', key='parkinsons_predict')

    if predict_button:
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
            
            parkinson_predict = parkinson_model.predict([input_data])
            probability = parkinson_model.predict_proba([input_data])
            
            if parkinson_predict[0] == 1:
                st.markdown(f"""
                <div class="result-card result-positive">
                ⚠️ Parkinson's Disease Detected<br>
                Risk Level: {probability[0][1]*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                st.warning("⚠️ **Prediction: Parkinson's disease symptoms detected. Please consult a neurologist for proper diagnosis.**")
            else:
                st.markdown(f"""
                <div class="result-card result-negative">
                ✅ No Parkinson's Disease Detected<br>
                Risk Level: {probability[0][1]*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ **Prediction: Your vocal metrics suggest no Parkinson's disease. Keep monitoring your health!**")
        
        except ValueError as e:
            st.error("❌ Please enter valid numeric values for all fields.")
