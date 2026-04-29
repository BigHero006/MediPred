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
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more restrained, clinical layout
st.markdown("""
    <style>
    :root {
        --primary-color: #204a87;
        --primary-hover: #173a6b;
        --surface: #ffffff;
        --surface-muted: #f6f8fb;
        --border: #dbe3ee;
        --text-main: #1f2937;
        --text-muted: #5b6472;
        --success-bg: #eef8f1;
        --success-border: #4f9d69;
        --warning-bg: #fff6e8;
        --warning-border: #d9822b;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-main);
    }

    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
    }

    section.main > div {
        padding-top: 1.25rem;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .page-shell {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(219, 227, 238, 0.8);
        border-radius: 20px;
        padding: 1.75rem 1.5rem;
        box-shadow: 0 18px 40px rgba(31, 41, 55, 0.06);
    }

    .page-header {
        background: linear-gradient(180deg, #ffffff 0%, #f9fbfd 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.25rem;
    }

    .page-header .eyebrow {
        margin: 0 0 0.35rem 0;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        font-weight: 700;
    }

    .page-header h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.15;
        color: var(--text-main);
    }

    .page-header p {
        margin: 0.45rem 0 0 0;
        color: var(--text-muted);
        font-size: 1rem;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-bottom: 1rem;
    }

    .info-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1rem 0.9rem 1rem;
    }

    .info-card .label {
        font-size: 0.78rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }

    .info-card .value {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-main);
    }

    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-main);
        margin: 1.15rem 0 0.8rem 0;
    }

    .card-surface {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1rem 0.25rem 1rem;
        margin-bottom: 1rem;
    }

    .result-card {
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-top: 1rem;
        border: 1px solid var(--border);
        background: var(--surface);
    }

    .result-card .status {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
        color: var(--text-muted);
    }

    .result-card .headline {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.2rem;
    }

    .result-card .detail {
        color: var(--text-muted);
    }

    .result-positive {
        background: var(--success-bg);
        border-color: var(--success-border);
    }

    .result-negative {
        background: #f4f8fb;
        border-color: var(--border);
    }

    .stButton > button {
        background: var(--primary-color);
        color: white;
        border: none;
        padding: 0.85rem 1.2rem;
        font-size: 0.98rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 8px 18px rgba(32, 74, 135, 0.16);
    }
    
    .stButton > button:hover {
        background: var(--primary-hover);
        box-shadow: 0 10px 22px rgba(32, 74, 135, 0.22);
    }

    .sidebar-header {
        font-size: 1.25em;
        font-weight: bold;
        margin-bottom: 14px;
        padding: 14px 16px;
        background: #203b57;
        color: white;
        border-radius: 12px;
        text-align: center;
    }

    .sidebar-note {
        background: var(--surface-muted);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem 0.95rem;
        color: var(--text-muted);
        font-size: 0.93rem;
        line-height: 1.5;
    }

    .hint-text {
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-top: -0.15rem;
        margin-bottom: 0.8rem;
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
    st.markdown('<div class="sidebar-header">MediPred</div>', unsafe_allow_html=True)
    st.markdown("---")
    select = st.radio(
        "Select Prediction Model:",
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        label_visibility="visible"
    )
    st.markdown("---")



def render_page_header(title: str, subtitle: str, eyebrow: str) -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(items) -> None:
    cards = ''.join(
        f'<div class="info-card"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="info-grid">{cards}</div>', unsafe_allow_html=True)


def render_result(is_positive: bool, label: str, probability: float) -> None:
    class_name = 'result-positive' if is_positive else 'result-negative'
    headline = 'Higher risk detected' if is_positive else 'Lower risk detected'
    detail_label = 'Risk estimate' if is_positive else 'Confidence'
    st.markdown(
        f"""
        <div class="result-card {class_name}">
            <div class="status">{label}</div>
            <div class="headline">{headline}</div>
            <div class="detail">{detail_label}: {probability:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Diabetes prediction page
if select == 'Diabetes Prediction':
    render_page_header(
        'Diabetes Prediction',
        'Enter a few measurements to estimate diabetes risk from the trained model.',
        'Risk assessment'
    )

    render_metric_cards([
        ('Model accuracy', '72%'),
        ('Input features', '8 clinical values'),
        ('Output', 'Binary risk estimate'),
    ])

    st.markdown('<div class="section-title">Patient measurements</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
            Glucose = st.number_input('Glucose (mg/dL)', min_value=0, max_value=300, value=120)

        with col2:
            BloodPressure = st.number_input('Blood pressure (mmHg)', min_value=0, max_value=200, value=70)
            SkinThickness = st.number_input('Skin thickness (mm)', min_value=0, max_value=100, value=20)

        with col3:
            Insulin = st.number_input('Insulin', min_value=0, max_value=800, value=80)
            BMI = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, step=0.1)

        col4, col5 = st.columns(2)
        with col4:
            DiabetesPedigreeFunction = st.number_input('Diabetes pedigree function', min_value=0.0, max_value=2.5, value=0.5, step=0.1)
        with col5:
            Age = st.number_input('Age', min_value=0, max_value=120, value=30)

    col_button, col_empty = st.columns([1, 3])
    with col_button:
        predict_button = st.button('Predict diabetes', key='diabetes_predict')

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
                render_result(True, 'Diabetes prediction', probability[0][1] * 100)
                st.warning('The model indicates elevated diabetes risk. Please review the result with a clinician.')
            else:
                render_result(False, 'Diabetes prediction', probability[0][1] * 100)
                st.success('The model indicates low diabetes risk based on the provided values.')

        except ValueError:
            st.error('Please enter valid numeric values for all fields.')


# Heart disease prediction page
if select == 'Heart Disease Prediction':
    render_page_header(
        'Heart Disease Prediction',
        'Use cardiovascular measurements to estimate disease risk.',
        'Cardiac assessment'
    )

    render_metric_cards([
        ('Model accuracy', '80%'),
        ('Input features', '13 clinical values'),
        ('Output', 'Binary risk estimate'),
    ])

    st.markdown('<div class="section-title">Patient measurements</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.number_input('Age', min_value=0, max_value=120, value=50)
            Sex = st.selectbox('Sex', ['Female (0)', 'Male (1)'], key='heart_sex')
            Sex = int(Sex.split('(')[1].split(')')[0])
            Trestbps = st.number_input('Resting blood pressure (mmHg)', min_value=0, max_value=250, value=120)

        with col2:
            CP = st.selectbox('Chest pain type', ['Typical angina (0)', 'Atypical angina (1)', 'Non-anginal pain (2)', 'Asymptomatic (3)'], key='heart_cp')
            CP = int(CP.split('(')[1].split(')')[0])
            Chol = st.number_input('Cholesterol (mg/dL)', min_value=0, max_value=600, value=200)
            Fbs = st.selectbox('Fasting blood sugar > 120 mg/dL', ['No (0)', 'Yes (1)'], key='heart_fbs')
            Fbs = int(Fbs.split('(')[1].split(')')[0])

        with col3:
            Restecg = st.selectbox('Resting ECG results', ['Normal (0)', 'ST-T abnormality (1)', 'LV hypertrophy (2)'], key='heart_ecg')
            Restecg = int(Restecg.split('(')[1].split(')')[0])
            Thalach = st.number_input('Max heart rate achieved (bpm)', min_value=0, max_value=250, value=150)
            Exang = st.selectbox('Exercise induced angina', ['No (0)', 'Yes (1)'], key='heart_exang')
            Exang = int(Exang.split('(')[1].split(')')[0])

        col4, col5 = st.columns(2)
        with col4:
            Oldpeak = st.number_input('ST depression (Oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            Slope = st.selectbox('ST slope', ['Upsloping (0)', 'Flat (1)', 'Downsloping (2)'], key='heart_slope')
            Slope = int(Slope.split('(')[1].split(')')[0])

        with col5:
            Ca = st.number_input('Major vessels (0-3)', min_value=0, max_value=4, value=0)
            Thal = st.selectbox('Thalassemia', ['Normal (0)', 'Fixed defect (1)', 'Reversible defect (2)', 'Other (3)'], key='heart_thal')
            Thal = int(Thal.split('(')[1].split(')')[0])

    col_button, col_empty = st.columns([1, 3])
    with col_button:
        predict_button = st.button('Predict heart disease', key='heart_predict')

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
                render_result(True, 'Heart disease prediction', probability[0][1] * 100)
                st.warning('The model indicates elevated heart disease risk. Please consult a cardiologist.')
            else:
                render_result(False, 'Heart disease prediction', probability[0][1] * 100)
                st.success('The model indicates lower heart disease risk based on the provided values.')

        except ValueError:
            st.error('Please enter valid values for all fields.')


# Parkinson's prediction page
if select == 'Parkinsons Prediction':
    render_page_header(
        "Parkinson's Prediction",
        'Use speech and vocal measurements to estimate Parkinsons risk.',
        'Neurological assessment'
    )

    render_metric_cards([
        ('Model accuracy', '95%'),
        ('Input features', '22 vocal metrics'),
        ('Output', 'Binary risk estimate'),
    ])

    st.markdown('<div class="section-title">Voice measurements</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            MDVP_Fo = st.number_input('MDVP Fo (Hz)', min_value=0.0, max_value=500.0, value=150.0, step=0.1)
            MDVP_Fhi = st.number_input('MDVP Fhi (Hz)', min_value=0.0, max_value=500.0, value=200.0, step=0.1)
            MDVP_Flo = st.number_input('MDVP Flo (Hz)', min_value=0.0, max_value=500.0, value=100.0, step=0.1)
            MDVP_Jitter_Percent = st.number_input('MDVP jitter (%)', min_value=0.0, max_value=1.0, value=0.005, step=0.0001, format='%.4f')
            MDVP_Jitter_Abs = st.number_input('MDVP jitter (abs)', min_value=0.0, max_value=0.1, value=0.00005, step=0.00001, format='%.5f')

        with col2:
            MDVP_Shimmer = st.number_input('MDVP shimmer', min_value=0.0, max_value=1.0, value=0.03, step=0.001, format='%.3f')
            MDVP_Shimmer_dB = st.number_input('MDVP shimmer (dB)', min_value=0.0, max_value=5.0, value=0.3, step=0.01, format='%.2f')
            Shimmer_APQ3 = st.number_input('Shimmer APQ3', min_value=0.0, max_value=0.1, value=0.015, step=0.001, format='%.3f')
            Shimmer_APQ5 = st.number_input('Shimmer APQ5', min_value=0.0, max_value=0.1, value=0.02, step=0.001, format='%.3f')
            MDVP_APQ = st.number_input('MDVP APQ', min_value=0.0, max_value=0.1, value=0.02, step=0.001, format='%.3f')
            Shimmer_DDA = st.number_input('Shimmer DDA', min_value=0.0, max_value=0.1, value=0.03, step=0.001, format='%.3f')

        with col3:
            NHR = st.number_input('NHR', min_value=0.0, max_value=1.0, value=0.02, step=0.01, format='%.2f')
            HNR = st.number_input('HNR', min_value=0.0, max_value=40.0, value=25.0, step=0.5)
            MDVP_RAP = st.number_input('MDVP RAP', min_value=0.0, max_value=0.1, value=0.003, step=0.0001, format='%.4f')
            MDVP_PPQ = st.number_input('MDVP PPQ', min_value=0.0, max_value=0.1, value=0.004, step=0.0001, format='%.4f')
            Jitter_DDP = st.number_input('Jitter DDP', min_value=0.0, max_value=0.1, value=0.006, step=0.0001, format='%.4f')

        col4, col5 = st.columns(2)

        with col4:
            RPDE = st.number_input('RPDE', min_value=0.0, max_value=1.0, value=0.5, step=0.01, format='%.2f')
            DFA = st.number_input('DFA', min_value=0.0, max_value=1.0, value=0.7, step=0.01, format='%.2f')
            Spread1 = st.number_input('Spread1', min_value=-10.0, max_value=10.0, value=-5.0, step=0.1, format='%.1f')

        with col5:
            Spread2 = st.number_input('Spread2', min_value=0.0, max_value=1.0, value=0.2, step=0.01, format='%.2f')
            D2 = st.number_input('D2', min_value=0.0, max_value=5.0, value=2.5, step=0.1, format='%.1f')
            PPE = st.number_input('PPE', min_value=0.0, max_value=1.0, value=0.2, step=0.01, format='%.2f')

    col_button, col_empty = st.columns([1, 3])
    with col_button:
        predict_button = st.button('Predict Parkinsons', key='parkinsons_predict')

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
                render_result(True, "Parkinson's prediction", probability[0][1] * 100)
                st.warning('The model indicates elevated Parkinsons risk. Please consult a neurologist.')
            else:
                render_result(False, "Parkinson's prediction", probability[0][1] * 100)
                st.success('The model indicates lower Parkinsons risk based on the provided values.')

        except ValueError:
            st.error('Please enter valid numeric values for all fields.')
