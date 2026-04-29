import pickle
from pathlib import Path
import numpy as np

BASE_DIR = Path('.').resolve()
MODELS_DIR = BASE_DIR / 'models'

# Load models
diabetes_model = pickle.load(open(MODELS_DIR / 'Diabetes_model.sav', 'rb'))
heart_model = pickle.load(open(MODELS_DIR / 'Heart_model.sav', 'rb'))
park_model = pickle.load(open(MODELS_DIR / 'Parkinson_model.sav', 'rb'))

# Test with all zeros
print("=" * 60)
print("TESTING WITH ALL ZEROS")
print("=" * 60)

# Diabetes: 8 features
diabetes_zeros = [[0, 0, 0, 0, 0, 0, 0, 0]]
diabetes_pred = diabetes_model.predict(diabetes_zeros)
diabetes_prob = diabetes_model.predict_proba(diabetes_zeros)
print(f"\nDIABETES (8 features):")
print(f"Prediction (0/1): {diabetes_pred[0]}")
print(f"Probabilities: {diabetes_prob[0]}")
print(f"Risk percentage: {diabetes_prob[0][1] * 100:.2f}%")

# Heart: 13 features
heart_zeros = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
heart_pred = heart_model.predict(heart_zeros)
heart_prob = heart_model.predict_proba(heart_zeros)
print(f"\nHEART DISEASE (13 features):")
print(f"Prediction (0/1): {heart_pred[0]}")
print(f"Probabilities: {heart_prob[0]}")
print(f"Risk percentage: {heart_prob[0][1] * 100:.2f}%")

# Parkinson: 22 features
park_zeros = [[0] * 22]
park_pred = park_model.predict(park_zeros)
park_prob = park_model.predict_proba(park_zeros)
print(f"\nPARKINSON'S (22 features):")
print(f"Prediction (0/1): {park_pred[0]}")
print(f"Probabilities: {park_prob[0]}")
print(f"Risk percentage: {park_prob[0][1] * 100:.2f}%")

# Check if models have StandardScaler
print("\n" + "=" * 60)
print("MODEL STRUCTURE")
print("=" * 60)
print(f"Diabetes model: {diabetes_model}")
print(f"\nHeart model: {heart_model}")
print(f"\nParkinson model: {park_model}")
