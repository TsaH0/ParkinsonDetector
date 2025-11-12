# Parkinson’s Voice Detection App

## Overview
A cross-platform mobile application for early Parkinson’s disease screening using sustained voice recordings.  
Users record a 5–8 second “aaaah” sound, which is processed through a **FastAPI backend** that extracts vocal biomarkers and predicts the likelihood of Parkinson’s disease using a **hyperparameter-tuned XGBoost model**.

---

## Dataset and Preprocessing
The model was trained on **81 .wav recordings** from the [Figshare Parkinson’s Voice Dataset](https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127),  
containing both Parkinson’s and healthy control samples recorded using phone microphones.

Each audio file is processed with **librosa**:
- Normalization and silence trimming  
- Pitch extraction using `pyin`  
- Amplitude-based shimmer and jitter computation  
- MFCC-based spectral spread  
- Harmonic-to-noise ratio (HNR) and RPDE calculation  
- Detrended fluctuation analysis (DFA)  

A total of **22 biomedical voice features** are extracted and standardized using **RobustScaler** to handle outliers and wide dynamic ranges.

---

## Model Training
The final model is an **XGBoost classifier**, trained using **5-fold stratified cross-validation** for balanced evaluation.

**Best Hyperparameters:**
```python
n_estimators = 150  
learning_rate = 0.01  
max_depth = 4  
subsample = 1.0  
colsample_bytree = 0.7
```

## System Architecture
```flowchart TD
    A[User holds 'aaaah' for 5–8 seconds] --> B[React Native App (Expo, TypeScript)]
    B --> C[Audio recorded via Expo AV]
    C --> D[FastAPI Backend (Python)]
    D --> E[Feature Extraction (librosa)]
    E --> F[Feature Scaling (RobustScaler)]
    F --> G[XGBoost Model Prediction]
    G --> H[JSON Response]
    H --> I[Frontend displays probability, diagnosis, and features]
```
## Tech Stack

# Frontend

- Framework: React Native (Expo + TypeScript with Expo Router)

# Backend

- FastAPI
- XGBoost
- Sklearn

# Features

- Record button toggles start/stop recording

- Animated pulse while recording

- Full-screen loading overlay (“Analyzing your voice...”)

- Displays diagnosis and confidence score

- Multilingual support
