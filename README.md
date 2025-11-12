# Parkinson’s Voice Detection App

## Overview

A cross-platform mobile application for early Parkinson’s disease screening using sustained voice recordings.  
Users record a 5–8 second “a” vowel sound, which is processed through a **FastAPI backend** that extracts vocal biomarkers and predicts the likelihood of Parkinson’s disease using a **hyperparameter-tuned XGBoost model**.

---

## Dataset and Preprocessing

The model was trained on **81 .wav recordings** from the [Figshare Parkinson’s Voice Dataset](https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127), containing both Parkinson’s and healthy control samples recorded using phone microphones.

Each audio file is processed with **librosa**:

* Normalization and silence trimming
* Pitch extraction using `pyin`
* Amplitude-based shimmer and jitter computation
* MFCC-based spectral spread
* Harmonic-to-noise ratio (HNR) and RPDE calculation
* Detrended fluctuation analysis (DFA)

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

The model is optimized for realistic smartphone audio and small dataset size.

---

## System Architecture
<img width="1764" height="2536" alt="Untitled diagram-2025-11-12-004831" src="https://github.com/user-attachments/assets/b1a925e0-a086-4d4a-a46d-cc726a76a249" />



---

## Frontend

**Framework:** React Native (Expo + TypeScript with Expo Router)

**Folder Structure:**

```
app/
├── _layout.tsx
├── index.tsx
└── (tabs)/
    ├── _layout.tsx
    ├── home.tsx
    ├── record.tsx
    ├── results.tsx
```

**Features:**

* Record button toggles start/stop recording
* Animated pulse while recording
* Full-screen loading overlay (“Analyzing your voice...”)
* Displays diagnosis and confidence score
* Multilingual support

---

## Backend

**Framework:** FastAPI (Python)

**Core Components:**

* `main.py`: Manages routes and integrates the trained model
* `/predict`: Accepts `.wav`, `.mp3`, `.m4a` files
* `extract_audio_features`: Extracts 22 key biomedical voice features
* `RobustScaler`: Normalizes features before inference
* `XGBoost Model`: Outputs Parkinson’s probability and confidence

**Sample API Response:**

```json
{
  "status": "success",
  "probability": 0.78,
  "diagnosis": "Likely Parkinson’s",
  "confidence": "high",
  "features": {
    "Fo": 195.43,
    "Jitter(%)": 1.14,
    "HNR": 9.22,
    "Shimmer": 0.0821,
    "RPDE": 0.214,
    "DFA": 7.82
  }
}
```

---

## Usage

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npx expo start
```

### Run

1. Launch the Expo app on your phone or emulator.
2. Tap “Start Voice Test” and hold “aaaah” for 5–8 seconds.
3. Wait for the backend to process and view your diagnosis.

---

## Tech Stack

| Layer            | Technology                             |
| ---------------- | -------------------------------------- |
| Frontend         | React Native (Expo, TypeScript)        |
| Backend          | FastAPI, Python                        |
| Machine Learning | XGBoost, RobustScaler                  |
| Audio Processing | Librosa, NumPy, SoundFile              |
| Communication    | REST API (Fetch + multipart/form-data) |

---

## Project Folder Structure

```
parkinsons-voice-app/
├── backend/
│   ├── main.py
│   ├── model/
│   │   ├── xgboost_model.json
│   │   └── scaler.pkl
│   ├── utils/
│   │   └── feature_extraction.py
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── app/
│   │   ├── _layout.tsx
│   │   ├── index.tsx
│   │   └── (tabs)/
│   │       ├── _layout.tsx
│   │       ├── home.tsx
│   │       ├── record.tsx
│   │       └── results.tsx
│   ├── components/
│   │   ├── RecordingButton.tsx
│   │   ├── ResultDisplay.tsx
│   │   └── LoadingOverlay.tsx
│   ├── assets/
│   ├── constants/
│   ├── services/
│   │   └── api.ts
│   ├── app.json
│   ├── package.json
│   └── tsconfig.json
├── .gitignore
└── README.md
```

---

## Notes

* Works best when recorded **10–15 cm away** from the microphone.
* Supports **.wav**, **.mp3**, and **.m4a** file formats.
* Intended for **screening**, not medical diagnosis.

---

## Example Workflow

1. User records a short sustained “aaaah” sound.
2. The app uploads it to the FastAPI backend.
3. The backend extracts acoustic and nonlinear voice features.
4. Features are scaled with RobustScaler.
5. The XGBoost model predicts Parkinson’s probability.
6. The frontend displays the diagnosis and confidence score.

---

## Conclusion

By regenerating features from realistic mobile recordings and retraining a tuned model, this system provides a reliable, multilingual, and smartphone-optimized approach for early Parkinson’s screening.

---

*Copy and paste this entire file as `README.md` into your repository root.*
