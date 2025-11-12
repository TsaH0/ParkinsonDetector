import os
import numpy as np
import joblib 
import xgboost
from xgboost import XGBClassifier
import logging
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
import csv
import datetime
import librosa
import tempfile
from typing import Dict,Any
from pathlib import Path    
import soundfile as sf

app = FastAPI(title="API for Parkinson Voice Detection",description="Analyzes the voice recordings to check whether the consumer has parkinsons disease or not.",version="1.0.0")

app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials = True,allow_methods = ["*"],allow_headers=["*"])

scaler = None
model = None
LOG_FILE = "predictions_log.csv"

FEATURE_NAMES =  [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", 
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

def load_models():
    global scaler, model 
    try:
        if not os.path.exists("./Models/scaler.joblib"):
            logger.error("scaler.joblib not found.")
            raise FileNotFoundError("scaler.joblib was not found in the Models directory")
        if not os.path.exists("./Models/xgboost.joblib"):
            logger.error("xgboost.joblib not found.")
            raise FileNotFoundError("xgboost.joblib was not found in the Models directory")

        scaler = joblib.load("./Models/scaler.joblib")
        model = joblib.load("./Models/xgboost.joblib")
        logger.info("Models and Scaler was loaded successfully")
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE,'w',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp','probability','diagnosis']+FEATURE_NAMES)

    except Exception as e:
        logger.error(f"Faced an error:{str(e)}")
        raise
@app.on_event("startup")
async def startup_event():
    """Load models as soon as the site is opened"""
    load_models()

@app.get("/health")
async def health_check():
    """Health Check Point"""
    model_status = "loaded" if model is not None and scaler is not None else "Not loaded"
    return {
        "status":"ok",
        "timestamp":datetime.now().isoformat(),
        "model_status":model_status
    }
def extract_audio_features(audio_path: str) -> Dict[str, float]:
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)

        # Normalize audio
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # Trim silence gently
        y_trimmed, _ = librosa.effects.trim(y, top_db=35)

        # Ensure minimum length (0.3 sec)
        min_len = int(0.3 * sr)
        if len(y_trimmed) < min_len:
            y_trimmed = np.pad(y_trimmed, (0, min_len - len(y_trimmed)))

        # Extract F0 (pitch)
        f0, voiced_flags, voiced_probs = librosa.pyin(
            y_trimmed,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )

        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) == 0:
            # fallback – use spectral centroid to approximate pitch
            f0_valid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]

        fo = float(np.mean(f0_valid))
        fhi = float(np.max(f0_valid))
        flo = float(np.min(f0_valid))

        # Jitter
        f0_diff = np.abs(np.diff(f0_valid))
        jitter_abs = float(np.mean(f0_diff)) if len(f0_diff) > 0 else 0.0
        jitter_percent = float((jitter_abs / fo) * 100) if fo > 0 else 0.0

        # RAP
        if len(f0_valid) > 2:
            rap = float(
                np.mean(
                    np.abs(f0_valid[1:-1] - (f0_valid[:-2] + f0_valid[2:]) / 2)
                )
                / fo
            )
        else:
            rap = 0.0

        # PPQ
        if len(f0_valid) >= 5:
            ppq = float(
                np.mean(
                    [
                        np.abs(f0_valid[i] - np.mean(f0_valid[i - 2 : i + 3]))
                        for i in range(2, len(f0_valid) - 2)
                    ]
                )
                / fo
            )
        else:
            ppq = rap

        jitter_ddp = rap * 3

        # Amplitude envelope
        S = np.abs(librosa.stft(y_trimmed))
        amp = np.mean(S, axis=0)

        amp_diff = np.abs(np.diff(amp))
        shimmer = float((np.mean(amp_diff) / np.mean(amp)) * 100)
        shimmer_db = float(
            20 * np.log10(np.mean(amp_diff) / np.mean(amp) + 1e-10)
        )

        # APQ3
        if len(amp) > 3:
            apq3 = float(
                np.mean(
                    [np.abs(amp[i] - np.mean(amp[i - 1 : i + 2])) for i in range(1, len(amp) - 1)]
                )
                / np.mean(amp)
            )
        else:
            apq3 = shimmer / 100

        # APQ5
        if len(amp) > 5:
            apq5 = float(
                np.mean(
                    [np.abs(amp[i] - np.mean(amp[i - 2 : i + 3])) for i in range(2, len(amp) - 2)]
                )
                / np.mean(amp)
            )
        else:
            apq5 = apq3

        mdvp_apq = (apq3 + apq5) / 2
        shimmer_dda = apq3 * 3

        # Noise ratio
        harmonics = librosa.effects.harmonic(y_trimmed)
        noise = y_trimmed - harmonics
        nhr = float(np.sum(noise**2) / (np.sum(harmonics**2) + 1e-10))
        hnr = float(
            10 * np.log10(np.sum(harmonics**2) / (np.sum(noise**2) + 1e-10))
        )

        # RPDE safe computation
        pf = voiced_probs[voiced_probs > 0]
        rpde = float(
            -np.sum(pf * np.log2(pf)) / (len(pf) + 1e-10)
            if len(pf) > 0
            else 0.0
        )

        # DFA
        cumsum = np.cumsum(f0_valid - np.mean(f0_valid))
        dfa = float(np.std(cumsum) / (len(cumsum) ** 0.5))

        # MFCC-based features
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        spread1 = float(np.std(mfccs[1]))
        spread2 = float(np.std(mfccs[2]))

        spectral_centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
        d2 = float(np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-10))

        # PPE
        f0_periods = 1 / (f0_valid + 1e-10)
        hist, _ = np.histogram(f0_periods, bins=30)
        prob = hist / (np.sum(hist) + 1e-10)
        ppe = float(-np.sum(prob[prob > 0] * np.log2(prob[prob > 0])))

        return {
            "MDVP:Fo(Hz)": fo,
            "MDVP:Fhi(Hz)": fhi,
            "MDVP:Flo(Hz)": flo,
            "MDVP:Jitter(%)": jitter_percent,
            "MDVP:Jitter(Abs)": jitter_abs,
            "MDVP:RAP": rap,
            "MDVP:PPQ": ppq,
            "Jitter:DDP": jitter_ddp,
            "MDVP:Shimmer": shimmer,
            "MDVP:Shimmer(dB)": shimmer_db,
            "Shimmer:APQ3": apq3,
            "Shimmer:APQ5": apq5,
            "MDVP:APQ": mdvp_apq,
            "Shimmer:DDA": shimmer_dda,
            "NHR": nhr,
            "HNR": hnr,
            "RPDE": rpde,
            "DFA": dfa,
            "spread1": spread1,
            "spread2": spread2,
            "D2": d2,
            "PPE": ppe,
        }

    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        raise ValueError(f"Failed to extract features: {str(e)}")


def log_prediction(probability:float,diagnosis:str,features:Dict[str,float]):
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                datetime.now().isoformat(),
                probability,
                diagnosis
            ] + [features[name] for name in FEATURE_NAMES]
            writer.writerow(row)
    except Exception as e:
        logger.error(f"Failed to log prediction: {str(e)}")

@app.post("/predict")
async def predict_parkinsons(audio: UploadFile = File(...)):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # ✅ Allowed formats
    allowed_exts = (".wav", ".mp3", ".m4a")
    allowed_types = (
        "audio/wav", "audio/x-wav",
        "audio/mpeg", "audio/mp3",
        "audio/aac", "audio/m4a", "audio/mp4"
    )

    filename = (audio.filename or "").lower()
    content_type = audio.content_type or ""

    # ✅ If no extension → treat as m4a
    suffix = Path(filename).suffix
    if suffix == "":
        suffix = ".m4a"

    # ✅ Block only truly unsupported formats
    if not (filename.endswith(allowed_exts) or content_type in allowed_types):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Allowed: wav, mp3, m4a"
        )

    temp_file = None
    try:
        # ✅ Create temporary local file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await audio.read())
        tmp.close()
        temp_file = tmp.name

        logger.info(f"Processing file: {filename} saved as {temp_file}")

        # ✅ Extract audio features
        features = extract_audio_features(temp_file)

        # ✅ Build vector shape (1, 22)
        try:
            feature_vector = np.array([[features[name] for name in FEATURE_NAMES]])
        except KeyError as missing:
            raise HTTPException(
                status_code=500,
                detail=f"Missing feature in extracted data: {missing}"
            )

        if feature_vector.shape != (1, len(FEATURE_NAMES)):
            raise HTTPException(
                status_code=500,
                detail=f"Bad feature vector shape: {feature_vector.shape}, expected (1,{len(FEATURE_NAMES)})"
            )

        # ✅ Scale features
        features_scaled = scaler.transform(feature_vector)

        # ✅ Model prediction (returns [p(HC), p(PD)])
        proba = model.predict_proba(features_scaled)[0]
        parkinsons_probability = float(proba[1])

        # ✅ Threshold logic (your improved version)
        if parkinsons_probability >= 0.7:
            diagnosis = "Likely Parkinsons"
        elif parkinsons_probability >= 0.5:
            diagnosis = "Uncertain Diagnosis - Medical Check Up is recommended"
        else:
            diagnosis = "Unlikely Parkinsons"

        # ✅ Log output
        log_prediction(parkinsons_probability, diagnosis, features)

        # ✅ Response payload
        response = {
            "status": "success",
            "probability": round(parkinsons_probability, 4),
            "diagnosis": diagnosis,
            "confidence": (
                "high"
                if parkinsons_probability >= 0.7 or parkinsons_probability <= 0.3
                else "moderate"
            ),
            "features": {
                "Fo": round(features["MDVP:Fo(Hz)"], 2),
                "Jitter(%)": round(features["MDVP:Jitter(%)"], 6),
                "HNR": round(features["HNR"], 2),
                "Shimmer": round(features["MDVP:Shimmer"], 4),
                "RPDE": round(features["RPDE"], 4),
                "DFA": round(features["DFA"], 4)
            },
            "all_features": features
        }

        logger.info(
            f"Prediction complete: {diagnosis} (PD Probability: {parkinsons_probability:.4f})"
        )

        return JSONResponse(content=response)

    except ValueError as ve:
        logger.error(f"Feature extraction error: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        # ✅ Always delete temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Failed to delete temp file: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Parkinson's Voice Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload audio for prediction",
            "/health": "GET - Check API health status",
            "/": "GET - API information"
        },
        "supported_formats": [".wav", ".mp3"],
        "features_extracted": len(FEATURE_NAMES)
    }

if __name__ =="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",port = 8000,reload = True)
