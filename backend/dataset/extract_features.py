import os
import csv
from pathlib import Path
from tqdm import tqdm

import numpy as np
import librosa

# ----------------------------
# Use your FIXED feature extractor
# ----------------------------
def extract_audio_features(audio_path: str):
    try:
        y, sr = librosa.load(audio_path, sr=None)

        # Normalize loudness
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # Gentle trim
        y_trimmed, _ = librosa.effects.trim(y, top_db=35)

        # Ensure minimum length (0.3 sec)
        min_len = int(0.3 * sr)
        if len(y_trimmed) < min_len:
            y_trimmed = np.pad(y_trimmed, (0, min_len - len(y_trimmed)))

        # Pitch (pyin)
        f0, voiced_flags, voiced_probs = librosa.pyin(
            y_trimmed,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        # Clean pitch values
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) == 0:
            # fallback: centroid as pseudo pitch
            f0_valid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]

        fo = float(np.mean(f0_valid))
        fhi = float(np.max(f0_valid))
        flo = float(np.min(f0_valid))

        # Jitter
        f0_diff = np.abs(np.diff(f0_valid))
        jitter_abs = float(np.mean(f0_diff)) if len(f0_diff) else 0.0
        jitter_percent = float((jitter_abs / fo) * 100) if fo > 0 else 0.0

        # RAP
        if len(f0_valid) > 2:
            rap = float(
                np.mean(
                    np.abs(f0_valid[1:-1] - (f0_valid[:-2] + f0_valid[2:]) / 2)
                ) / fo
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

        # Amplitude envelope → shimmer
        S = np.abs(librosa.stft(y_trimmed))
        amp = np.mean(S, axis=0)

        amp_diff = np.abs(np.diff(amp))
        shimmer = float((np.mean(amp_diff) / np.mean(amp)) * 100)
        shimmer_db = float(20 * np.log10(np.mean(amp_diff) / np.mean(amp) + 1e-10))

        # APQ3
        if len(amp) > 3:
            apq3 = float(
                np.mean(
                    [np.abs(amp[i] - np.mean(amp[i - 1 : i + 2])) for i in range(1, len(amp) - 1)]
                ) / np.mean(amp)
            )
        else:
            apq3 = shimmer / 100

        # APQ5
        if len(amp) > 5:
            apq5 = float(
                np.mean(
                    [np.abs(amp[i] - np.mean(amp[i - 2 : i + 3])) for i in range(2, len(amp) - 2)]
                ) / np.mean(amp)
            )
        else:
            apq5 = apq3

        mdvp_apq = (apq3 + apq5) / 2
        shimmer_dda = apq3 * 3

        # Noise metrics
        harmonics = librosa.effects.harmonic(y_trimmed)
        noise = y_trimmed - harmonics
        nhr = float(np.sum(noise**2) / (np.sum(harmonics**2) + 1e-10))
        hnr = float(10 * np.log10(np.sum(harmonics**2) / (np.sum(noise**2) + 1e-10)))

        # RPDE
        pf = voiced_probs[voiced_probs > 0]
        rpde = float(
            -np.sum(pf * np.log2(pf)) / (len(pf) + 1e-10)
            if len(pf) else 0.0
        )

        # DFA
        cumsum = np.cumsum(f0_valid - np.mean(f0_valid))
        dfa = float(np.std(cumsum) / (len(cumsum) ** 0.5))

        # MFCC → spreads
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        spread1 = float(np.std(mfccs[1]))
        spread2 = float(np.std(mfccs[2]))

        # Spectral centroid variation
        centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
        d2 = float(np.std(centroids) / (np.mean(centroids) + 1e-10))

        # PPE
        f0_periods = 1 / (f0_valid + 1e-10)
        hist, _ = np.histogram(f0_periods, bins=30)
        prob = hist / (np.sum(hist) + 1e-10)
        ppe = float(-np.sum(prob[prob > 0] * np.log2(prob[prob > 0])))

        return [
            fo, fhi, flo,
            jitter_percent, jitter_abs, rap, ppq, jitter_ddp,
            shimmer, shimmer_db, apq3, apq5, mdvp_apq, shimmer_dda,
            nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe
        ]

    except Exception as e:
        print(f"Feature extraction failed for {audio_path}: {e}")
        return None

# ----------------------------
# EXTRACT FROM DATASET FOLDERS
# ----------------------------
HEADER = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
    "label"
]

rows = []

for folder, label in [("PD_AH", 1), ("HC_AH", 0)]:
    path = Path("dataset") / folder
    wav_files = list(path.glob("*.wav"))

    print(f"\nProcessing {folder} ({len(wav_files)} files)...")

    for wav in tqdm(wav_files):
        feats = extract_audio_features(str(wav))
        if feats:
            rows.append(feats + [label])

# Write CSV
with open("audio_features.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(rows)

print("\n✅ Feature extraction complete! Saved to audio_features.csv")

