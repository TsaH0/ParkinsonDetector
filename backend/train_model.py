import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

# =============================
# Load dataset
# =============================
df = pd.read_csv("audio_features.csv")
print("✅ Loaded dataset:", df.shape)

X = df.drop("label", axis=1)
y = df["label"]

# =============================
# Scale features
# =============================
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# =============================
# Train-test split (hold-out for final evaluation)
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# Parameter Grid
# =============================
param_grid = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
}

# =============================
# Base model
# =============================
xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)

# =============================
# 5-fold Stratified Cross-Validation
# =============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="accuracy",
    n_jobs=-1,
    cv=cv,
    verbose=2
)

# =============================
# Run Grid Search
# =============================
print("\n🚀 Running GridSearchCV (5-fold Stratified)...")
grid.fit(X_train, y_train)

print("\n✅ Best Parameters Found:")
print(grid.best_params_)

print("\n✅ Best Cross-Val Accuracy:")
print(round(grid.best_score_ * 100, 2), "%")

# =============================
# Train Final Model with Best Params
# =============================
print("\n🎯 Training final XGBoost model with best parameters...")
best_model = grid.best_estimator_
best_model.fit(X_train, y_train)

# =============================
# Evaluate on test set
# =============================
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n✅ FINAL TEST ACCURACY:", round(acc * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =============================
# Save Model + Scaler
# =============================
os.makedirs("models_new", exist_ok=True)
joblib.dump(best_model, "models_new/xgboost.joblib")
joblib.dump(scaler, "models_new/scaler.joblib")

print("\n✅ Saved model to models_new/xgboost.joblib")
print("✅ Saved scaler to models_new/scaler.joblib")
