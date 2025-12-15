# train_model.py
# Train a sign language classification model from collected hand landmark CSV files

import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ================= CONFIG =================
RAW_DATA_DIR = "data/raw"          # where individual label CSVs are stored
MODEL_DIR = "models"               # where trained model will be saved
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.pkl")
TEST_SIZE = 0.2
RANDOM_STATE = 42
# =========================================

os.makedirs(MODEL_DIR, exist_ok=True)

# -------- Load and merge all CSV files --------
csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}. Run collect_data.py first.")

X_list = []
y_list = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file, header=None)

    # Last column is the label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_list.append(X)
    y_list.append(y)

X = np.vstack(X_list)
y = np.concatenate(y_list)

print(f"Loaded samples: {X.shape[0]}")
print(f"Feature size: {X.shape[1]}")
print(f"Classes: {set(y)}")

# -------- Encode labels --------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -------- Train / Test split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)

# -------- Train model --------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------- Evaluate --------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# -------- Save model and encoder --------
joblib.dump({
    "model": model,
    "label_encoder": label_encoder
}, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print("Training complete")
