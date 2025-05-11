
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from datetime import datetime

# --- Load training data ---
csv_path = "confirmed_cases.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError("No confirmed_cases.csv file found.")

df = pd.read_csv(csv_path, header=None)

# Match feature list to the app
symptoms = [
    "pain_at_rest", "night_pain", "radiating_pain", "numbness", "weakness", "bladder",
    "calf_swelling", "exertional_pain", "posterior_leg_pain", "medial_leg_pain", "lateral_leg_pain",
    "sudden_onset", "swelling", "tenderness", "morning_stiffness", "constant_pain",
    "pain_with_resisted_pf", "pain_with_dorsiflex_straight_knee", "pain_with_dorsiflex_bent_knee",
    "palpation_medial_tibia", "palpation_fibula"
]
df.columns = symptoms + ["confirmed_diagnosis"]

X = df[symptoms]
y = df["confirmed_diagnosis"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

if len(set(y_encoded)) < 2:
    raise ValueError("Need at least 2 unique diagnoses to train model.")

if len(X) < 5:
    X_train, y_train = X, y_encoded
    X_test, y_test = X, y_encoded
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model and save feature names
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
model.feature_names_in_ = X.columns.tolist()

# Evaluation
labels_present = sorted(set(y_test) | set(y_pred := model.predict(X_test)))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, labels=labels_present, target_names=encoder.inverse_transform(labels_present)))

# Save models
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(model, f"model_backup_{timestamp}.pkl")
joblib.dump(encoder, f"label_encoder_backup_{timestamp}.pkl")
print(f"\n✅ Model saved as model.pkl and versioned backup saved.")
