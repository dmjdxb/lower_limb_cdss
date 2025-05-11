# train_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load your dataset
df = pd.read_csv("lower_limb_20case_training_set.csv")

# Separate features (X) and label (y)
X = df.drop(columns=["confirmed_diagnosis"])
y = df["confirmed_diagnosis"]

# Encode the diagnoses
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train/test split (optional but best practice)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("📊 Classification Report:\n")
from sklearn.utils.multiclass import unique_labels
labels_used = unique_labels(y_test, y_pred)
target_names = encoder.inverse_transform(labels_used)
print(classification_report(y_test, y_pred, labels=labels_used, target_names=target_names))


# Save the model and label encoder
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
print("\n✅ Model and encoder saved.")
