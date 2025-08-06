import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# Load enriched dataset
df = pd.read_csv("../data/persona_reviews.csv")

# Drop missing target
df = df.dropna(subset=["Recommended IND"])

# Define features and target
features = [
    'Age', 'Rating', 'Positive Feedback Count', 'Sentiment_Score',
    'Persona', 'Topic'
]
X = df[features]
y = df["Recommended IND"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Initialize and train XGBoost
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, "../models/recommendation_model.pkl")
