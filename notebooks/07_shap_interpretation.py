import os
import shap
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Set base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model
model_path = os.path.join(base_dir, "..", "models", "recommendation_model.pkl")
model = joblib.load(model_path)

# Load the data
data_path = os.path.join(base_dir, "..", "data", "persona_reviews.csv")
df = pd.read_csv(data_path)

#Rename columns for consistency across app and model
df.rename(columns={
    "Positive Feedback Count": "Positive_Feedback_Count",
    "SentimentScore": "Sentiment_Score",
    "Persona": "Persona_ID",
    "Topic": "Topic_ID"
}, inplace=True)


# Drop rows with missing values
df.dropna(inplace=True)

# Select relevant features
features = ['Age', 'Positive_Feedback_Count', 'DepartmentEncoded', 'Sentiment_Score']
X = df[features]

# Run SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Save SHAP output
shap_output_path = os.path.join(base_dir, "..", "data", "shap_values.csv")
shap_df = pd.DataFrame(shap_values[1], columns=features)
shap_df.to_csv(shap_output_path, index=False)

# Summary Plot
shap.summary_plot(shap_values[1], X)
