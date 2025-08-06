import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np


# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
ASSETS_DIR = os.path.join(BASE_DIR, "..", "assets")

# --- LOAD DATA ---
df = pd.read_csv(os.path.join(DATA_DIR, "persona_reviews.csv")).dropna()

# --- LOAD MODEL ---
model = joblib.load(os.path.join(MODELS_DIR, "recommendation_model.pkl"))

# --- UI HEADER ---
st.markdown(
    "<h1 style='text-align: center; color: #FF69B4;'>AI-POWERED FASHION REVIEW ANALYZER</h1>",
    unsafe_allow_html=True
)
st.caption("Analyze thousands of reviews to understand what drives product recommendations.")

st.markdown("---")

# --- SENTIMENT DISTRIBUTION ---
st.subheader("Overall Sentiment Distribution ")
st.caption("Gauge how customers feel — from joy to frustration — using VADER sentiment analysis.")
st.bar_chart(df['Sentiment'].value_counts())

st.markdown("---")

# --- SENTIMENT BY DEPARTMENT ---
st.subheader("Sentiment by Department ")
st.caption("See how tone varies across departments like Dresses, Tops, Intimates.")
dept_sentiment = df.groupby(['Department Name', 'Sentiment']).size().unstack().fillna(0)
st.bar_chart(dept_sentiment)

st.markdown("---")

# --- SAMPLE REVIEWS ---
st.subheader("Sample Reviews")
selected_dept = st.selectbox("Pick a Department", sorted(df['Department Name'].dropna().unique()))
filtered = df[df['Department Name'] == selected_dept]

selected_sentiment = st.radio("Filter by Sentiment", ['Positive', 'Neutral', 'Negative'])
filtered = filtered[filtered['Sentiment'] == selected_sentiment]

st.write(filtered[['Age', 'Rating', 'Review Text', 'Sentiment']].sample(min(5, len(filtered)), random_state=42))

st.markdown("---")

# --- TOPIC DISTRIBUTION ---
st.subheader("Topic Breakdown ")
st.caption("Top 5 topics extracted via LDA — like sizing issues, fabric complaints, etc.")
st.bar_chart(df['Topic Label'].value_counts())

st.markdown("---")

# --- REVIEWS BY TOPIC ---
st.subheader("Example Reviews by Topic")
selected_topic = st.selectbox("Pick a Topic", sorted(df['Topic Label'].dropna().unique()))
topic_reviews = df[df['Topic Label'] == selected_topic]
st.write(topic_reviews[['Review Text', 'Sentiment']].sample(min(5, len(topic_reviews)), random_state=42))

st.markdown("---")

# --- PERSONA VISUALIZATION ---
st.subheader("Customer Personas ")
st.caption("Clusters from KMeans show distinct shopper behavior profiles.")
st.markdown("Visualized using KMeans + PCA — each dot = one reviewer.")
st.image(os.path.join(ASSETS_DIR, "persona_clusters.png"))
st.caption("Personas cluster shoppers based on behavior, tone, and preferences.")

st.markdown("---")

# --- EXPLORE BY PERSONA ---
st.subheader("Explore Reviews by Persona")
persona_label = st.selectbox("Pick a Persona", sorted(df['Persona Label'].unique()))
persona_reviews = df[df['Persona Label'] == persona_label]
st.write(persona_reviews[['Age', 'Rating', 'Review Text', 'Sentiment', 'Topic Label']].sample(min(5, len(persona_reviews)), random_state=42))

# --- SIDEBAR SIMULATION ---
# --- SIDEBAR SIMULATION (BETTER LOOKING) ---
with st.sidebar:
    st.markdown("<h2 style='color:#FF69B4;'> Shopper Simulation</h2>", unsafe_allow_html=True)
    st.caption("Adjust the traits to see if this person would recommend the product 💬")

    age = st.slider("Age", 18, 80, 30)
    rating = st.selectbox("Rating Given", [1, 2, 3, 4, 5])
    feedback = st.slider("Helpful Votes", 0, 100, 5)
    sentiment = st.slider("Sentiment Score", -1.0, 1.0, 0.0, step=0.1)

    st.info("Sentiment ranges from -1 (angry) to +1 (ecstatic)")


# Input: Encoded Persona & Topic
persona_map = dict(zip(df["Persona"], df["Persona Label"]))
persona_label_map = {v: k for k, v in persona_map.items()}
persona_label = st.sidebar.selectbox("Shopper Persona", sorted(persona_label_map.keys()))
persona = persona_label_map[persona_label]

topic_map = dict(zip(df["Topic"], df["Topic Label"]))
topic_label_map = {v: k for k, v in topic_map.items()}
topic_label = st.sidebar.selectbox("Review Topic", sorted(topic_label_map.keys()))
topic = topic_label_map[topic_label]

# Predict
user_input = pd.DataFrame([{
    "Age": age,
    "Rating": rating,
    "Positive Feedback Count": feedback,
    "Sentiment_Score": sentiment,
    "Persona": persona,
    "Topic": topic
}])

prediction = model.predict(user_input)[0]
if prediction == 1:
    st.sidebar.success("Prediction: **Yes — this shopper would recommend it!**")
else:
    st.sidebar.error("Prediction: **No — this shopper wouldn’t recommend it.**")

# Add SHAP-based explanations to build trust in predictions

# --- SHAP EXPLAINABILITY ---
with st.expander("Model Explainability (SHAP)"):
    st.subheader("Model Explainability (SHAP) 📈")
    st.caption("Visualize exactly why the model made each prediction.")

    # Prepare input for SHAP
    shap_input = df[[
        'Age', 'Rating', 'Positive Feedback Count',
        'Sentiment_Score', 'Persona', 'Topic'
    ]].copy()

    # Label encode 'Persona' and 'Topic' exactly like training
    for col in ['Persona', 'Topic']:
        le = LabelEncoder()
        shap_input[col] = le.fit_transform(shap_input[col])

    # Build SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_input)

    st.markdown("---")

    # Top 5 SHAP Feature Importance Bar Chart
    st.subheader("Top 5 Feature Drivers")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": shap_input.columns,
        "Importance": mean_abs_shap
    }).sort_values(by="Importance", ascending=False).head(5)

    import seaborn as sns  # make sure seaborn is imported

    fig_bar, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
    st.pyplot(fig_bar)

    st.markdown("---")

    #  GLOBAL FEATURE IMPORTANCE
    st.subheader("Global Feature Importance")
    fig_summary, ax = plt.subplots()
    shap.summary_plot(shap_values, shap_input, show=False)
    st.pyplot(fig_summary)
    plt.clf()

    st.markdown("---")

    #  LOCAL EXPLANATION (per review)
    st.subheader("Local Explanation")
    review_index = st.slider("Pick review index", 0, len(shap_input) - 1, 10)
    st.write(df.iloc[review_index][['Review Text']])


    # SHAP Decision Plot
    # SHAP decision plot (clean fix)
    import matplotlib.pyplot as plt

    st.subheader("SHAP Decision Plot")

    # Create the plot using matplotlib backend
    plt.figure()
    shap.decision_plot(
        explainer.expected_value,
        shap_values[review_index],
        shap_input.iloc[review_index],
        feature_names=shap_input.columns.tolist(),
        show=False
    )
    st.pyplot(plt.gcf())  # Show current figure
    plt.clf()

    st.markdown("""
     **How to Read This:**  
    - **Positive SHAP value** → pushes the model toward predicting "Recommend".
    - **Negative SHAP value** → pushes toward "Not Recommend".
    - The bigger the bar, the more influence that feature had.
    """)

