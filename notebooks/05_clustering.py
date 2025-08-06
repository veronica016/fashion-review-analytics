import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "topic_reviews.csv")


# Drop NaNs
df = df.dropna(subset=['Age', 'Rating', 'Positive Feedback Count', 'Sentiment', 'Topic'])

# Map Sentiment to numbers
sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['Sentiment_Score'] = df['Sentiment'].map(sentiment_map)

# Features to cluster on
X = df[['Age', 'Rating', 'Positive Feedback Count', 'Sentiment_Score', 'Topic']]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
df['Persona'] = kmeans.fit_predict(X_scaled)

# Analyze each persona's profile
for i in range(5):
    print(f"\n--- Persona {i} ---")
    print(df[df['Persona'] == i][['Age', 'Rating', 'Positive Feedback Count', 'Sentiment_Score', 'Topic']].describe())
# Assign human-readable persona names
persona_labels = {
    0: "Luxury-Conscious Loyalist",
    1: "Detail-Oriented Evaluator",
    2: "Value-Seeking Researcher",
    3: "Fit-Focused Decision Maker",
    4: "Critical Feedback Provider"
}

df['Persona Label'] = df['Persona'].map(persona_labels)

# Save
df.to_csv("../data/persona_reviews.csv", index=False)
print("Clusters assigned.")

# Reduce to 2D for viz
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Persona Label', palette='tab10', data=df)

plt.title("Customer Segmentation Based on Behavior & Feedback", fontsize=14)
plt.xlabel("Shopping Behavior Pattern (compressed)", fontsize=12)
plt.ylabel("Review Style & Sentiment Pattern (compressed)", fontsize=12)

plt.legend(title="Identified Customer Personas", loc='best', fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.savefig("../assets/persona_clusters.png")

df.to_csv("../data/persona_reviews.csv", index=False)
