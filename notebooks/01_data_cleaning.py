import pandas as pd

# Load CSV
df = pd.read_csv("../data/raw_reviews.csv")

# Drop empty reviews
df = df[df['Review Text'].notna()]

# Keep only what you need
df = df[['Age', 'Title', 'Review Text', 'Rating', 'Recommended IND', 'Department Name', 'Class Name', 'Positive Feedback Count']]

# Save cleaned version
df.to_csv("../data/cleaned_reviews.csv", index=False)

print("Cleaned data saved.")
