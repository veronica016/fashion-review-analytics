import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/cleaned_reviews.csv")

# Ratings by department
sns.countplot(x='Rating', hue='Department Name', data=df)
plt.title("Ratings by Department")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../assets/ratings_by_department.png")

# Age distribution
sns.histplot(df['Age'], bins=20)
plt.title("Customer Age Distribution")
plt.savefig("../assets/age_distribution.png")

print("Charts saved in assets/")
