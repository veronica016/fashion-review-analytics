import pandas as pd
import gensim
from gensim import corpora
import spacy

# Load data
df = pd.read_csv("../data/sentiment_reviews.csv")
texts = df['Review Text'].dropna().astype(str).tolist()

# Preprocess text using spaCy
nlp = spacy.load("en_core_web_sm")
def preprocess(texts):
    tokens = []
    for i, doc in enumerate(nlp.pipe(texts, disable=["ner", "parser"])):
        tokens.append([token.lemma_.lower() for token in doc
                       if token.is_alpha and not token.is_stop])
        if (i + 1) % 1000 == 0:
            print(f" Processed {i + 1}/{len(texts)} reviews...")

    return tokens

print("Starting preprocessing...")

tokenized = preprocess(texts)  # isolate the issue, get control back

print("Finished preprocessing.")

# Create dictionary + corpus for Gensim
dictionary = corpora.Dictionary(tokenized)
corpus = [dictionary.doc2bow(text) for text in tokenized]

# Train LDA Model
lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=5,  # adjust based on how many themes you want
    passes=10,
    random_state=42
)

# Show topics
for idx, topic in lda_model.print_topics(-1):
    print(f" Topic {idx}: {topic}")

# Assign dominant topic to each review
def get_dominant_topic(bow):
    topics = lda_model.get_document_topics(bow)
    return sorted(topics, key=lambda x: -x[1])[0][0] if topics else None

df['Topic'] = [get_dominant_topic(bow) for bow in corpus]

# Map numeric topic to human-readable labels
topic_labels = {
    0: "Material & Fabric Complaints",
    1: "Sizing & Fit Issues",
    2: "Dress Praise & Fit",
    3: "Style & Comfort Praise",
    4: "Fit & Length Concerns"
}

df['Topic Label'] = df['Topic'].map(topic_labels)

# Save it
df.to_csv("../data/topic_reviews.csv", index=False)
print("Topics assigned + saved.")
