import re
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from textblob import TextBlob

# Load Dataset
data = pd.read_csv("reviews_dataset.csv")  # Ensure dataset has 'review' and 'fake' columns

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

data['cleaned_text'] = data['review'].apply(clean_text)

# Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    return "positive" if analysis.sentiment.polarity > 0 else "negative" if analysis.sentiment.polarity < 0 else "neutral"

data['sentiment'] = data['cleaned_text'].apply(get_sentiment)

# Train Model
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['fake'], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(X_train, y_train)

# Save Model
with open("review_fake_detector.pkl", "wb") as model_file:
    pickle.dump(pipeline, model_file)

print("✅ Model trained and saved successfully!")
