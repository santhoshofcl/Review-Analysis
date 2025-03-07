import pickle
import re
from flask import Flask, request, render_template, jsonify
from textblob import TextBlob

app = Flask(__name__)

# Load ML Model
with open("review_fake_detector.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    return "positive" if analysis.sentiment.polarity > 0 else "negative" if analysis.sentiment.polarity < 0 else "neutral"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    review_text = request.form.get("review", "")
    cleaned_text = clean_text(review_text)
    sentiment = get_sentiment(cleaned_text)
    fake_prob = model.predict_proba([cleaned_text])[0][1]
    fake_status = "Fake" if fake_prob > 0.5 else "Genuine"

    return jsonify({
        "review": review_text,
        "sentiment": sentiment,
        "fake_status": fake_status,
        "fake_probability": round(fake_prob, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
