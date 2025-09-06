import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens) if tokens else ' '

def predict_fake_news(new_text):
    pre_text = preprocess(new_text)
    vec = vectorizer.transform([pre_text])
    pred = model.predict(vec)[0]
    return "Fake" if pred == 1 else "Real"

# Test
articles = [
    "New vaccine approved by health authorities.",
    "Moon landing was faked, claims new study."
]
for article in articles:
    print(f"Prediction for '{article}': {predict_fake_news(article)}")