from flask import Flask, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

app = Flask(__name__)
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

@app.route('/')
def home():
    return '''
    <h1>Fake News Detector</h1>
    <form method="POST" action="/predict">
        <input type="text" name="text" placeholder="Enter news article" style="width: 300px;">
        <input type="submit" value="Predict">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        text = data['text']
    else:
        text = request.form['text']
    pre_text = preprocess(text)
    vec = vectorizer.transform([pre_text])
    pred = model.predict(vec)[0]
    if request.is_json:
        return jsonify({'prediction': 'Fake' if pred == 1 else 'Real'})
    return f"Prediction: {'Fake' if pred == 1 else 'Real'}"

if __name__ == '__main__':
    app.run(debug=True)