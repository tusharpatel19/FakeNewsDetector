# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK data
nltk.download('stopwords', quiet=True)

# Step 1: Load Synthetic Data (larger dataset)
data = {
    'text': [
        # Real news (label 0)
        "The economy is growing steadily according to official reports.",
        "New government policy aims to reduce taxes for middle class.",
        "Election results announced with record voter turnout.",
        "Stock market reaches new all-time high amid positive earnings.",
        "Scientists confirm new species discovered in Pacific Ocean.",
        "Local hospital opens new wing for pediatric care.",
        "Government invests in renewable energy projects nationwide.",
        "New study shows benefits of regular exercise on mental health.",
        # Fake news (label 1)
        "Scientists discover a cure for all diseases overnight in secret lab.",
        "Celebrity revealed to be a robot from the future in shocking video.",
        "Aliens invade Earth and demand all the world's chocolate supply.",
        "Moon confirmed to be made of green cheese by NASA astronauts.",
        "Secret government plan to control weather exposed by insider.",
        "Elvis sighted alive on Mars playing guitar with aliens.",
        "Time travel device invented in garage, claims anonymous blogger.",
        "World's richest person secretly lives in underwater city."
    ],
    'label': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # 8 real, 8 fake
}
df = pd.DataFrame(data)
print("Using synthetic data with", len(df), "samples.")

# Step 2: Preprocess the Text
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub(r'<.*?>', '', str(text))  # Remove HTML tags, ensure text is string
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-letters
    text = text.lower()  # Lowercase
    tokens = text.split()  # Split into words
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]  # Stem, remove stopwords
    return ' '.join(tokens) if tokens else ' '  # Return space if empty

X = df['text'].apply(preprocess)  # Preprocess all texts
y = df['label']

# Step 3: Split Data (stratified to balance classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Step 4: Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000)  # Reduced features for small dataset
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train the Model
model = LogisticRegression(max_iter=1000)  # Increased iterations for convergence
model.fit(X_train_vec, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, zero_division=0))  # Suppress warnings

# Step 7: Save Model and Vectorizer
import joblib
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer saved.")

# Step 8: Make Predictions on New Text
def predict_fake_news(new_text):
    pre_text = preprocess(new_text)
    vec = vectorizer.transform([pre_text])
    pred = model.predict(vec)[0]
    return "Fake" if pred == 1 else "Real"

# Additional test predictions
more_articles = [
    "New study finds coffee improves memory in adults.",
    "Secret government base discovered on the dark side of the moon.",
    "Local charity raises funds for homeless shelter.",
    "Time machine invented, available for sale next week."
]
for article in more_articles:
    print(f"Prediction for '{article}': {predict_fake_news(article)}")