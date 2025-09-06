import requests

url = 'http://127.0.0.1:5000/predict'
articles = [
    {"text": "Aliens invade Earth and demand chocolate."},
    {"text": "Government announces new tax cuts for small businesses."}
]
for article in articles:
    response = requests.post(url, json=article)
    print(f"Article: {article['text']}")
    print(f"Prediction: {response.json()['prediction']}\n")