import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text

# Load dataset
data = pd.read_csv("dataset.csv")

# Clean text
data["review"] = data["review"].apply(clean_text)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["review"])
y = data["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

def predict_review(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return prediction[0]
