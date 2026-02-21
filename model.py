import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

from preprocess import clean_text
from sklearn.svm import LinearSVC
# Load dataset
data = pd.read_csv("dataset.csv")

# Preprocess
data["cleaned"] = data["review"].apply(clean_text)

X = data["cleaned"]

y = data[[
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]]

vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# 🔥 Multi-label classifier


model = OneVsRestClassifier(LinearSVC())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMulti-Label Classification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Multi-label model saved successfully!")