import joblib
from flask import Flask, render_template, request, redirect, url_for, session
from preprocess import clean_text
import pandas as pd
from datetime import datetime
import os
import numpy as np
import csv


if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
else:
    print("Model files not found!")

app = Flask(__name__)
app.secret_key = "supersecretkey123"


# 🔥 Spoiler Detection
def detect_spoiler(review):
    spoiler_keywords = [
        "ending",
        "dies",
        "death",
        "killer",
        "murderer",
        "spoiled",
        "plot twist",
        "married in the end",
        "final episode",
        "revealed"
    ]
    review_lower = review.lower()
    return any(word in review_lower for word in spoiler_keywords)


# 🔥 Severity Engine
def get_severity(labels):
    severity_map = {
        "Safe": 0,
        "Spoiler": 1,
        "Insult": 2,
        "Obscene": 3,
        "Toxic": 3,
        "Severe Toxic": 4,
        "Threat": 4,
        "Identity Hate": 4
    }

    highest = 0
    for label in labels:
        highest = max(highest, severity_map.get(label, 0))

    return highest


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    action = None
    confidence = None

    if request.method == "POST":
        review = request.form["review"]
        age = int(request.form["age"])

        cleaned_review = clean_text(review)
        vectorized_review = vectorizer.transform([cleaned_review])

        prediction = model.predict(vectorized_review)[0]

        # 🔥 Confidence score using decision function
        decision_scores = model.decision_function(vectorized_review)
        max_score = np.max(np.abs(decision_scores))
        confidence = round(float(max_score), 2)

        labels = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate"
        ]

        predicted_labels = []

        for label, value in zip(labels, prediction):
            if value == 1:
                predicted_labels.append(label)

        # Safe logic
        if not predicted_labels and not detect_spoiler(review):
            predicted_labels.append("Safe")

        # Spoiler detection
        if detect_spoiler(review):
            if "Spoiler" not in predicted_labels:
                predicted_labels.append("Spoiler")

        # Pretty names
        pretty_map = {
            "toxic": "Toxic",
            "severe_toxic": "Severe Toxic",
            "obscene": "Obscene",
            "threat": "Threat",
            "insult": "Insult",
            "identity_hate": "Identity Hate"
        }

        predicted_labels = [pretty_map.get(label, label) for label in predicted_labels]

        result = predicted_labels

        # 🔥 Severity-based parental control
        severity = get_severity(result)

        if severity >= 4 and age < 18:
            action = "🚫 Blocked (High Risk Content)"
        elif severity == 3 and age < 16:
            action = "⚠ Restricted (16+ Content)"
        elif severity == 2 and age < 13:
            action = "⚠ Restricted (13+ Content)"
        else:
            action = "✅ Allowed"

        # 🔥 Save to moderation log
        log_file = "moderation_log.csv"

        log_data = {
            "Review": review,
            "Age": age,
            "Labels": ", ".join(result),
            "Status": action,
            "Confidence": confidence,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        log_df = pd.DataFrame([log_data])

        log_df.to_csv(
            log_file,
            mode="a",
            header=not os.path.exists(log_file),
            index=False,
            quoting=csv.QUOTE_ALL
        )

    return render_template(
        "index.html",
        result=result,
        action=action,
        confidence=confidence
    )


# 🔐 Admin Dashboard
@app.route("/admin")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("login"))

    log_file = "moderation_log.csv"

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        reviews = df.to_dict(orient="records")
    else:
        reviews = []

    return render_template("admin.html", reviews=reviews)


# 🔐 Delete Review
@app.route("/delete/<int:index>")
def delete_review(index):
    if not session.get("admin"):
        return redirect(url_for("login"))

    log_file = "moderation_log.csv"

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)

        if 0 <= index < len(df):
            df = df.drop(index)
            df = df.reset_index(drop=True)
            df.to_csv(log_file, index=False)

    return redirect(url_for("admin_dashboard"))


# 🔐 Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "admin123":
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")


# 🔐 Logout
@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("login"))




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)