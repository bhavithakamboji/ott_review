from flask import Flask, render_template, request
from model import predict_review

app = Flask(__name__)

def filter_by_age(label, age):
    age = int(age)

    # Kids
    if age < 13:
        if label == "Safe":
            return "Visible"
        else:
            return "Blocked for Kids"

    # Teens
    if 13 <= age < 18:
        if label == "Toxic":
            return "Blocked for Teens"
        else:
            return "Visible"

    # Adults
    return "Visible"


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    action = None

    if request.method == "POST":
        review = request.form["review"]
        age = request.form["age"]

        label = predict_review(review)
        action = filter_by_age(label, age)

        result = label

    return render_template("index.html", result=result, action=action)

if __name__ == "__main__":
    app.run(debug=True)
