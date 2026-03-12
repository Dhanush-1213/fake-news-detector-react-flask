from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Create app
app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return "Fake News Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data["text"]

    # Convert text to vector
    text_vector = vectorizer.transform([text])

    # Predict
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector).max()

    return jsonify({
        "prediction": prediction,
        "probability": float(probability)
    })


if __name__ == "__main__":
    app.run(debug=True)