from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Telecom Tower Prediction API is Running Successfully!"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return jsonify({
        "prediction": int(prediction[0])
    })

app.run(debug=True)
