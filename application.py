from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    sex = int(request.form["sex"])
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    smoker = int(request.form["smoker"])
    region = int(request.form["region"])

    features = np.array([[age, sex, bmi, children, smoker, region]])

    # Scale the features
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction_text=f"Estimated Insurance Charge: ₹ {prediction:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)

