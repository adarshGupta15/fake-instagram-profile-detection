from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)
    result = "FAKE PROFILE" if prediction[0] == 1 else "REAL PROFILE"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
