from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load('svm_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        float(data['Pregnancies']),
        float(data['Glucose']),
        float(data['BloodPressure']),
        float(data['SkinThickness']),
        float(data['Insulin']),
        float(data['BMI']),
        float(data['DiabetesPedigreeFunction']),
        float(data['Age'])
    ]
    scaled = scaler.transform([features])
    probability = model.predict_proba(scaled)[0][1] * 100
    return jsonify({'probability': round(probability, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
