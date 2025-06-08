from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('svm_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract features in correct order
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

        # Convert and scale
        input_array = np.array([features])
        input_scaled = scaler.transform(input_array)

        # Predict probability
        probability = model.predict_proba(input_scaled)[0][1] * 100
        return jsonify({"diabetes_chance_percent": round(probability, 2)})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
