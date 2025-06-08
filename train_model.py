import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import joblib

# Load dataset
diabetes_data = pd.read_csv("D:\\ML\\diabetes_web\\diabetes.csv")

# Split features and labels
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

# Standardize the features
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train SVM with probability support
classifier = svm.SVC(kernel='linear', probability=True)
classifier.fit(X_train, Y_train)

# Save model and scaler
joblib.dump(classifier, 'svm_diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and Scaler saved.")