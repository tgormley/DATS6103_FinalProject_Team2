import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Load dataset
diabetes_data = pd.read_csv('C:/Users/smdkh/Downloads/diabetes_prediction_dataset.csv')

# Preprocessing: Handle missing values
diabetes_data['gender'] = diabetes_data['gender'].map({'Male': 1, 'Female': 0})
diabetes_data['gender'].fillna(diabetes_data['gender'].mode()[0], inplace=True)
diabetes_data['smoking_history'] = diabetes_data['smoking_history'].astype('category').cat.codes

# Define features and target
X = diabetes_data.drop(columns=['diabetes'])
y = diabetes_data['diabetes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Partial Dependence Plots for 'bmi' and 'blood_glucose_level'
PartialDependenceDisplay.from_estimator(
    model, X_test, ['bmi', 'blood_glucose_level'], kind='average', grid_resolution=50
)
plt.show()

# Threshold Analysis
high_risk_glucose = diabetes_data.groupby('diabetes')['blood_glucose_level'].mean()
high_risk_bmi = diabetes_data.groupby('diabetes')['bmi'].mean()

threshold_glucose = high_risk_glucose[1] * 0.9  # 90% of diabetic group mean
threshold_bmi = high_risk_bmi[1] * 0.9          # 90% of diabetic group mean

print(f"Average Blood Glucose Levels (Diabetes=1): {high_risk_glucose[1]:.2f}")
print(f"Average BMI (Diabetes=1): {high_risk_bmi[1]:.2f}")
print(f"Suggested Blood Glucose Threshold: {threshold_glucose:.2f}")
print(f"Suggested BMI Threshold: {threshold_bmi:.2f}")
