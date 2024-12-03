import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
diabetes_data = pd.read_csv('diabetes_prediction_dataset.csv')

# Inspect the dataset
print(diabetes_data.info())
print(diabetes_data.head())

# Check for missing values
print(diabetes_data.isnull().sum())

# One-hot encoding for categorical variables
diabetes_data = pd.get_dummies(diabetes_data, columns=['gender', 'smoking_history'], drop_first=True)

# Compute the correlation matrix
correlation_matrix = diabetes_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# Define features and target
features = ['age', 'bmi', 'hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level']
target = 'diabetes'

X = diabetes_data[features]
y = diabetes_data[target]

# Define model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Perform RFE
rfe = RFE(estimator=log_reg, n_features_to_select=5)
rfe.fit(diabetes_data[features], diabetes_data['diabetes'])

# Get selected features
selected_features = diabetes_data[features].columns[rfe.support_]
print("Selected Features:", selected_features)




# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Feature importance
importances = rf.feature_importances_
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
print(feature_importances.sort_values(by='Importance', ascending=False))

rf_features = {
    'Feature': ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age', 'hypertension', 'heart_disease'],
    'Importance': [0.407410, 0.327367, 0.144707, 0.101647, 0.011596, 0.007273]
}
rf_df = pd.DataFrame(rf_features)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=rf_df.sort_values(by='Importance', ascending=False), palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()


from sklearn.linear_model import LogisticRegression

# Train a Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_lr = log_reg.predict(X_test)

# Evaluate the model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# Extract coefficients
coefficients = pd.DataFrame({'Feature': features, 'Coefficient': log_reg.coef_[0]})
print(coefficients.sort_values(by='Coefficient', ascending=False))
