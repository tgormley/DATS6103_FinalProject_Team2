import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

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


# Select features for VIF calculation
features = ['age', 'bmi', 'hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level']
X = diabetes_data[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute VIF
vif_data = pd.DataFrame()
vif_data['Feature'] = features
vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print(vif_data)


## Define model for RFE
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Perform RFE
rfe = RFE(estimator=log_reg, n_features_to_select=5)
rfe.fit(diabetes_data[features], diabetes_data['diabetes'])

# Get selected features
selected_features = diabetes_data[features].columns[rfe.support_]
print("Selected Features:", selected_features)

##Random Forest

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


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_rf))

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_proba = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", auc)

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()



##Logistic regression

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



