import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

features = ['age', 'bmi', 'hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level']

# Define model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Perform RFE
rfe = RFE(estimator=log_reg, n_features_to_select=5)
rfe.fit(diabetes_data[features], diabetes_data['diabetes'])

# Get selected features
selected_features = diabetes_data[features].columns[rfe.support_]
print("Selected Features:", selected_features)


