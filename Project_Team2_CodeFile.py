#%%

#############
## Imports ##
#############

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

###############
## Load Data ##
###############

diabetes_data = pd.read_csv('diabetes_prediction_dataset.csv')

print(diabetes_data.info())
print(diabetes_data.head())


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

################
## Data Preprocessing ##
################
# Check for missing values
print(diabetes_data.isnull().sum())
print(diabetes_data.describe())

# One-hot encoding for categorical variables
diabetes_data = pd.get_dummies(diabetes_data, columns=['gender', 'smoking_history'], drop_first=True)

print(diabetes_data.head())

#%%
################
## Smart Question 1 ##
################


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#


#%%
################
## Smart Question 2 ##
################


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%
################
## Smart Question 3 ##
################

# Compute the correlation matrix
correlation_matrix = diabetes_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Define features and target
features = ['age', 'bmi', 'hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level','gender_Male','gender_Other','smoking_history_current','smoking_history_ever','smoking_history_former','smoking_history_never','smoking_history_not current']
target = 'diabetes'

X = diabetes_data[features]
y = diabetes_data[target]


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute VIF
vif_data = pd.DataFrame()
vif_data['Feature'] = features
vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print(vif_data)


# Plotting VIF values
plt.figure(figsize=(10, 6))
plt.barh(vif_data['Feature'], vif_data['VIF'], color='royalblue')
plt.xlabel('Variance Inflation Factor (VIF)')
plt.ylabel('Feature')
plt.title('VIF Values for Features')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()


# ##Random Forest to check feature importance

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

# rf_features = {
#     'Feature': ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age', 'hypertension', 'heart_disease'],
#     'Importance': [0.407410, 0.327367, 0.144707, 0.101647, 0.011596, 0.007273]
# }
# rf_df = pd.DataFrame(rf_features)

# plt.figure(figsize=(8, 6))
# sns.barplot(x='Importance', y='Feature', data=rf_df.sort_values(by='Importance', ascending=False), palette="viridis")
# plt.title("Feature Importance (Random Forest)")
# plt.xlabel("Importance Score")
# plt.ylabel("Feature")
# plt.show()


# print(classification_report(y_test, y_pred_rf))
# y_proba = rf.predict_proba(X_test)[:, 1]
# auc = roc_auc_score(y_test, y_proba)
# print("ROC-AUC:", auc)

# fpr, tpr, thresholds = roc_curve(y_test, y_proba)
# plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.show()

## Improving model with smote

# Defining features and target
features = ['age', 'bmi', 'hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level','gender_Male','smoking_history_former','smoking_history_never',]
target = 'diabetes'

X = diabetes_data[features]
y = diabetes_data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train a Random Forest model on SMOTE-resampled data
rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)

# Predict on the test set
y_pred_smote = rf_smote.predict(X_test)

# Evaluate the model
accuracy_smote = accuracy_score(y_test, y_pred_smote)
print("Random Forest Accuracy after SMOTE:", accuracy_smote)

classification_report_smote = classification_report(y_test, y_pred_smote)
print("\nClassification Report after SMOTE:\n", classification_report_smote)

# ##################################################
# #<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

# #%%
# ################
# ## Smart Question 4 ##
# ################


# ##################################################
# #<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#
# %%
