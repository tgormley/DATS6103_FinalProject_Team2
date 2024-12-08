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
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset
diabetes_data = pd.read_csv('diabetes_prediction_dataset.csv')

# Inspect the dataset
print(diabetes_data.info())
print(diabetes_data.head())

# Check for missing values
print(diabetes_data.isnull().sum())

print(diabetes_data.describe())

# One-hot encoding for categorical variables
diabetes_data = pd.get_dummies(diabetes_data, columns=['gender', 'smoking_history'], drop_first=True)

# Compute the correlation matrix
correlation_matrix = diabetes_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

print(diabetes_data.head())
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


# Plot VIF values
plt.figure(figsize=(10, 6))
plt.barh(vif_data['Feature'], vif_data['VIF'], color='royalblue')
plt.xlabel('Variance Inflation Factor (VIF)')
plt.ylabel('Feature')
plt.title('VIF Values for Features')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()


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


# smote ----------

# Import necessary libraries
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score



# Define features and target
features = ['age', 'bmi', 'hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level']
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

#-----------

# # Import necessary libraries
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, roc_curve, roc_auc_score
# from sklearn.model_selection import GridSearchCV
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np



# # Define features and target
# features = ['age', 'bmi', 'hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level']
# target = 'diabetes'

# X = diabetes_data[features]
# y = diabetes_data[target]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Apply SMOTE to oversample the minority class
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Step 1: Threshold Adjustment
# rf_smote = RandomForestClassifier(random_state=42)
# rf_smote.fit(X_train_smote, y_train_smote)

# # Predict probabilities
# y_proba_smote = rf_smote.predict_proba(X_test)[:, 1]

# # Calculate ROC Curve
# fpr, tpr, thresholds = roc_curve(y_test, y_proba_smote)
# plt.figure(figsize=(10, 6))
# plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba_smote):.2f})")
# plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.grid()
# plt.show()

# # Optimal threshold
# optimal_idx = np.argmax(tpr - fpr)
# optimal_threshold = thresholds[optimal_idx]
# print(f"Optimal Threshold: {optimal_threshold}")

# # Adjust predictions
# y_pred_optimal = (y_proba_smote >= optimal_threshold).astype(int)
# print("\nClassification Report with Optimal Threshold:\n", classification_report(y_test, y_pred_optimal))

# # Step 2: Hyperparameter Tuning
# param_grid = {
#     'n_estimators': [100, 150],  # Reduce the number of estimators
#     'max_depth': [None, 10],    # Focus on key depths
#     'min_samples_split': [2, 5], 
#     'min_samples_leaf': [1, 2]
# }


# from sklearn.model_selection import RandomizedSearchCV

# # Define a RandomizedSearchCV
# random_search = RandomizedSearchCV(
#     estimator=RandomForestClassifier(random_state=42),
#     param_distributions=param_grid,
#     n_iter=10,  # Number of parameter settings sampled
#     scoring='f1',
#     cv=3,
#     verbose=1,
#     n_jobs=-1,
#     random_state=42
# )

# # Fit RandomizedSearchCV
# random_search.fit(X_train_smote, y_train_smote)

# # Best parameters and model
# best_params = random_search.best_params_
# best_model = random_search.best_estimator_

# # Evaluate best model
# y_pred_best_model = best_model.predict(X_test)
# print("\nBest Model Parameters:", best_params)
# print("\nClassification Report with Best Model:\n", classification_report(y_test, y_pred_best_model))
