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
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report



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

# Preprocessing: Handle missing values
diabetes_data['gender'] = diabetes_data['gender'].map({'Male': 1, 'Female': 0})
diabetes_data['gender'].fillna(diabetes_data['gender'].mode()[0], inplace=True)

# One-hot encoding for categorical variables
diabetes_data = pd.get_dummies(diabetes_data, columns=['smoking_history'], drop_first=True)

print(diabetes_data.head())


#%%
################
## Smart Question 1 ##
################

# Compute the correlation matrix
diabetes_data.head()
correlation_matrix = diabetes_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Define features and target
features = ['age', 'bmi', 'hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level','gender','smoking_history_current','smoking_history_ever','smoking_history_former','smoking_history_never','smoking_history_not current']
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
plt.gca().invert_yaxis() 
plt.show()


# ##Random Forest to check feature importance

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)


# Feature importance
importances = rf.feature_importances_
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
print(feature_importances.sort_values(by='Importance', ascending=False))

sorted_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(sorted_importances['Feature'], sorted_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.grid(axis='x')
plt.show()

## training random forest model with smote for diabetes prediction

# Defining features and target
features = ['age', 'bmi', 'hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level','gender','smoking_history_former','smoking_history_never',]
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

classification_report_smote = classification_report(y_test, y_pred_smote)
print("\nClassification Report with SMOTE:\n", classification_report_smote)
accuracy_smote = accuracy_score(y_test, y_pred_smote)
print("Random Forest Accuracy with SMOTE:", accuracy_smote)
precision_smote = precision_score(y_test, y_pred_smote)
print(f"Precision with SMOTE: {precision_smote:.2f}")
recall_smote = recall_score(y_test, y_pred_smote)
print(f"Recall with SMOTE: {recall_smote:.2f}")

y_proba = rf_smote.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", auc)

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#


#%%
################
## Smart Question 2 ##
################
def reverse_one_hot(df):
    # Reverse 'gender' one-hot encoding
    if 'gender_Male' in df.columns:
        df['gender'] = df['gender_Male'].apply(lambda x: 'Male' if x == 1 else 'Female')
        df.drop(columns=['gender_Male'], inplace=True)

    # Reverse 'smoking_history' one-hot encoding
    smoking_columns = [col for col in df.columns if col.startswith('smoking_history_')]
    if smoking_columns:
        df['smoking_history'] = df[smoking_columns].idxmax(axis=1).str.replace('smoking_history_', '')
        df.drop(columns=smoking_columns, inplace=True)
    
    return df

#%%
# Reverse one-hot encoding and assign to 'df'
df = reverse_one_hot(diabetes_data)
print(df.head())

#%%
# Change binary 0 and 1 data to categorical 'no' and 'yes'
df['diabetes'] = df['diabetes'].replace({0:'no', 1:'yes'})

#%%
#Creating numerical columns
num_cols=['age','blood_glucose_level']
#Creating categorical variables
cat_cols= ['gender','smoking_history','diabetes']

#%%
df['diabetes'] = df['diabetes'].replace({'no': 0, 'yes': 1}).astype(int)



#%%
# Calculate diabetes prevalence by gender
gender_diabetes_prevalence = df.groupby('gender')['diabetes'].value_counts(normalize=True).unstack()

# Plot the data directly
gender_diabetes_prevalence.plot(kind='bar', stacked=True, figsize=(8, 6), color=['blue', 'orange'])
plt.title('Diabetes Prevalence by Gender')
plt.ylabel('Proportion')
plt.xlabel('Gender')
plt.legend(['No Diabetes', 'Diabetes'], title='Diabetes Status')
plt.tight_layout()
plt.show()

#%%
df['age_group'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 80], labels=['Young', 'Young-adult', 'Adult', 'Elderly'])

# Group by age group and calculate diabetes prevalence
age_group_prevalence = df.groupby('age_group')['diabetes'].mean()
print("Diabetes prevalence by age group:\n", age_group_prevalence)

# Visualize the prevalence
sns.barplot(x=age_group_prevalence.index, y=age_group_prevalence.values)
plt.title('Diabetes Prevalence by Age Group')
plt.ylabel('Prevalence')
plt.show()



#%%
# Inspect unique values in the smoking_history column
print("Unique Smoking History Categories:")
print(df['smoking_history'].unique())

# Calculate diabetes prevalence by smoking history
smoking_prevalence = df.groupby('smoking_history')['diabetes'].mean()

# Plot the data
plt.figure(figsize=(10, 6))
sns.barplot(x=smoking_prevalence.index, y=smoking_prevalence.values, palette='viridis')
plt.title('Diabetes Prevalence by Smoking History')
plt.ylabel('Diabetes Prevalence')
plt.xlabel('Smoking History')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#%%
# Encode smoking_history using label encoding
le = LabelEncoder()
df['smoking_history_encoded'] = le.fit_transform(df['smoking_history'])

# Create a binary indicator for high blood glucose levels
df['high_glucose'] = (df['blood_glucose_level'] > 140).astype(int)

# Select features and target
X = df[['smoking_history_encoded', 'high_glucose']]
y = df['diabetes']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%
# Train logistic regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Evaluate logistic regression
log_preds = log_reg.predict(X_test)
log_probs = log_reg.predict_proba(X_test)[:, 1]
print("Logistic Regression Report:")
print(classification_report(y_test, log_preds))
print("ROC AUC Score:", roc_auc_score(y_test, log_probs))

# Display coefficients
coefficients = pd.DataFrame({
    'Feature': ['Smoking History', 'High Blood Glucose'],
    'Coefficient': log_reg.coef_[0]
})
print("Logistic Regression Coefficients:")
print(coefficients)



#%%
# Train random forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Evaluate random forest
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]
print("Random Forest Report:")
print(classification_report(y_test, rf_preds))
print("ROC AUC Score:", roc_auc_score(y_test, rf_probs))

# Feature importance
importances = rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': ['Smoking History', 'High Blood Glucose'],
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print("Random Forest Feature Importance:")
print(importance_df)



#%%
# Create glucose categories
df['glucose_category'] = pd.cut(df['blood_glucose_level'], bins=[0, 100, 140, 300], labels=['Low', 'Normal', 'High'])

# Group by smoking history and glucose category
smoking_glucose_prevalence = df.groupby(['smoking_history', 'glucose_category'])['diabetes'].mean().unstack()

# Heatmap of diabetes prevalence
plt.figure(figsize=(10, 6))
sns.heatmap(smoking_glucose_prevalence, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Diabetes Prevalence by Smoking History and Glucose Level')
plt.ylabel('Smoking History')
plt.xlabel('Glucose Category')
plt.show()


#%%
# Boxplot of age distribution by smoking history
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='smoking_history', y='age', palette='viridis')
plt.title('Age Distribution by Smoking History')
plt.xlabel('Smoking History')
plt.ylabel('Age')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




#%%
# Balance the dataset (oversampling the minority class)
minority_class = df[df['diabetes'] == 1]
majority_class = df[df['diabetes'] == 0]

oversampled_minority = minority_class.sample(len(majority_class), replace=True, random_state=42)
balanced_df = pd.concat([majority_class, oversampled_minority])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and target
X_balanced = balanced_df[['smoking_history_encoded', 'high_glucose', 'age']]
y_balanced = balanced_df['diabetes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)


#%%
# Logistic regression with balanced dataset
log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_reg.fit(X_train, y_train)

# Evaluate logistic regression
log_preds = log_reg.predict(X_test)
log_probs = log_reg.predict_proba(X_test)[:, 1]

print("Logistic Regression Report (with Age):")
print(classification_report(y_test, log_preds))
print("ROC AUC Score (with Age):", roc_auc_score(y_test, log_probs))

# Coefficients
coefficients = pd.DataFrame({
    'Feature': ['Smoking History', 'High Blood Glucose', 'Age'],
    'Coefficient': log_reg.coef_[0]
})
print("Logistic Regression Coefficients:")
print(coefficients)

#%%
# Random Forest with balanced dataset
rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=7)
rf.fit(X_train, y_train)

# Evaluate random forest
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("Random Forest Report:")
print(classification_report(y_test, rf_preds))
print("ROC AUC Score:", roc_auc_score(y_test, rf_probs))

# Feature Importance
importance = pd.DataFrame({
    'Feature': ['Smoking History', 'High Blood Glucose', 'Age'],
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Random Forest Feature Importance:")
print(importance)




#%%
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
# Get probabilities from the random forest model
rf_probs = rf.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
roc_auc = roc_auc_score(y_test, rf_probs)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line (random classifier)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#%%
diabetes_data = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

print("Dataframe with one-hot encoding reapplied (diabetes_data):")
print(diabetes_data.head())
# %%


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%
################
## Smart Question 3 ##
################

# Preprocessing: Handle missing values
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data['gender'].fillna(data['gender'].mode()[0], inplace=True)
data['smoking_history'] = data['smoking_history'].astype('category').cat.codes

# Features and target
X = data.drop(columns=['diabetes'])
y = data['diabetes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nAUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.2f}")

# Partial Dependence Plots for 'bmi' and 'blood_glucose_level'
PartialDependenceDisplay.from_estimator(
    model, X_test, ['bmi', 'blood_glucose_level'], kind='average', grid_resolution=50
)
plt.show()

# Statistical Summaries
diabetes_yes = data[data['diabetes'] == 1]
diabetes_no = data[data['diabetes'] == 0]

bmi_yes_mean = diabetes_yes['bmi'].mean()
bmi_no_mean = diabetes_no['bmi'].mean()
hba1c_yes_mean = diabetes_yes['HbA1c_level'].mean()
hba1c_no_mean = diabetes_no['HbA1c_level'].mean()

print("Statistical Summary:")
print(f"BMI Mean (Diabetes: Yes) = {bmi_yes_mean:.2f}")
print(f"BMI Mean (Diabetes: No) = {bmi_no_mean:.2f}")
print(f"HbA1c Mean (Diabetes: Yes) = {hba1c_yes_mean:.2f}")
print(f"HbA1c Mean (Diabetes: No) = {hba1c_no_mean:.2f}")

# Threshold Analysis
avg_glucose_by_diabetes = data.groupby('diabetes')['blood_glucose_level'].mean()
avg_bmi_by_diabetes = data.groupby('diabetes')['bmi'].mean()

threshold_glucose = avg_glucose_by_diabetes[1] * 0.9  # 90% of diabetic group mean
threshold_bmi = avg_bmi_by_diabetes[1] * 0.9          # 90% of diabetic group mean

print(f"Suggested Blood Glucose Threshold: {threshold_glucose:.2f}")
print(f"Suggested BMI Threshold: {threshold_bmi:.2f}")

# Visualizations
plt.figure(figsize=(14, 6))

# BMI Distribution
plt.subplot(1, 2, 1)
sns.kdeplot(diabetes_yes['bmi'], label='Diabetes: Yes', fill=True, color='red')
sns.kdeplot(diabetes_no['bmi'], label='Diabetes: No', fill=True, color='blue')
plt.title('BMI Distribution by Diabetes Status')
plt.xlabel('BMI')
plt.ylabel('Density')
plt.legend()

# HbA1c Level Distribution
plt.subplot(1, 2, 2)
sns.kdeplot(diabetes_yes['HbA1c_level'], label='Diabetes: Yes', fill=True, color='red')
sns.kdeplot(diabetes_no['HbA1c_level'], label='Diabetes: No', fill=True, color='blue')
plt.title('HbA1c Level Distribution by Diabetes Status')
plt.xlabel('HbA1c Level')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()


# ##################################################
# #<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#
# %%
#%%
################
## Smart Question 4 ##
################

# Preprocessing: Handle missing values
# diabetes_data['smoking_history'] = diabetes_data['smoking_history'].astype('category').cat.codes

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


# ##################################################
# #<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#
# %%
