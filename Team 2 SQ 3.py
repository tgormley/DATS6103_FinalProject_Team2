# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Features and target
X = data.drop(columns=['diabetes'])  
y = data['diabetes']  

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nAUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.2f}")

# Split data into groups: with and without diabetes
diabetes_yes = data[data['diabetes'] == 1]
diabetes_no = data[data['diabetes'] == 0]

# Statistical Summaries
bmi_yes_mean = diabetes_yes['bmi'].mean()
bmi_no_mean = diabetes_no['bmi'].mean()
hba1c_yes_mean = diabetes_yes['HbA1c_level'].mean()
hba1c_no_mean = diabetes_no['HbA1c_level'].mean()

print("Statistical Summary:")
print(f"BMI Mean (Diabetes: Yes) = {bmi_yes_mean:.2f}")
print(f"BMI Mean (Diabetes: No) = {bmi_no_mean:.2f}")
print(f"HbA1c Mean (Diabetes: Yes) = {hba1c_yes_mean:.2f}")
print(f"HbA1c Mean (Diabetes: No) = {hba1c_no_mean:.2f}")

# T-Test for Statistical Significance
bmi_tstat, bmi_pval = ttest_ind(diabetes_yes['bmi'], diabetes_no['bmi'])
hba1c_tstat, hba1c_pval = ttest_ind(diabetes_yes['HbA1c_level'], diabetes_no['HbA1c_level'])

print("\nT-Test Results:")
print(f"BMI: t-stat = {bmi_tstat:.2f}, p-value = {bmi_pval:.4f}")
print(f"HbA1c: t-stat = {hba1c_tstat:.2f}, p-value = {hba1c_pval:.4f}")

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


# Boxplots
plt.figure(figsize=(14, 6))

# BMI Boxplot
plt.subplot(1, 2, 1)
sns.boxplot(x='diabetes', y='bmi', data=data, hue='diabetes', palette='Set2', dodge=False, legend=False)
plt.title('BMI by Diabetes Status')
plt.xlabel('Diabetes')
plt.ylabel('BMI')
plt.xticks([0, 1], ['No', 'Yes'])
plt.legend([], [], frameon=False)  

# HbA1c Level Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x='diabetes', y='HbA1c_level', data=data, hue='diabetes', palette='Set2', dodge=False, legend=False)
plt.title('HbA1c Level by Diabetes Status')
plt.xlabel('Diabetes')
plt.ylabel('HbA1c Level')
plt.xticks([0, 1], ['No', 'Yes'])
plt.legend([], [], frameon=False)  

plt.tight_layout()
plt.show()
