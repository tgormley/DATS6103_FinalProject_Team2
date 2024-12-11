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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, roc_auc_score


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
## Clean Data ##
################
# Check for missing values
print(diabetes_data.isnull().sum())

print(diabetes_data.describe())

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


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%
################
## Smart Question 4 ##
################


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#