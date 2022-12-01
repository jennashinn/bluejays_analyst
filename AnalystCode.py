# General Data Processing
import pandas as pd
from pandas import Series
import numpy as np

# Visualizations
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score

from imblearn.over_sampling import SMOTE

## EDA and Cleaning 

data_raw = pd.read_csv('training.csv')
data_raw.head(5)

data_raw.info()

# Basic NaN code I use for most projects
total = data_raw.isnull().sum().sort_values(ascending=False)
percent_1 = data_raw.isnull().sum()/data_raw.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data = missing_data.loc[missing_data['Total'] > 0]
print(missing_data)

print('Missing Values', missing_data['%'].sum(), '%')
                   
# SpinRate       6
# Missing Values 0.1 %          

data = data_raw.dropna()
data['InPlay'].value_counts()/len(data['InPlay']) # Possibly imbalanced

data.describe()
sns.heatmap(data.corr(), annot=True, cmap = "BuPu")

## First Model 
features = ['Velo', 'SpinRate', 'HorzBreak', 'InducedVertBreak']

X = data[features]
y = data['InPlay']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=1, stratify=y)

dt = DecisionTreeClassifier()
dt = dt.fit(X_train,y_train)
dt_preds = dt.predict(X_test)

acc_dt = accuracy_score(y_test, dt_preds)
print("Accuracy:", acc_dt.round(2))  # 0.61

prec_dt = precision_score(y_test, dt_preds)
print("Precision:", prec_dt.round(2)) # 0.28 

recall_dt = precision_score(y_test, dt_preds)
print("Recall:", recall_dt.round(2)) # 0.28

report_dt = classification_report(y_test, dt_preds)
print(report_dt)

cm_dt = confusion_matrix(y_test, dt_preds)

dt.fit(data[features], data['InPlay'])
importances_dt = Series(dt.feature_importances_, features).sort_values(ascending=False)

###

rf = RandomForestClassifier()
rf = rf.fit(X_train,y_train)
rf_preds = rf.predict(X_test)

acc_rf = accuracy_score(y_test, rf_preds)
print("Accuracy:", acc_rf.round(2)) # 0.7

prec_rf = precision_score(y_test, rf_preds)
print("Precision:", prec_rf.round(2)) # 0.28

recall_rf = recall_score(y_test, rf_preds)
print('Recall:', recall_rf.round(2)) # 0.06

report_rf = classification_report(y_test, rf_preds)
print(report_rf)


######################################################
### PreProcess
data_raw = pd.read_csv('training.csv')
df = data_raw.dropna()

# removing outliers
def find_outliers(df, variable):
    #Takes two parameters: dataframe & variable of interest as string
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3-q1
    inner_fence = 1.5*iqr
    outer_fence = 3*iqr
    
    #inner fence lower and upper end
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence
    
    #outer fence lower and upper end
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence
    
    outliers_prob = []
    for index, x in enumerate(df[variable]):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(index)
            
    outliers_poss = []
    for index, x in enumerate(df[variable]):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(index)
            
    return outliers_prob, outliers_poss

prob_velo, poss_velo = find_outliers(df, 'Velo')
prob_spin, poss_spin = find_outliers(df, 'SpinRate')
prob_hbreak, poss_hbreak = find_outliers(df, 'HorzBreak')
prob_vbreak, poss_vbreak = find_outliers(df, 'InducedVertBreak')

prob_outs = prob_velo + prob_spin + prob_hbreak + prob_vbreak
poss_outs = poss_velo + poss_spin + poss_hbreak + poss_vbreak

outliers = prob_outs + poss_outs

data = df.loc[~df.index.isin(outliers)]
data.isnull().sum()

X = data.drop('InPlay', axis=1)
y = data['InPlay']

from sklearn.preprocessing import MinMaxScaler
to_scale = [col for col in df.columns if df[col].max() > 1]
mms = MinMaxScaler()
scaled = mms.fit_transform(data[to_scale])
scaled = pd.DataFrame(scaled, columns=to_scale)
for col in scaled:
    data[col] = scaled[col]
scaled

#scaled.to_csv('preprocessed_training.csv')

features = ['Velo', 'SpinRate', 'HorzBreak', 'InducedVertBreak']
X = scaled[features]
y = data['InPlay']

sm = SMOTE()
X_sm, y_sm = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.25, random_state=42)

rf = RandomForestClassifier()
rf = rf.fit(X_train,y_train)
rf_preds = rf.predict(X_test)

acc_rf = accuracy_score(y_test, rf_preds)
print("Accuracy:", acc_rf.round(2)) # 0.75

prec_rf = precision_score(y_test, rf_preds)
print("Precision:", prec_rf.round(2)) # 0.73

recall_rf = recall_score(y_test, rf_preds)
print('Recall:', recall_rf.round(2)) # 0.78

report_rf = classification_report(y_test, rf_preds)
print(report_rf)

cm = confusion_matrix(y_test, rf_preds)
cmnorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
sns.heatmap(cmnorm, annot=True, fmt='.2%', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

import pickle
filename = 'trained_model.sav'
pickle.dump(rf, open(filename, 'wb'))

############################################
#### Applying Model 

df = pd.read_csv('deploy.csv')
df.head()
df.info()
data = df.dropna()

from sklearn.preprocessing import MinMaxScaler
to_scale = data.columns
mms = MinMaxScaler()
scaled = mms.fit_transform(data[to_scale])
scaled = pd.DataFrame(scaled, columns=to_scale)
for col in scaled:
    data[col] = scaled[col]
scaled.head()


import pickle
model = pickle.load(open('trained_rf_model.pkl', 'rb'))
predictions = model.predict(scaled)
scaled['InPlay'] = predictions
scaled['InPlay'].value_counts() 
# 0 - 7859  1 - 2128
scaled.to_csv('deploy_model.csv')