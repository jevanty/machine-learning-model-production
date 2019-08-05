# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
df = pd.read_csv('imports_85_data 2.csv')
df = df.drop('Unnamed: 0', axis=1)

mask = df['price'].isna() == False

df = df.loc[mask,:]

print(df.isna().sum())

mask_2 = df.isna().sum()/len(df.index) < 0.15

df = df.loc[:,mask_2]

# dummies
data = df.select_dtypes('object')
feature_to_delete = ['make','num-of-doors','body-style']

#data = data.drop(feature_to_delete,axis=1)
data['num-of-cylinders'] = data['num-of-cylinders'].astype('object')
data = pd.get_dummies(data,drop_first=True)

dummies_to_delete = ['fuel-system_idi','drive-wheels_rwd']
#data = data.drop(dummies_to_delete,axis=1)




X_train, X_test, y_train, y_test = train_test_split(data,df['price'] , test_size=0.25, random_state=0)

# Fit the random forest model to the training data
# Set the feature eliminator to remove 2 features on each step
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=3, step=2, verbose=1)

# Fit the model to the training data
rfe.fit(X_train, y_train)

# Create a mask
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = data.loc[:, mask]
pickle.dump(list(reduced_X.columns),open('dummies.sav','wb'))