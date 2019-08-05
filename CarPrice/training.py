# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer
df = pd.read_csv('imports_85_data 2.csv')
import pickle
data = df.drop('Unnamed: 0', axis=1)

mask = data['price'].isna() == False

data = data.loc[mask,:]

print(data.isna().sum())

mask_2 = data.isna().sum()/len(data.index) < 0.15

data = data.loc[:,mask_2]

# numercial data 
numbercal_data = data.select_dtypes('number')
numbercal_data = numbercal_data/numbercal_data.mean()

variance_of_feature = numbercal_data.var()
"""
for i in data.select_dtypes('number').columns :
    plt.scatter(data[i],data['price'])
    plt.title(i)
    plt.show()
"""    
nb_feature_2b_cat = ['symboling']

feature_2_remove = ['peak-rpm','height','bore','stroke','compression-ratio']

feature_2_log = ['curb-weight','horsepower', 'length', 'width','price']

data = data.drop(feature_2_remove, axis=1)

for i in feature_2_log :
    data[i] = np.log( data[i])

data['horsepower'] = data['horsepower'].fillna(4.74493)


# Calculate the correlation matrix and take the absolute value
corr_matrix = data[list(data.select_dtypes('number').columns)].corr().abs()

# Create a True/False mask and apply it
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)

# List column names of highly correlated features (r > 0.95)
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]
print(data[list(data.select_dtypes('number').columns)])
# Drop the features in the to_drop list
data = data.drop(to_drop, axis=1)


#providing categorical mask
number_columns_mask = data.select_dtypes('number').columns
print(number_columns_mask)
data = pd.get_dummies(data)
dummies_mask = pickle.load(open('dummies.sav','rb'))

data = data.loc[:,[*number_columns_mask ,*dummies_mask]] 



X = data.drop('price',axis=1)
X = data.drop('symboling',axis=1)
y = data.loc[:,'price']

est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
Xt = est.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25)


rf = RandomForestRegressor(bootstrap=False,max_depth=93,max_features='sqrt',min_samples_leaf=1,min_samples_split=3, n_estimators=667 )
rf.fit(X_train,y_train)
score = r2_score(y_test,rf.predict(X_test))

plt.plot(y_test,rf.predict(X_test))
plt.show()

"""
parameters = {'bootstrap': [False],
 'max_depth': [ 93],
 'max_features': ['sqrt'],
 'min_samples_leaf': [1],
 'min_samples_split': [3],
 'n_estimators': [667]}
               

grid_search = GridSearchCV(estimator = rf, param_grid = parameters, cv = 3, n_jobs = -1 )
grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_parameters)
"""