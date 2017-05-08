#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 01:06:25 2017
d
@author: kevin
"""

# Required Python Machine learning Packages
import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.cross_validation import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

adult_df = pd.read_csv('BCall.csv',
                       header = None, delimiter=' *, *', engine='python')

adult_df.columns = ['userid','bookid','rating',
					'titlewords','authorwords',
					'year','publisher','country','age']


adult_df_rev = adult_df

le = preprocessing.LabelEncoder()
country_cat = le.fit_transform(adult_df.country)


adult_df_rev['country_cat'] = country_cat

#drop the old categorical columns from dataframe
dummy_fields = ['country']
adult_df_rev = adult_df_rev.drop(dummy_fields, axis = 1)

adult_df_rev = adult_df_rev.reindex_axis(['userid','bookid','rating',
						'titlewords','authorwords',
						'year','publisher','country_cat','age'], axis= 1)

num_features = ['userid','bookid','rating',
				'titlewords','authorwords',
				'year','publisher','country_cat','age']

features = adult_df_rev.values[:,0:8]
target = adult_df_rev.values[:,8]
features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target, test_size = 0.40, random_state = 10)

clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)

print(accuracy_score(target_test, target_pred, normalize=True))