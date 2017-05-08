#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 00:55:41 2017

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

adult_df = pd.read_csv('LFsub2all.csv',
                       header = None, delimiter=' *, *', engine='python')

adult_df.columns = ['userid','artistid','plays','numscrobbles',
					'numlistens','rock','electronic','indie',
					'pop','hiphop','gender','age','country','year']


adult_df_rev = adult_df
#print(adult_df_rev)
le = preprocessing.LabelEncoder()

rock_cat = le.fit_transform(adult_df.rock)
electronic_cat = le.fit_transform(adult_df.electronic)
indie_cat = le.fit_transform(adult_df.indie)
pop_cat = le.fit_transform(adult_df.pop)
hiphop_cat = le.fit_transform(adult_df.hiphop)
gender_cat = le.fit_transform(adult_df.gender)
country_cat = le.fit_transform(adult_df.country)

adult_df_rev['rock_cat'] = rock_cat
adult_df_rev['electronic_cat'] = electronic_cat
adult_df_rev['indie_cat'] = indie_cat
adult_df_rev['pop_cat'] = pop_cat
adult_df_rev['hiphop_cat'] = hiphop_cat
adult_df_rev['gender_cat'] = gender_cat
adult_df_rev['country_cat'] = country_cat
print(adult_df_rev)
dummy_fields = ['rock','electronic','indie','pop',
				'hiphop','gender','country']
adult_df_rev = adult_df_rev.drop(dummy_fields, axis = 1)

adult_df_rev = adult_df_rev.reindex_axis(['userid','artistid','plays',
							'numscrobbles','numlistens','rock_cat',
							'electronic_cat','indie_cat','pop_cat',
							'hiphop_cat','gender_cat','age','country_cat',
							'year'], axis= 1)
#print(adult_df_rev)
num_features = ['userid','artistid','plays',
				'numscrobbles','numlistens','rock_cat',
				'electronic_cat','indie_cat','pop_cat',
				'hiphop_cat','gender_cat','age','country_cat',
				'year']


features = adult_df_rev.values[:,0:13]
target = adult_df_rev.values[:,13]
features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target, test_size = 0.40, random_state = 10)

clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)

print(accuracy_score(target_test, target_pred, normalize=True))