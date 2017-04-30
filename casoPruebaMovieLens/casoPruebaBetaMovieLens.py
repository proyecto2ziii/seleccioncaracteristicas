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

adult_df = pd.read_csv('movieLensPrueba.csv',
                       header = None, delimiter=' *, *', engine='python')

adult_df.columns = ['movieid','userid','rating',
                    'gender','age','occupation','zipcode',
                    'namewords','namepar','year','action',
                    'adventure','animation','childrens',
                    'comedy','crime','documentary','drama',
                    'fantasy','filmnoir','horror','musical',
                    'mystery','romance',
                    'scifi','thriller','war','western']

#print(adult_df)

#print(adult_df.isnull().sum())


adult_df_rev = adult_df
#print(adult_df_rev.describe(include= 'all'))

for value in ['gender','namepar','action','adventure','animation',
              'childrens','comedy','crime',
              'documentary','drama','fantasy',
              'filmnoir','horror','musical',
              'mystery','romance','scifi',
              'thriller','war','western']:
    adult_df_rev[value].replace(['?'], [adult_df_rev.describe(include='all')[value][2]],
                                inplace='True')

adult_df_rev = adult_df

le = preprocessing.LabelEncoder()
gender_cat = le.fit_transform(adult_df.gender)
namepar_cat = le.fit_transform(adult_df.namepar)
action_cat = le.fit_transform(adult_df.action)
adventure_cat = le.fit_transform(adult_df.adventure)
animation_cat = le.fit_transform(adult_df.animation)
childrens_cat = le.fit_transform(adult_df.childrens)
comedy_cat = le.fit_transform(adult_df.comedy)
crime_cat = le.fit_transform(adult_df.crime)
documentary_cat = le.fit_transform(adult_df.documentary)
drama_cat = le.fit_transform(adult_df.drama)
fantasy_cat = le.fit_transform(adult_df.fantasy)
filmnoir_cat = le.fit_transform(adult_df.filmnoir)
horror_cat = le.fit_transform(adult_df.horror)
musical_cat = le.fit_transform(adult_df.musical)
mystery_cat = le.fit_transform(adult_df.mystery)
romance_cat = le.fit_transform(adult_df.romance)
scifi_cat = le.fit_transform(adult_df.scifi)
thriller_cat = le.fit_transform(adult_df.thriller)
war_cat = le.fit_transform(adult_df.war)
western_cat = le.fit_transform(adult_df.western)
#initialize the encoded categorical columns

adult_df_rev['gender_cat'] = gender_cat
adult_df_rev['namepar_cat'] = namepar_cat
adult_df_rev['action_cat'] = action_cat
adult_df_rev['adventure_cat'] = adventure_cat
adult_df_rev['animation_cat'] = animation_cat
adult_df_rev['childrens_cat'] = childrens_cat
adult_df_rev['comedy_cat'] = comedy_cat
adult_df_rev['crime_cat'] = crime_cat
adult_df_rev['documentary_cat'] = documentary_cat
adult_df_rev['drama_cat'] = drama_cat
adult_df_rev['fantasy_cat'] = fantasy_cat
adult_df_rev['filmnoir_cat'] = filmnoir_cat
adult_df_rev['horror_cat'] = horror_cat
adult_df_rev['musical_cat'] = musical_cat
adult_df_rev['mystery_cat'] = mystery_cat
adult_df_rev['romance_cat'] = romance_cat
adult_df_rev['scifi_cat'] = scifi_cat
adult_df_rev['thriller_cat'] = thriller_cat
adult_df_rev['war_cat'] = war_cat
adult_df_rev['western_cat'] = western_cat
            
#print(adult_df_rev)
            
#drop the old categorical columns from dataframe
dummy_fields = ['gender','namepar','action','adventure','animation',
              'childrens','comedy','crime',
              'documentary','drama','fantasy',
              'filmnoir','horror','musical',
              'mystery','romance','scifi',
              'thriller','war','western']
adult_df_rev = adult_df_rev.drop(dummy_fields, axis = 1)

#print(adult_df_rev)
adult_df_rev = adult_df_rev.reindex_axis(['movieid','userid','rating',
                    'gender_cat','age','occupation','zipcode',
                    'namewords','namepar_cat','year','action_cat',
                    'adventure_cat','animation_cat','childrens_cat',
                    'comedy_cat','crime_cat','documentary_cat','drama_cat',
                    'fantasy_cat','filmnoir_cat','horror_cat','musical_cat',
                    'mystery_cat','romance_cat',
                    'scifi_cat','thriller_cat','war_cat','western_cat'], axis= 1)
#print(adult_df_rev)

            
features = adult_df_rev.values[:,0:27]
target = adult_df_rev.values[:,27]
features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target, test_size = 0.33, random_state = 10)

x_scalar = StandardScaler()
features_train = x_scalar.fit_transform(features_train)
features_test = x_scalar.transform(features_test)

clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)

print(target_pred)
print(accuracy_score(target_test, target_pred, normalize = True))