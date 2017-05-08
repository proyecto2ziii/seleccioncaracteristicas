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

adult_df = pd.read_csv('casoPruebaMovieLens/MLall.csv',
                       header = None, delimiter=' *, *', engine='python')

adult_df.columns = ['movieid','userid',
                    'gender','age','occupation','zipcode',
                    'namewords','namepar','year','action',
                    'adventure','animation','childrens',
                    'comedy','crime','documentary','drama',
                    'fantasy','filmnoir','horror','musical',
                    'mystery','romance',
                    'scifi','thriller','war','western','rating']

array_registros_existentes = [0]

for registro in adult_df['movieid']:
   cambio = 0
   print("reg "+str(registro))
   for existe in array_registros_existentes:
       print("exi "+str(existe))
       if existe == registro:
           #print(str(registro)+" - "+str(existe))
           print(len(array_registros_existentes))
           cambio = 1
           
        
   if cambio == 1:
       array_registros_existentes.append(registro)
      
print(len(array_registros_existentes))
         

    


