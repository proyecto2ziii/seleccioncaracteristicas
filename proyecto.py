import csv
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

#read in the data
data = pd.read_csv('data_so.csv', header=None)
X = data.iloc[:,0:18]
y = data.iloc[:,19]

depth = 5
maxFeat = 3 

result = cross_val_score(ensemble.RandomForestClassifier(n_estimators=1000, max_depth=depth, max_features=maxFeat, oob_score=False), X, y, scoring='roc_auc', cv=2)

print(result)
# result is now something like array([ 0.66773295,  0.58824739])

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50)

RFModel = ensemble.RandomForestClassifier(n_estimators=1000, max_depth=depth, max_features=maxFeat, oob_score=False)
RFModel.fit(xtrain,ytrain)
prediction = RFModel.predict_proba(xtest)
auc = roc_auc_score(ytest, prediction[:,1:2])
print (auc)    #something like 0.83

RFModel.fit(xtest,ytest)
prediction = RFModel.predict_proba(xtrain)
auc = roc_auc_score(ytrain, prediction[:,1:2])
print (auc)    #also something like 0.83
