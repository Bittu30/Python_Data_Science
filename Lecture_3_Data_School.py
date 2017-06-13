# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 19:51:03 2017

@author: Ratan Mishra
"""
#read the data from internet
import pandas as pd
url='https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data' 
col_names=['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima=pd.read_csv(url,header=None,names=col_names)

pima.head()

#define X and y
feature_cols=['pregnant','insulin','bmi','age']
X=pima[feature_cols]
y=pima.label

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)

y_pred_class=logreg.predict(X_test)
#calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred_class))

y_test.value_counts()
y_test.mean()

1-y_test.mean()
#calculate null accuracy(for multiclass problems)
max(y_test.mean(),1-y_test.mean())
from  __future__ import print_function
print('True:',y_test.values[0:25])
print('Pred:',y_pred_class[0:25])


#Confusion matrix
print(metrics.confusion_matrix(y_test,y_pred_class))
confusion=metrics.confusion_matrix(y_test,y_pred_class)
TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,0]
sensetivity=TP/(TP+FN)
print(sensetivity)
specificity=TN/(TN+FP)
print(specificity)


#Adjust the clAssification threshold
#print the first 10 predicted responses
logreg.predict(X_test)[0:10]
#print the first 10 probabilty 
logreg.predict_proba(X_test)[0:10,:]
logreg.predict_proba(X_test)[0:10,1]
y_pred_prob=logreg.predict_proba(X_test)[:,1]
# histogram of predicted probabilities
import matplotlib.pyplot as plt
plt.hist(y_pred_prob, bins=8)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')

from sklearn.preprocessing import binarize
y_pred_class=binarize([y_pred_prob],0.3)[0]
print(confusion)

print(metrics.confusion_matrix(y_test,y_pred_class))

# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test, y_pred_prob))
