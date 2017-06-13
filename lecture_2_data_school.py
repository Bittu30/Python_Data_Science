# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 19:20:15 2017

@author: Ratan Mishra
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data
y=iris.target

knn=KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,X,y,cv=10,scoring="accuracy")
print(scores)
print(scores.mean())
#search for the optimal value for k
k_range=list(range(1,31))
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,X,y,cv=10,scoring="accuracy")
    k_scores.append(scores.mean())
    
print(k_scores)

#Plot the data
plt.plot(k_range,k_scores)
plt.xlabel("value of K for knn")
plt.ylabel("cross validated accuracy")

#More effiecint tuning in the Grid Search CV
from sklearn.grid_search import GridSearchCV
k_range=list(range(1,31))
print(k_range)
param_grid=dict(n_neighbors=k_range)
print(param_grid)

#instantiate the grid
grid=GridSearchCV(knn,param_grid,cv=10,scoring="accuracy")
grid.fit(X,y)
#view the grid with data
grid.grid_scores_
#examine the best model
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)

#Searching the multiple parameters simultaneously
k_range=list(range(1,31))
weight_options=['uniform','distance']
#create the paramter grid
param_grid=dict(n_neighbors=k_range,weights=weight_options)
print(param_grid)

#instiate the grid
grid=GridSearchCV(knn,param_grid,cv=10,scoring="accuracy")

grid.fit(X,y)

#examine the best model
print(grid.best_score_)
print(grid.best_params_)

#Reducing the computional expense using RandomizedSearchCv
from sklearn.grid_search import RandomizedSearchCV
param_dist=dict(n_neighbors=k_range,weights=weight_options)
#n_iters control the searches
rand=RandomizedSearchCV(knn,param_dist,cv=10,scoring="accuracy",n_iter=10,random_state=5)
rand.fit(X,y)

print(rand.best_score_)
print(rand.best_params_)
