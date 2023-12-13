from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection, tree

# requires data from exercise 1.5.1
from ex1_5_1 import *

# This script crates predictions from 1-KNN classifiers and decision tree using cross-validation

CV = model_selection.LeaveOneOut()
i = 0

# store predictions.
yhat = []
y_true = []
for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit decision tree classifier, Gini split criterion
    dtc = tree.DecisionTreeClassifier()
    dtc = dtc.fit(X_train, y_train)
    
    dy = []
    y_est = dtc.predict(X_test)
    dy.append(y_est)

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    knclassifier = KNeighborsClassifier(n_neighbors=1)
    knclassifier.fit(X_train, y_train)
    y_est = knclassifier.predict(X_test)

    dy.append(y_est)
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)

    i+=1

yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)
yhat[:,0] # predictions made by first classifier.

# Compute accuracy here.
accuracy = [np.sum(yhat[:,i]==y_true) for i in range(yhat.shape[1])]
print(accuracy)
# output is [143, 144]
