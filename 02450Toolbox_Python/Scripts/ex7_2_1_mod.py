from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import sklearn.tree
import numpy as np, scipy.stats as st

# requires data from exercise 1.5.1
from ex5_1_5 import *

# This script creates linear regression and regression tree 
# classifiers using 10-fold cross-validation
X,y = X[:,:10], X[:,10:]

# cross validation method
K = 10
CV = model_selection.KFold(n_splits=K)

# training
yhat=[]
y_true = []
for train_index, test_index in CV.split(X):
    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]
    
    mA = sklearn.linear_model.LinearRegression().fit(X_train,y_train)
    mB = sklearn.tree.DecisionTreeRegressor().fit(X_train, y_train)

    yhatA = mA.predict(X_test)
    yhatB = mB.predict(X_test)[:,np.newaxis]  #  justsklearnthings

    y_true.append(y_test)
    yhat.append(np.concatenate([yhatA, yhatB], axis=1))

#### perform statistical comparison of the models ####
# compute z with squared error.
zA = np.abs(y_test - yhatA ) ** 2
zB = np.abs(y_test - yhatB ) ** 2

# compute confidence interval of model A and B
alpha = 0.05
CI_A = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval
CI_B = st.t.interval(1-alpha, df=len(zB)-1, loc=np.mean(zB), scale=st.sem(zB))  # Confidence interval
print("CI_A: ", CI_A)
print("CI_B: ", CI_B)

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf(-np.abs( np.mean(z))/st.sem(z), df=len(z)-1)  # p-value

print("CI: ", CI)
print("p: ", p)

'''
Output
CI_A:  [0.15524047], [0.20379836]
CI_B:  [0.36514537], [0.49783908]
CI:  [-0.31317529], [-0.19077032]
p:  [3.20361213e-15]
'''
# Conclusion
# Model B (regression tree) has a better performance than
# model A (linear regression). CI are both clear of 0
# p-value is low, thus result is unlikely due to chance
