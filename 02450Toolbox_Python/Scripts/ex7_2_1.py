from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import sklearn.tree
import numpy as np, scipy.stats as st

# requires data from exercise 1.5.1
from ex5_1_5 import *

# This script creates linear regression and regression tree 
# classifiers using hold-out cross-validation
X,y = X[:,:10], X[:,10:]

# cross validation method
test_proportion = 0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)

# training
mA = sklearn.linear_model.LinearRegression().fit(X_train,y_train)
mB = sklearn.tree.DecisionTreeRegressor().fit(X_train, y_train)

yhatA = mA.predict(X_test)
yhatB = mB.predict(X_test)[:,np.newaxis]  #  justsklearnthings

#### perform statistical comparison of the models ####
# compute z with squared error.
zA = np.abs(y_test - yhatA ) ** 2
zB = np.abs(y_test - yhatB ) ** 2

# compute confidence interval of model A
alpha = 0.05
CI_A = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

# compute confidence interval of model B
CI_B = st.t.interval(1-alpha, df=len(zB)-1, loc=np.mean(zB), scale=st.sem(zB))  # Confidence interval

print("CI_A: ", CI_A)
print("CI_B: ", CI_B)

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print("CI: ", CI)
print("p: ", p)

'''
Output
CI_A:  [0.23943442],[0.29614625]
CI_B:  [0.29875395], [0.3998156]
CI:  [-0.1334764]), [-0.02951248]
p:  [0.00214547]
'''
# Conclusion
# Model B (regression tree) has a better performance than
# model A (linear regression). CI are both clear of 0
# p-value is low, thus result is unlikely due to chance
