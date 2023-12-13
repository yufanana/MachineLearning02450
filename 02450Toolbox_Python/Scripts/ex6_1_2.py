# exercise 6.1.2

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from scipy.io import loadmat
from sklearn import model_selection, tree
import numpy as np

# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/wine2.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

# K-fold crossvalidation
K = 100
CV = model_selection.KFold(n_splits=K, shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc), K))        # 19 by 10
Error_test = np.empty((len(tc), K))

k = 0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1, K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index, :], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train, y_train.ravel())
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)

        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = np.sum(
            y_est_test != y_test) / float(len(y_est_test))
        misclass_rate_train = np.sum(
            y_est_train != y_train) / float(len(y_est_train))
        Error_test[i, k], Error_train[i,
                                      k] = misclass_rate_test, misclass_rate_train
    k += 1


f = figure()
# Error_test: row - tree complexity, col - test error
# Error_test.T: row - test error, col - tree complexity
boxplot(Error_test.T)       # box plot of test error, for each tree complexity
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))

f = figure()
# take mean of errors, plot for each tree complexity
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train', 'Error_test'])
show()

print('Ran Exercise 6.1.2')