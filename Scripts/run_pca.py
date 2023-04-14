# Use data structure from load_data.py
from load_data import *
from os import path


from scipy.linalg import svd
import matplotlib.pyplot as plt

temp = df.columns.get_loc("temp")
X = X[:, temp:]

# PCA by computing SVD
U, S, V = svd(X, full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

# Project data onto principal component space
Z = X @ V

# from ex2_1_3.py

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho)+1), rho, 'x-')
plt.plot(range(1, len(rho)+1), np.cumsum(rho), 'o-')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual', 'Cumulative'])
plt.grid()
plt.show()

# Plot scatter diagram
plt.figure()
plt.title('Bike data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(X[class_mask, i], X[class_mask, j], 'o', alpha=.3)

# plt.legend(attributeNames)
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])
plt.show()

# Show successful code execution
print("Finished running", path.basename(__file__))
