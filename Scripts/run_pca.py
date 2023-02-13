# Use data structure from load_data.py
from load_data import *


from scipy.io import loadmat
from scipy.linalg import svd
import matplotlib.pyplot as plt

# PCA by computing SVD
U, S, V = svd(X2, full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

# from ex2_1_3.py

# threshold_1 = 0.9
# threshold_2 = 0.95
# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho)+1), rho, 'x-')
plt.plot(range(1, len(rho)+1), np.cumsum(rho), 'o-')
# plt.plot([1, len(rho)], [threshold_1, threshold_1], 'k--')
# plt.plot([1, len(rho)], [threshold_2, threshold_2], 'y--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual', 'Cumulative', '90% Threshold', '95% Threshold'])
plt.grid()
plt.show()

print("Finished running run_pca.py")
