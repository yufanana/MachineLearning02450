import numpy as np
import pandas as pd

# Load the csv data using the Pandas library
filename = '../Data/day.csv'
df = pd.read_csv(filename)


raw_data = df.values
cols = range(0, df.shape[1])
X = raw_data[:, cols]

# Extract attribute names
attributeNames = np.asarray(df.columns[cols])

# Check attributes and data shape
N, M = X.shape
# print("No. of instances, N: ", N)
# print("No. of attributes, M: ", M)
# print("Attribute names: \n", attributeNames)


X1 = X
# undo the original max-min normalization
for row in range(0, N):
    X1[row, 9] = X[row, 9]*(39-(-8)) + (-8)
    X1[row, 10] = X[row, 10]*(50-(-16)) + (-16)
    X1[row, 11] = X[row, 11]*100
    X1[row, 12] = X[row, 12]*67

X2 = np.empty((731, 7))
# standarize ratio data attributes for PCA
X2 = X1[:, 9:16]
X2 = X2 - np.ones((N, 1))*X2.mean(0)
X2 = X2*(1/np.std(X2))

# concatatenate the 2 arrays
Y = np.concatenate((X1[:, 0:9], X2), axis=1)

# Check attributes and data shape
N, M = X2.shape
# print("No. of instances, N: ", N)
# print("No. of attributes, M: ", M)

for i in range(N):
    for j in range(M):
        X2[i, j] = float(X2[i, j])

# print(type(X2[0, 0]))
# print(type(X2[0, 1]))
# print(type(X2[0, 2]))
# print(type(X2[0, 3]))
# print(type(X2[0, 4]))
# print(type(X2[0, 5]))
# print(type(X2[0, 6]))
print("Finished running load_data.py")
