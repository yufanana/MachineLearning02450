import numpy as np
import pandas as pd
from os import path

# Load the csv data using the Pandas library
filename = '../Data/day.csv'
df = pd.read_csv(filename)
df = df.drop('instant', axis=1)
df = df.drop('dteday', axis=1)


# (df['temp']-df['temp'].mean())/df['temp'].std()
# df.dtypes

X = df.values
cols = range(0, df.shape[1])
# X = raw_data[:, cols]

# Extract attribute names
attributeNames = np.asarray(df.columns[cols])

# Check attributes and data shape
N, M = X.shape
print("No. of instances, N: ", N)
print("No. of attributes, M: ", M)

# undo the original max-min normalization
temp = df.columns.get_loc("temp")
atemp = df.columns.get_loc("atemp")
hum = df.columns.get_loc("hum")
windspeed = df.columns.get_loc("windspeed")
for row in range(0, N):
    X[row, temp] = X[row, temp]*(39-(-8)) + (-8)
    X[row, atemp] = X[row, atemp]*(50-(-16)) + (-16)
    X[row, hum] = X[row, hum]*100
    X[row, windspeed] = X[row, windspeed]*67

# standarize ratio data attributes for PCA
cnt = df.columns.get_loc("cnt")
# print("mean cnt: ", X[:, cnt].mean(0))
# print("std cnt: ", np.std(X[:, cnt]))
for col in range(temp, cnt):
    # subtract mean, column by column
    X[:, col] = X[:, col] - np.ones(N) * X[:, col].mean(0)
    X[:, col] = X[:, col] * (1/np.std(X[:, cnt]))

# Show successful code execution
print("Finished running", path.basename(__file__))
