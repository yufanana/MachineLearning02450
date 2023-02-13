import numpy as np
import pandas as pd

# Load the csv data using the Pandas library
filename = '../Data/day.csv'
df = pd.read_csv(filename)


raw_data = df.values
cols = range(0, df.shape[1])
X = raw_data[:, cols]

# Extract header/attribute names
attributeNames = np.asarray(df.columns[cols])

# Check attributes and data shape
N, M = X.shape
print("No. of instances, N: ", N)
print("No. of attributes, M: ", M)
print("Attribute names: \n", attributeNames)

print("Finished running load_data.py")
