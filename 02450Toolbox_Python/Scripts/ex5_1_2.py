# exercise 5.1.2
import numpy as np
from sklearn import tree
from platform import system
from os import getcwd
import matplotlib.pyplot as plt
from matplotlib.image import imread

# requires data from exercise 5.1.1
from ex5_1_1 import *

# Fit regression tree classifier, Gini split criterion, no pruning
criterion = 'gini'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=2)
dtc = dtc.fit(X, y)

# Visualize the graph (you can also inspect the generated image file in an external program)
# NOTE: depending on your setup you may need to decrease or increase the figsize and DPI setting
# to get a readable plot. Hint: Try to maximize the figure after it displays.
fname='tree_ex512_' + criterion + '.png'

fig = plt.figure(figsize=(10,10),dpi=100) 
_ = tree.plot_tree(dtc, filled=False,feature_names=attributeNames)
plt.savefig(fname)
plt.show()

print('Ran Exercise 5.1.2')