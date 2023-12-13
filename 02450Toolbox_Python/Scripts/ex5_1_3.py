# exercise 5.1.3
import os
import numpy as np
from sklearn import tree
from platform import system
from os import getcwd
import matplotlib.pyplot as plt
from matplotlib.image import imread
#import graphviz
#import pydotplus

# requires data from exercise 5.1.1
from ex5_1_1 import *

# Fit regression tree classifier, Entropy split criterion, no pruning
criterion='entropy'
# dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=2)
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=1.0/N)
dtc = dtc.fit(X,y)

# convert the tree into a png file using the Graphviz toolset
fname='tree_ex513_' + criterion + '.png'

# Visualize the graph (you can also inspect the generated image file in an external program)
# NOTE: depending on your setup you may need to decrease or increase the figsize and DPI setting
# to get a useful plot. Hint: Try to maximize the figure after it displays.
fig = plt.figure(figsize=(10,10),dpi=100) 
_ = tree.plot_tree(dtc, filled=False,feature_names=attributeNames)
plt.savefig(fname)
plt.show()

print('Ran Exercise 5.1.3')

