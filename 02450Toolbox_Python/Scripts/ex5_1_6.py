# exercise 5.1.6
import numpy as np
from sklearn import tree
from platform import system
from os import getcwd
import matplotlib.pyplot as plt
from matplotlib.image import imread

# requires data from exercise 5.1.5
from ex5_1_5 import *

# Fit classification tree using, Gini split criterion, no pruning
criterion='gini'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=100)
dtc = dtc.fit(X,y)

# Visualize the graph (you can also inspect the generated image file in an external program)
# NOTE: depending on your screen resolution and setup you may need to decrease or increase 
# the figsize and DPI setting to get a useful plot. 
# Hint: Try to open the generated png file in an external image editor as it can be easier 
# to inspect outside matplotlib's figure environment.
fname='tree_ex516_' + criterion + '_wine_data.png'
fig = plt.figure(figsize=(10,10),dpi=300) 
_ = tree.plot_tree(dtc, filled=False,feature_names=attributeNames)
plt.savefig(fname)
plt.close() 

fig = plt.figure()
plt.imshow(imread(fname))
plt.axis('off')
plt.box('off')
plt.show()

print('Ran Exercise 5.1.6')
