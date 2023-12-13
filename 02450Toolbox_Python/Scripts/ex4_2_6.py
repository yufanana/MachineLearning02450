# Exercise 4.2.6

from matplotlib.pyplot import figure, show
from mpl_toolkits.mplot3d import Axes3D

# requires data from exercise 4.1.1
from ex4_2_1 import *

# Indices of the variables to plot
# choose 3 out of 4 the attributes to plot in 3D
ind = [0, 1, 2]
colors = ['blue', 'green', 'red']

f = figure()
ax = f.add_subplot(111, projection='3d') #Here the mpl_toolkits is used
for c in range(C):
    class_mask = (y==c)
    s = ax.scatter(X[class_mask,ind[0]], X[class_mask,ind[1]], X[class_mask,ind[2]], c=colors[c])

# with class 0, plot observations of attribute 0, attribute 1, and attribute 2, with the 0th color
# with class 1, plot observations of attribute 0, attribute 1, and attribute 2, with the 1st color
# with class 2, plot observations of attribute 0, attribute 1, and attribute 2, with the 2nd color

ax.view_init(30, 220)
ax.set_xlabel(attributeNames[ind[0]])
ax.set_ylabel(attributeNames[ind[1]])
ax.set_zlabel(attributeNames[ind[2]])

show()

print('Ran Exercise 4.2.6')