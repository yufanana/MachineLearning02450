# Exercise 4.2.5

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show)

# requires data from exercise 4.2.1
from ex4_2_1 import *

print("X shape: ", X.shape)

figure(figsize=(12,10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)

        # in each subplot, do 3 plots in total
        # one for each class
        for c in range(C):
            # class_mask is a boolean. True if the
            # observation's class is equal to c
            class_mask = (y==c)
            
            # if m1==0 and m2==0 and c==2:
            #     print(X[class_mask,m2].shape)

            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            
            # label the axes with attribute names
            if m1==M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)

show()

print('Ran Exercise 4.2.5')