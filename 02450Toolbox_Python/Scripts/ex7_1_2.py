from toolbox_02450 import jeffrey_interval
from ex7_1_1 import *

# Compute the Jeffreys interval
alpha = 0.01
[thetahatA_1, CIA_1] = jeffrey_interval(y_true, yhat[:,0], alpha=alpha)
[thetahatA_2, CIA_2] = jeffrey_interval(y_true, yhat[:,1], alpha=alpha)
[thetahatA_3, CIA_3] = jeffrey_interval(y_true, yhat[:,2], alpha=alpha) 

print("Theta point estimate", thetahatA_1, " CI: ", CIA_1)
print("Theta point estimate", thetahatA_2, " CI: ", CIA_2)
print("Theta point estimate", thetahatA_3, " CI: ", CIA_3)

'''
Output

alpha = 0.05
Theta point estimate 0.956953642384106  CI:  (0.9194225123023887, 0.9831344032786383)
Theta point estimate 0.9768211920529801  CI:  (0.947595470192869, 0.9943357273513206)   
Theta point estimate 0.8774834437086093  CI:  (0.8208649682806228, 0.9246522400250042) 

alpha = 0.1
Theta point estimate 0.956953642384106  CI:  (0.9268651478327419, 0.9801903026556568)
Theta point estimate 0.9768211920529801  CI:  (0.9538135438545823, 0.9927410729129449)  
Theta point estimate 0.8774834437086093  CI:  (0.8310541533673992, 0.9182188891809034) 

alpha = 0.01
Theta point estimate 0.956953642384106  CI:  (0.9036812666318764, 0.9879664211985192)
Theta point estimate 0.9768211920529801  CI:  (0.9341061764605542, 0.9966802153327683)  
Theta point estimate 0.8774834437086093  CI:  (0.8001055911151742, 0.9362620636255784)

Lower alpha results in a wider confidence interval
'''
