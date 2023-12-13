from toolbox_02450 import mcnemar
from ex7_1_1 import *

# Compute the McNemar test
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)

print("\n")
print("theta = theta_A-theta_B point estimate", thetahat, "\nCI: ", CI, "\np-value", p)

[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)

print("\n")
print("theta = theta_A-theta_C point estimate", thetahat, "\nCI: ", CI, "\np-value", p)


'''
Output: model A vs model B
Result of McNemars test using alpha= 0.05
Comparison matrix n
[[143.   1.]
 [  4.   2.]]
Warning, n12+n21 is low: n12+n21= 5.0
Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] =  (-0.048935661848805934, 0.008952198322319305)
p-value for two-sided test A and B have same accuracy (exact binomial test): p= 0.375   

theta = theta_A-theta_B point estimate -0.02  
CI:  (-0.048935661848805934, 0.008952198322319305)
p-value: 0.375
'''

# theta_hat < 0 means theta_B is higher, thus model B is better A
# reached same conclusion as 7.1.2
# but this is weak evidence due to high p-value, 
# and confidence interval barely doesnt contain 0,
# so theta_hat was very close to being inside the 95% CI

'''
Output: model A vs model C
Result of McNemars test using alpha= 0.05
Comparison matrix n
[[129.  15.]
 [  3.   3.]]
Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] =  (0.026135137059841274, 0.13363519131178903)
p-value for two-sided test A and B have same accuracy (exact binomial test): p= 0.007537841796875

theta = theta_A-theta_C point estimate 0.08
 CI:  (0.026135137059841274, 0.13363519131178903)
 p-value 0.007537841796875
'''

# theta_hat > 0 meams theta_A is higher, thus modeal A is better than C
# p-value = 0.00754, which is less than the significance level alpha = 0.05
# thus, this indicates the result is unlikely due to chance
