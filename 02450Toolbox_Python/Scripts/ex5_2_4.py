# exercise 5.2.4
from matplotlib.pylab import figure, subplots, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm

# requires wine data from exercise 5.1.5
from ex5_1_5 import *

# Split dataset into features and target vector
alcohol_idx = attributeNames.index('Alcohol')
y = X[:,alcohol_idx]

X_cols = list(range(0,alcohol_idx)) + list(range(alcohol_idx+1,len(attributeNames)))
X = X[:,X_cols]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)
print(model.coef_)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
fig, (ax0, ax1) = subplots(nrows=2)
ax0.plot(y, y_est, '.')
ax0.set_xlabel('Alcohol content (true)')
ax0.set_ylabel('Alcohol content (estimated)');
ax0.set_title('Scatter Plot')

ax1.hist(residual,40)
ax1.set_title('Residual Histogram')

fig.tight_layout()
show()

print('Ran Exercise 5.2.4')
