#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:33:53 2023

@author: karrar
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the "day" file into a data frame
filename = '../Data/day.csv'
df = pd.read_csv(filename)

# Drop unnecessary columns
df = df.drop(['instant', 'dteday'], axis=1)

# Convert temperature, humidity, and windspeed to their original scales
df['temp'] = df['temp'] * (39 - (-8)) + (-8)
df['atemp'] = df['atemp'] * (50 - (-16)) + (-16)
df['hum'] = df['hum'] * 100
df['windspeed'] = df['windspeed'] * 67


plt.scatter(df['temp'], df['cnt'])
plt.xlabel('Temperature (°C)')
plt.ylabel('Count of Total Rental Bikes')
plt.title('Relationship between Temperature and Bike Rentals')
plt.show()



# Select the columns of interest
cols_of_interest = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
df = df[cols_of_interest]


# Compute basic statistics for each column
mean = df.mean().round(2)
std = df.std(ddof=1).round(2)
var = df.var(ddof=1).round(2)
median = df.median().round(2)
q25 = df.quantile(0.25).round(2)
q75 = df.quantile(0.75).round(2)
min = df.min().round(2)
max = df.max().round(2)
cov = df.cov().round(2)

# Print the results
print('Mean:\n', mean)
print('Standard deviation:\n', std)
print('Variance:\n', var)
print('Median:\n', median)
print('25th percentile:\n', q25)
print('75th percentile:\n', q75)
print('Minimum:\n', min)
print('Maximum:\n', max)
print('Covariance matrix:\n', cov)

plt.scatter(df['temp'], df['cnt'], alpha=0.2)


# Show the plot
plt.show()