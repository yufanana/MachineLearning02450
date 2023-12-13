#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 20:44:10 2023

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
df = df.drop(['instant'], axis=1)

# Convert temperature, humidity, and windspeed to their original scales
df['temp'] = df['temp'] * (39 - (-8)) + (-8)
df['atemp'] = df['atemp'] * (50 - (-16)) + (-16)
df['hum'] = df['hum'] * 100
df['windspeed'] = df['windspeed'] * 67

# Extract year from "dteday" column
df['year'] = pd.to_datetime(df['dteday']).dt.year

# Filter data for years 2011 and 2012
df = df[df['year'].isin([2011, 2012])]

# Group data by season and year, and calculate mean number of rentals
season_labels = ['Spring', 'Summer', 'Fall', 'Winter']
casual_counts = df.groupby(['season', 'year'])['casual'].mean().round(2).unstack()
registered_counts = df.groupby(['season', 'year'])['registered'].mean().round(2).unstack()
season_counts = df.groupby(['season', 'year'])['cnt'].mean().round(2).unstack()

# Create a single plot with 6 lines
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(season_labels, season_counts.loc[:, 2011], label='Bike Rentals 2011', marker='o')
ax.plot(season_labels, casual_counts.loc[:, 2011], label='Casual Rentals 2011', marker='o')
ax.plot(season_labels, registered_counts.loc[:, 2011], label='Registered Rentals 2011', marker='s')

ax.plot(season_labels, season_counts.loc[:, 2012], label='Bike Rentals 2012', marker='s')
ax.plot(season_labels, casual_counts.loc[:, 2012], label='Casual Rentals 2012',marker='^')
ax.plot(season_labels, registered_counts.loc[:, 2012], label='Registered Rentals 2012',marker='^')

ax.set_xlabel('Season')
ax.set_ylabel('Average Number of Rentals')
ax.set_title('Bike Rentals by Season and Year')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)

plt.show()