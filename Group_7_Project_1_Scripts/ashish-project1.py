#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[12]:


df=pd.read_csv('C:/Users/User/Downloads/02450Group7-main/02450Group7-main/Data/day_n.csv')

df.head()


# In[13]:


drop_col=['instant','dteday']

for i in df.columns:
    if i in drop_col:
        df.drop(labels=i,axis=1,inplace=True)


print(df.head(2))


# In[14]:


df.describe()


# In[18]:


df['weathersit']=df['weathersit'].map({1:'Clear to Partly Cloudy',2:'Misty and Cloudy',3:'Light Rain or Snow',4:'Heavy Rain or Snow'})
df['season']=df['season'].map({1:'Spring',2:'Summer',3:'Fall',4:'Winter'})
df['mnth']=df['mnth'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
df['weekday']=df['weekday'].map({0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'})
print("New column values: ")
print(df.head())


# In[19]:


#checking outliers after visualization

# count the number of values in the 'column_name' column that are greater than 30
count_windspeed = (df['windspeed'] > 30).sum()

# print the count
print('Number of values greater than 30 in windspeed:', count_windspeed)

# count the number of values in the 'column_name' column that are greater than 30
count_hum = (df['hum'] < 20).sum()
print('Number of values greater than 20 in hum', count_hum)


# In[20]:


#creating subplot grid

fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(12,5))  # this creates a figure with 1 row and 2 column of subplots
# fig variable represents the figure object
# ax represents an array of 2 axes objects , one for each column


# creating main title
fig.suptitle('visualization of target variable (cnt)'.upper(),fontsize=12)


#populating plot1:

sns.boxplot(y=df['cnt'],ax=ax[0],palette='Purples')  # other color palettes= Blues, Greens, Reds, Purples,Oranges, PuBu,coolwarm

ax[0].set_title('boxplot of target variable[cnt]'.upper(),fontsize=10,fontweight=20,y=1.02)

ax[0].set_ylabel(' ')  # instead of ylabel , added space between plots
ax[0].set_xlabel(' ')

#populating plot2:

sns.histplot(df['cnt'],ax=ax[1],color='purple',kde=True)
ax[1].set_title('histogram target variable (cnt)'.upper(),fontsize=10,fontweight=20,y=1.02)

ax[1].set_ylabel(' ')
ax[1].set_xlabel(' ')


#plt.show()


# In[21]:


#Creating a list of continuous independent variables

cont_var=[i for i in df.select_dtypes(exclude='object').columns if df[i].nunique()>2 and i != 'cnt']

#Creating a subplot grid

fig,ax=plt.subplots(nrows=1,ncols=len(cont_var), figsize=(12,5))

#title
plt.suptitle('boxplots of continuous independent varibles- weather trends'.upper(),fontsize=12)

#Looping to fill subplot grid with plots
for i in range(len(cont_var)):
    sns.boxplot(y=df[cont_var[i]],ax=ax[i], palette= 'Purples')
                
    #setting aesthetics and readability
    #    
    ax[i].set_title(f'{cont_var[i].upper()}',fontsize=15)
    ax[i].set_ylabel(' ')


#setting final aesthetics
plt.tight_layout()
#plt.show()                


# In[22]:



#checking for null values 
df.isnull().sum()


# In[26]:


#covariance plot

df = df.select_dtypes(include=['float64', 'int64'])  # only keep numerical columns
cov_matrix = df.cov()
print(cov_matrix)

plt.figure(figsize=(15, 10))

# Create a red-blue colormap
cmap = sns.diverging_palette(10, 240, s=85, l=50, n=4, center='light')
sns.heatmap(cov_matrix, cmap='coolwarm', annot=True, fmt='.2f')


# In[ ]:


#corelation plot

#setting plot size
plt.figure(figsize=(10,5))

print(df.corr())

#plotting heatmap
sns.heatmap(df.corr(),annot=True,cmap='Purples_r')

# Show the plot
plt.show()


# In[24]:


X=df.values

col1 = df.columns.get_loc("temp")
col2 = df.columns.get_loc("cnt")
print(df.columns)


X=X[:,col1:col2+1]
print(X)
print(X.shape)


# In[25]:


scaler =StandardScaler()
scaler.fit(X)

scaled_data=scaler.transform(X)
print(scaled_data)
print("the mean of scaled data :",scaled_data.mean(0))


# In[27]:


# Compute the singular value decomposition of the centered data matrix
U, s, Vt = np.linalg.svd(scaled_data)
Z = scaled_data.dot(Vt)

print("U:",U)
print("s:", s)
print("Vt:",Vt)
print("Z:",Z)


plt.scatter(Z[:, 0], Z[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# In[29]:


#loadings
data=np.array([[0.46,0.46,0.02,-0.16,0.38,0.42,0.48],
               [0.12,0.14,0.79,-0.51,-0.13,-0.18,-0.19],
               [0.4,0.38,0.18,0.7,0.02,-0.32,-0.25],
               [-0.03,-0.03,0.28,0.31,-0.69,0.56,0.2],
               [-0.34,-0.34,0.52,0.37,0.54,0.08,0.26],
               [-0.7,0.71,-0.01,0.02,0.0,-0.0,-0.0],
               [0.0,-0.0,0.0,0.0,-0.27,-0.6,0.75]])
# Create a red-blue colormap
cmap = sns.diverging_palette(10, 240, s=85, l=50, n=7, center='light')

# Create a heatmap with the red-blue colormap
sns.heatmap(data, cmap=cmap)

y_labels=['temp','atemp','hum','wndspd','casual','reg','count']
x_labels=['PC1','PC2','PC3','PC4','PC5','PC6','PC7']
# Set x-axis and y-axis labels
plt.xlabel('principal components')
plt.ylabel('attributes')

# Set x-axis and y-axis tick labels
plt.xticks(np.arange(len(x_labels))+0.5, x_labels)
plt.yticks(np.arange(len(y_labels))+0.5, y_labels) 

# Adjust the position of the tick labels
plt.tick_params(axis='x', which='major', pad=15)
plt.tick_params(axis='y', which='major', pad=15)

