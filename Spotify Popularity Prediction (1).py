#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[58]:


spotify = pd.read_csv("spotify_music.csv")
print(spotify)


# In[59]:


spotify.describe()


# In[60]:


spotify.head(20)


# In[92]:


print(spotify.columns)


# In[62]:


#Seeing the correlation numbers. Default is Pearsons.
spotify.corr()


# In[93]:


#Creating a variable pop as the popularity column.
pop= spotify['popularity']


# In[64]:


pop.describe()


# In[65]:


pop1= spotify[['popularity', 'name']]


# In[66]:


print(pop1.sort_values(by=['popularity'], ascending=False).head(50))


# In[67]:


spotify = spotify[spotify['year'] >= 2011]
print(spotify)


# In[68]:


#Per standard, using a histogram will show the skewness of the column. The need to scale or take the log of the
#column might be required but further exploration is required.
plt.figure(figsize = (16,8))
plt.hist(pop)


# In[69]:


spotify.columns


# In[75]:


train, test =  train_test_split(spotify, test_size=0.25, random_state=1)


# In[76]:


x_train = train[['year', 'energy', 'explicit', 'danceability', 'loudness']]
y_train = train[['popularity']]

x_test = test[['year', 'energy', 'explicit', 'danceability', 'loudness']]
y_test = test[['popularity']]


# In[77]:


model = LinearRegression()


# In[85]:


#As an initial regression model, I always run linear regression first to get a feel for the range of the error numbers.
model.fit(x_train, y_train)
model.predict(x_test)

pred = model.predict(x_test)
y_lin_pred = model.predict(x_test)

print('Score: %.3f' % model.score(x_train, y_train))
print('Mean squared error: %.3f' % mean_squared_error(y_test, y_lin_pred))
print('R2 Score: %.3f' % r2_score(y_test, y_lin_pred))


# In[89]:


#Importing support vector regression from sklearn, running the analysis.
from sklearn.svm import SVR

svr = SVR(kernel='rbf', gamma=1e-3, C=150, epsilon=0.3)
svr.fit(x_train, y_train.values.ravel())

y_svr_pred = svr.predict(x_test)
print('Score: %.3f' % svr.score(x_train, y_train))
print('Mean squared error: %.3f' % mean_squared_error(y_test, y_svr_pred))
print('R2: %.3f' % r2_score(y_test, y_svr_pred))


# In[91]:


#Importing nearest neighbors regression from sklearn, running the analysis.
from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors = 7, weights = 'uniform')
knn.fit(x_train, y_train)
y_knn = knn.predict(x_test)

print('Score: %.3f' % knn.score(x_train, y_train))
print('RMSE: %.3f' % mean_squared_error(y_test, y_knn))
print('R2 Score: %.5f' % r2_score(y_test, y_knn))

