#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[2]:


forest=pd.read_csv("forestfires.csv")


# In[3]:


forest.head()


# In[4]:


forest.shape


# In[5]:


forest.info()


# In[6]:


forest['month']=forest['month'].astype('category')
forest['day']=forest['day'].astype('category')


# In[7]:


forest.dtypes


# In[8]:


from sklearn import preprocessing                      
label_encoder = preprocessing.LabelEncoder()


# In[9]:


#we need size_category string type data into binary numbers


# In[10]:


forest['size_category'] = label_encoder.fit_transform(forest['size_category'])


# In[11]:


forest.size_category


# In[12]:


forest.size_category.unique()


# In[13]:


forest.size_category.value_counts()


# In[14]:


#we also need to convert categories into numbers


# In[15]:


forest['month']=label_encoder.fit_transform(forest['month'])
forest['day']=label_encoder.fit_transform(forest['day'])


# In[17]:


forest


# In[18]:


forest.shape


# In[19]:


# Splitting the data into x and y as input and output

X = forest.iloc[:,0:30]
Y = forest.iloc[:,30]


# In[20]:


X


# In[21]:


Y


# In[22]:


# Splitting the data into training and test dataset

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0)


# In[23]:


#Model Building by SVM


# In[24]:


clf=SVC()
clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)


# In[25]:


y_pred


# In[26]:


#it predict 1 as small and 0 as Large


# In[ ]:




