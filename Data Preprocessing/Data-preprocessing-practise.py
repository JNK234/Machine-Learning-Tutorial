#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[24]:


data = pd.read_csv('Data.csv')
data.shape


# In[25]:


data.head()


#  ## Split the data

# In[26]:


X = data.iloc[:,:-1].values
y = data.iloc[:,3].values


# In[27]:


X


# In[28]:


y


# ## Handle Missing data

# In[29]:


data.isna().sum()


# Therefore Nan is present in Age and Salary columns We can replace the missing Salary by replacing it by average of all the salaries.

# In[37]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")

imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


# In[38]:


X


# ## Encoding categorical Values

# In[58]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder() ,[0])], remainder = "passthrough")
X = np.array(ct.fit_transform(X))


# ## Encoding dependent Variable

# In[59]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[60]:


y


# In[52]:


X


# In[51]:


type(X)


# ## Splitting the Dataset 

# In[61]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2, random_state =1)


# In[71]:


print(X_train)


# In[63]:


print(X_test)


# In[64]:


print(y_train)


# In[65]:


print(y_test)


# ## Feature Scaling

# There are two methods of feature scaling i.e Standardisation and Normalization

# ### Standardization

# In[75]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train[:,8:10] = sc.fit_transform(X_train[:,8:10])

## use only transform on testing set
X_test[:,8:10] = sc.transform(X_test[:,8:10])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




