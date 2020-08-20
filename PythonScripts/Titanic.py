#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd


# In[78]:


train_data = pd.read_csv("../../../Downloads/train.csv") 


# In[79]:


new_train = train_data[['PassengerId','Age','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]


# In[80]:


mapping_for_train = {'male':0,'female':1,'S':2,'C':3,'Q':4}


# In[81]:


new_train = new_train.applymap(lambda s: mapping_for_train.get(s) if s in mapping_for_train else s)


# In[82]:


import numpy as np
new_train = new_train.replace(np.nan, 0)


# In[83]:


train_y = train_data[['Survived']]


# In[84]:


test_data = pd.read_csv("../../../Downloads/test.csv") 


# In[85]:


new_test = test_data[['PassengerId','Age','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]


# In[86]:


new_test = new_test.applymap(lambda s: mapping_for_train.get(s) if s in mapping_for_train else s)
new_test = new_test.replace(np.nan, 0)


# In[87]:


from sklearn.linear_model import LinearRegression


# In[88]:


reg = LinearRegression().fit(new_train,train_y)


# In[89]:


res = reg.predict(new_test)
res = [0 if i <=0.5 else 1 for i in res]


# In[90]:


print (res)


# In[92]:


sur = pd.DataFrame({'Survived':res})


# In[103]:


ress = pd.concat([new_test[['PassengerId']], sur], axis=1)


# In[104]:


print (ress)


# In[107]:


ress.to_csv (r'data1.csv', index = False, header=True)


# In[ ]:




