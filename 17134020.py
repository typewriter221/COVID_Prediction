#!/usr/bin/env python
# coding: utf-8

# In[42]:
#Importing modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[31]:
#Importing data

train=pd.read_csv('./Downloads/train.csv')
test=pd.read_csv('./Downloads/test.csv')


# In[32]:
# Filling NaN values in Province_State column

train['Province_State'].fillna('single',inplace=True)
test['Province_State'].fillna('single',inplace=True)



# In[34]:

# Defining a new column with province and country name combined
train['unique_region']=train['Province_State']+' '+train['Country_Region']
test['unique_region']=test['Province_State']+' '+test['Country_Region']


# Based on train and test file there are 77 values for each region and 43 predictions are to be made . Total of 306 different regions are there.


# In[57]:


train_date_number=list(range(1,78))*306
test_date_number=list(range(78,78+43))*306


# In[58]:

# date_num defines number of days (starting from 1).
train['date_num']=train_date_number
test['date_num']=test_date_number


# In[61]:

# Encoding regions
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(train['unique_region'].values)
train['unique_region_num']=le.transform(train['unique_region'])
test['unique_region_num']=le.transform(test['unique_region'])


# In[70]:

# Since growth is exponential, adding 1 and taking log transform
train['ConfirmedCases_log']=train['ConfirmedCases'].apply(lambda x: np.log(x+1) )
train['Fatalities_log']=train['Fatalities'].apply(lambda x:np.log(x+1))


# In[72]:


X_train=train[['date_num', 'unique_region_num']].values
X_test=train[['date_num', 'unique_region_num']].values
Y_train_conf_cases=train['ConfirmedCases_log'].values
Y_train_fat_cases=train['Fatalities_log'].values       


# In[75]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(X_train,Y_train_conf_cases)
new_conf_cases=(np.exp(model.predict(X_test))-1).astype(int)
model.fit(X_train,Y_train_fat_cases)
new_fat_cases=(np.exp(model.predict(X_test))-1).astype(int)


# In[79]:




