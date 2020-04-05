#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import time
import math
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


confirmed_cases=pd.read_csv("~/Downloads/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")


# In[8]:


deaths_reported=pd.read_csv("~/Downloads/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")


# In[9]:


recovered_cases=pd.read_csv("~/Downloads/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")


# In[10]:


confirmed_cases.head()


# In[11]:


deaths_reported.head()


# In[12]:


recovered_cases.head()


# In[14]:


cols=confirmed_cases.keys()
cols


# In[35]:


# extracting only the date column
confirmed=confirmed_cases.loc[:,cols[4]:cols[-1]]
confirmed
deaths=deaths_reported.loc[:,cols[4]:cols[-1]]
recoveries=recovered_cases.loc[:,cols[4]:cols[-1]]

# Finfing total confirmed, recoered and death cases by taking empty lists. 
#Also finding mortality rate

dates= confirmed.keys()
world_cases=[]
total_death=[]
total_recovered=[]
mortality_rate=[]

for i in dates:
    confirmed_sum=confirmed[i].sum()
    death_sum=deaths[i].sum()
    recovered_sum=recoveries[i].sum()
    world_cases.append(confirmed_sum)
    total_death.append(death_sum)
    total_recovered.append(recovered_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    


# In[ ]:





# In[31]:


confirmed_sum


# In[32]:


death_sum


# In[33]:


recovered_sum


# In[36]:


world_cases


# In[38]:


mortality_rate


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




