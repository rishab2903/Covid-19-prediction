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

# Finfing total confirmed, recovered and death cases by taking empty lists. 
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


# In[68]:


# Converting all the dates and cases in form of a numpy array

days_since_1_22=np.array([i for i in range(len(dates))]).reshape(-1,1)
world_cases=np.array(world_cases).reshape(-1,1)
total_death=np.array(total_death).reshape(-1,1)
total_recovered=np.array(total_recovered).reshape(-1,1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[73]:


days_since_1_22


# In[ ]:


# Future forecasting for the next 10 days

days_in_future=10
future_forecast=np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates=future_forecast[:10]


# In[74]:


adjusted_dates


# In[107]:


# Convert all the integers into datetime for better visualization
import datetime
start='1/22/2020'
start_date=datetime.datetime.strptime(start,'%m/%d/%Y')
future_forecast_dates=[]
for i in range(len(future_forecast)):
    future_forecast_dates.append(start_date+datetime.timedelta(days=i))


# In[112]:


# For vizualization with the latest data of 3rd APRIL- extracting last column values
latest_confirmed=confirmed_cases[dates[-1]]
latest_deaths=deaths_reported[dates[-1]]
latest_recovered=recovered_cases[dates[-1]]


# In[115]:


confirmed_cases


# In[ ]:





# In[117]:


# Find the list of unique countries

unique_countries=list(confirmed_cases['Country/Region'].unique())


# In[135]:


# Total no. of confirmed cases by each country
country_confirmed_cases=[]
no_cases=[]
for i in unique_countries:
    cases=latest_confirmed[confirmed_cases['Country/Region']==i].sum()
    if cases>0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
        
for i in no_cases:
    unique_countries.remove(i)

country_confirmed_cases


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[150]:


unique_countries=[k for k, v in sorted (zip(unique_countries,country_confirmed_cases),key=operator.itemgetter(1))]    
for i in range(len(unique_countries)):
    country_confirmed_cases[i]=latest_confirmed[confirmed_cases['Country/Region']==unique_countries[i]].sum()
unique_countries


# In[151]:


# number of cases per country/region
print("confirmed cases by countries/regions:")
for i in range(len(unique_countries)):
    print(f'{unique_countries[i]}: {country_confirmed_cases[i]}cases')
    


# In[163]:


# Finding the list of unique provinces

unique_provinces=list(confirmed_cases['Province/State'].unique())


# In[166]:


# Finding the no. of confirmed cases per province, state or city

province_confirmed_case=[]
no_cases=[]
for i in unique_provinces:
    cases=latest_confirmed[confirmed_cases['Province/State']==i].sum()
    if cases>0:
        province_confirmed_case.append(cases)
    else:
        no_cases.append(i)
        
for i in no_cases:
    unique_provinces.remove(i)


# In[167]:


# no. of cases per province/state/city
for i in range(len(unique_provinces)):
    print(f'{unique_provinces[i]}: {province_confirmed_case[i]}cases')


# In[182]:


# plot bar graph to see total confirmed cases across countries

plt.figure(figsize=(23,32))
plt.barh(unique_countries,country_confirmed_cases)
plt.title('No. of COVID-19 cases in countries')
plt.xlabel('No. of COVID-19 confirmed cases')
plt.show()


# In[217]:


# Building the SVM model

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

kernel=['poly','ply','rbf']
c=[0.01]
gamma=[0.1]
epsilon=[0.01]
shrinking=[True,False]
svm_grid={'kernel':kernel,'c':c,'gamma':gamma,'epsilon':epsilon,'shrinking':shrinking}

svm=SVR()


X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False)


svm_search=RandomizedSearchCV(svm,svm_grid, cv =3,return_train_score=True,n_jobs=-1,n_iter=40,verbose=1)

svm_search.fit(X_train_confirmed,y_train_confirmed)


# In[218]:


svm_confirmed


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[214]:





# In[ ]:





# In[ ]:





# In[223]:


#  Finding the best estimator using 
# Predicting the future forecoast using the svm_predict function


svm_pred=svm_confirmed.predict(future_forecast)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[224]:





# In[226]:


svm_pred=svm_confirmed.predict(future_forecast)


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





# In[198]:


svm_search.best_params_


# In[ ]:





# In[ ]:





# In[194]:





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





# In[ ]:





# In[161]:





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





# In[152]:





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





# In[70]:





# In[75]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[76]:





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





# In[38]:





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




