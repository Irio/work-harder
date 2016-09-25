
# coding: utf-8

# # Catarse - Predicting project success

# ```
# $ curl --header "Accept:text/csv" https://api.catarse.me/projects\?state\=notin.draft,rejected,deleted --output projects.csv
# $ curl --header "Accept:text/csv" https://api.catarse.me/project_details\?limit=1000
# ```

# In[1]:

from altair import *
from datetime import datetime
import json
import numpy as np
import pandas as pd


# In[2]:

projects = pd.read_csv('../data/projects.csv')
projects.shape


# In[3]:

projects.head()


# In[4]:

projects.iloc[0]


# ## Type convertions

# In[5]:

datetime_columns = ['online_date', 'expires_at']
for column in datetime_columns:
    projects[column] = pd.to_datetime(projects[column])


# In[6]:

category_cols = ['mode', 'state', 'state_order', 'state_acronym']
for column in category_cols:
    projects[column] = projects[column].astype('category')


# In[7]:

boolean_cols = ['recommended', 'open_for_contributions', 'contributed_by_friends']
projects[boolean_cols] = projects[boolean_cols].replace('f', 0).replace('t', 1)
projects[boolean_cols] = projects[boolean_cols].astype(np.bool)


# In[8]:

json_cols = ['remaining_time', 'elapsed_time']
for column in json_cols:
    projects[column] = projects[column].apply(lambda row: json.loads(row))


# In[9]:

projects.sample(random_state=0).iloc[0]


# ## Project status

# In[10]:

Chart(projects).mark_bar().encode(
    x=X('online_date:T', timeUnit='yearmonth'),
    y='count(*)',
    color='state',
)


# ## Project status since the beginning of the year

# In[11]:

year_projects = projects[projects['online_date'] > datetime(2016, 1, 1)]


# In[12]:

Chart(year_projects).mark_bar(stacked='normalize').encode(
    x=X('online_date:T', timeUnit='month'),
    y='count(*)',
    color='state',
)


# It seems to have expired projects online.

# In[13]:

is_expired_and_online = (year_projects['expires_at'] < datetime.now()) &     (year_projects['state'] == 'online')


# In[14]:

year_projects.loc[is_expired_and_online]


# In[15]:

year_projects.loc[is_expired_and_online].iloc[0]


# In[16]:

projects['state'].unique()


# Since Catarse has been receiving many more projects in the last year, we're going to limit our model just for the last 12 months.

# In[17]:

predicate = (projects['online_date'] > datetime(2015, 9, 23)) &     (projects['state'] != 'online')
last_year_projects = projects[predicate]


# In[18]:

Chart(last_year_projects).mark_bar().encode(
    x=X('online_date:T', timeUnit='yearmonth'),
    y='count(*)',
    color='state',
)


# Status `waiting_funds`, for the purpose of our analysis, is the same as `successful`.

# In[19]:

last_year_projects['state'] =     last_year_projects['state'].replace('waiting_funds', 'successful')
last_year_projects['state'].unique()


# In[20]:

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[21]:

X_cols = ['category_id']
y = last_year_projects['state']
X = last_year_projects[X_cols]
X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=.25, random_state=0)


# In[22]:

clf = RandomForestClassifier(n_estimators=10,
                             max_depth=None,
                             min_samples_split=1,
                             random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# ## Download project details

# Run `python fetch_projects.py` to download information.

# In[23]:

import glob

details_list = []
for filename in glob.glob('../data/project_details/*.csv'):
    details = pd.read_csv(filename)
    details_list.append(details)
project_details = pd.concat(details_list)
del(details_list)


# In[24]:

project_details.shape


# In[25]:

def get_serenata(dataset):
    return dataset[dataset['permalink'] == 'serenata'].iloc[0]


# In[26]:

get_serenata(project_details)


# In[27]:

data = pd.merge(projects,
                project_details,
                on='project_id',
                how='left',
                suffixes=('', '_details'))


# `projects_details` seems to contain everything we need. Forget about `data`, the merged dataset and just work with what we just collected.

# In[38]:

get_serenata(data)


# In[29]:

datetime_columns = ['expires_at', 'online_date']
for column in datetime_columns:
    project_details[column] = pd.to_datetime(project_details[column])


# In[30]:

project_details[project_details['expires_at'].isnull()].iloc[0]


# In[31]:

project_details['state'] =     project_details['state'].replace('waiting_funds', 'successful')


# In[32]:

datetime_columns = ['online_date', 'expires_at']
for column in datetime_columns:
    project_details[column] = pd.to_datetime(project_details[column])


# In[33]:

predicate = (project_details['online_date'] > datetime(2015, 9, 23)) &     (project_details['state'] != 'online') &     (project_details['mode'] == 'aon')
project_details = project_details[predicate]


# In[34]:

project_details['online_days_delta'] =     project_details['expires_at'] - project_details['online_date']
project_details['online_days_delta'] =     project_details['online_days_delta'].apply(lambda row: row.days)


# In[35]:

project_details.iloc[0]


# In[36]:

X_cols = ['category_id', 'goal', 'online_days_delta']
y = project_details['state']
X = project_details[X_cols]
X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=.25, random_state=0)


# In[37]:

clf = RandomForestClassifier(min_samples_split=1, random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# In[ ]:



