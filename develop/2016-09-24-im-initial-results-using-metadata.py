
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

last_year_projects['state'] = last_year_projects['state'].     replace('successful', 1).     replace('waiting_funds', 1).     replace('failed', 0)
last_year_projects['state'] = last_year_projects['state'].astype(np.bool)


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


# In[23]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(model, X, y):
    false_positive_rate, true_positive_rate, _ = roc_curve(y, model.predict(X))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate,
             true_positive_rate,
             sns.xkcd_rgb['denim blue'],
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1],
             [0, 1],
             color=sns.xkcd_rgb['pale red'],
             linestyle='dashed')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    print('Model score: %0.2f%%' % (model.score(X, y) * 100))
    print('True positive rate: %0.2f%%' % (true_positive_rate[1] * 100))
    print('False positive rate: %0.2f%%' % (false_positive_rate[1] * 100))


# In[24]:

plot_roc_curve(clf, X_test, y_test)


# ## Download project details

# Run `python fetch_projects.py` to download information.

# In[25]:

import glob

details_list = []
for filename in glob.glob('../data/project_details/*.csv'):
    details = pd.read_csv(filename)
    details_list.append(details)
project_details = pd.concat(details_list)
del(details_list)


# In[26]:

project_details.shape


# In[27]:

def get_serenata(dataset):
    return dataset[dataset['permalink'] == 'serenata'].iloc[0]


# In[28]:

get_serenata(project_details)


# In[29]:

data = pd.merge(projects,
                project_details,
                on='project_id',
                how='left',
                suffixes=('', '_details'))


# `projects_details` seems to contain everything we need. Forget about `data`, the merged dataset and just work with what we just collected.

# In[30]:

get_serenata(data)


# In[31]:

datetime_columns = ['expires_at', 'online_date']
for column in datetime_columns:
    project_details[column] = pd.to_datetime(project_details[column])


# In[32]:

project_details[project_details['expires_at'].isnull()].iloc[0]


# In[33]:

project_details['state'] =     project_details['state'].replace('waiting_funds', 'successful')


# In[34]:

datetime_columns = ['online_date', 'expires_at']
for column in datetime_columns:
    project_details[column] = pd.to_datetime(project_details[column])


# In[35]:

predicate = (project_details['online_date'] > datetime(2015, 9, 23)) &     (project_details['state'] != 'online') &     (project_details['mode'] == 'aon')
project_details = project_details[predicate]


# In[36]:

project_details['online_days_delta'] =     project_details['expires_at'] - project_details['online_date']
project_details['online_days_delta'] =     project_details['online_days_delta'].apply(lambda row: row.days)


# In[37]:

project_details.iloc[0]


# In[38]:

project_details['state'] = project_details['state'].     replace('successful', 1).     replace('failed', 0)
project_details['state'] = project_details['state'].astype(np.bool)


# In[39]:

X_cols = ['category_id', 'goal', 'online_days_delta']
y = project_details['state']
X = project_details[X_cols]
X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=.25, random_state=0)


# In[40]:

clf = RandomForestClassifier(min_samples_split=1, random_state=0)
clf.fit(X_train, y_train)


# In[41]:

plot_roc_curve(clf, X_test, y_test)


# ## Feature selection

# In[42]:

project_details.iloc[0]


# In[43]:

def parse_user_id(row):
    row = row.replace('\\"', '\"')
    return json.loads(row)['id']

project_details['user_id'] = project_details['user'].apply(parse_user_id)


# In[44]:

def parse_address(row):
    row = row.replace('\\"', '\"')
    return json.loads(row)['state_acronym']

project_details['address_state'] = project_details['address'].apply(parse_address)
project_details['address_state'] =     project_details['address_state'].astype('category')


# In[45]:

from sklearn.preprocessing import LabelEncoder

project_details[['address_state']] =     project_details[['address_state']].apply(LabelEncoder().fit_transform)


# In[46]:

boolean_cols = ['is_published',
                'is_expired',
                'open_for_contributions',
                'user_signed_in',
                'in_reminder',
                'can_request_transfer',
                'is_admin_role',
                'contributed_by_friends']
project_details[boolean_cols] = project_details[boolean_cols].     replace('f', 0).replace('t', 1)
project_details[boolean_cols] = project_details[boolean_cols].astype(np.bool)


# In[47]:

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

X_cols = ['goal',
          'category_id',
          'pledged',
          'total_contributions',
          'total_contributors',
          'expires_at',
          'online_date',
          'online_days',
          'posts_count',
          'address_state',
          'user_id',
          'is_owner_or_admin',
          'total_posts',
          'is_admin_role',
          'contributed_by_friends',
          'online_days_delta']
X, y = project_details[X_cols], project_details['state']
X.shape


# In[48]:

clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
clf.feature_importances_


# In[ ]:

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape


# In[ ]:



