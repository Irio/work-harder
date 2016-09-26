from concurrent import futures
from datetime import datetime
import glob
import pandas as pd
import requests

def download_project(project_id):
    url = 'https://api.catarse.me/project_contributions_per_day?project_id=eq.%i' % project_id
    print(url)
    response = requests.get(url, headers={'Accept': 'text/csv'})
    return response.text

def save_project(project_id, content):
    text_file = open('data/project_contributions/%i.csv' % project_id, 'w')
    text_file.write(content)
    text_file.close()
    print('%i project saved' % project_id)



projects = pd.read_csv('data/projects.csv')
ids = [int(name.split('/')[-1].split('.')[0]) for name in glob.glob('data/project_contributions/*.csv')]
projects = projects[~projects['project_id'].isin(ids)]

with futures.ThreadPoolExecutor(max_workers=8) as executor:
    future_to_result = dict()
    for index, project in projects.iterrows():
        future = executor.submit(download_project, project['project_id'])
        future_to_result[future] = project
    for future in futures.as_completed(future_to_result):
        project = future_to_result[future]
        if future.exception() is not None:
            print('%r raised an exception: %s' % (project['project_id'],
                                                  future.exception()))
        elif future.result() is not None:
            save_project(project['project_id'], future.result())
