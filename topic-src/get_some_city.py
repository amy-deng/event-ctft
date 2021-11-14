import pandas as pd

import pickle

with open('../RUS_w7h7_minday7/hetero_2014-2015_1000.pkl','rb') as f:
    d1=pickle.load(f)

with open('../RUS_w7h7_minday7/hetero_2015-2016_1000.pkl','rb') as f:
    d2=pickle.load(f)


with open('../RUS_w7h7_minday7/hetero_2016-2017_1000.pkl','rb') as f:
    d3=pickle.load(f)

city1 = d1['city']
city2 = d2['city']
city3 = d3['city']



from datetime import datetime

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

    
n = 0
for i, row in df3.iterrows():
    his = row['story_list'][-7:]
    story_list_flatten = list(set([item for sublist in his for item in sublist]))
    if len(story_list_flatten) < 10:
        continue
    n+=1