import pandas as pd

import pickle
import numpy as np


with open('../EGY_w7h7_minday7/hetero_2015-2016_1000.pkl','rb') as f:
    d1=pickle.load(f)



with open('../EGY_w7h7_minday7/static_tf_2016-2017_800.pkl','rb') as f:
    static14 = pickle.load(f)
#
with open('../EGY_w7h7_minday7/dynamic_tf_2016-2017_800.pkl','rb') as f:
    dynamic14 = pickle.load(f)
with open('../EGY_w7h7_minday7/attr_tf_2016-2017_800.pkl','rb') as f:
    attr14 = pickle.load(f)

city = attr14['city']
y = attr14['y']
date = attr14['date']
print('y',type(y),y.shape)
print('date',type(date),len(date))
print('city',type(city),len(city))

new_static14 = []
new_dynamic14 = []
new_attr = []
new_y = []
new_date = []
new_city = []
c =0

for i in range(len(static14)):
    static = static14[i]
    num_doc = static.num_nodes('doc')
    if num_doc <= 10:
        c+=1
    else:
        new_static14.append(static)
        new_dynamic14.append(dynamic14[i])
        new_city.append(city[i])
        new_date.append(date[i])
        new_y.append(y[i])

print('new_date',type(new_date),len(new_date))
print('new_city',type(new_city),len(new_city))
new_y = np.stack(new_y,0)
new_attr14 = {'y':new_y,'date':new_date,'city':new_city}

with open('../RUS_w7h7_minday10/static_tf_2016-2017_800.pkl','wb') as f:
    pickle.dump(new_static14,f)
#
with open('../RUS_w7h7_minday10/dynamic_tf_2016-2017_800.pkl','wb') as f:
    pickle.dump(new_dynamic14,f)

with open('../RUS_w7h7_minday10/attr_tf_2016-2017_800.pkl','wb') as f:
    pickle.dump(new_attr14,f)







with open('../EGY_w7h7_minday7/hetero_2014-2015_1000.pkl','rb') as f:
    d1=pickle.load(f)

with open('../EGY_w7h7_minday7/hetero_2015-2016_1000.pkl','rb') as f:
    d2=pickle.load(f)


with open('../EGY_w7h7_minday7/hetero_2016-2017_1000.pkl','rb') as f:
    d3=pickle.load(f)

city1 = d1['city']
city2 = d2['city']
city3 = d3['city']



from datetime import datetime

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


# n = 0
# for i, row in df1.iterrows():
#     his = row['story_list'][-7:]
#     story_list_flatten = list(set([item for sublist in his for item in sublist]))
#     if len(story_list_flatten) < 7:
#         continue
#     n+=1


# df4=df.loc[(df['date']<'2016-01-01') & (df['date']>'2015-01-01')]

# df1=df.loc[(df['date']<'2015-01-01') & (df['date']>'2014-01-01')]
import pickle
dataset = 'THA_w7h7_mind3n7df0.01'
dataset = 'EGY_w7h7_mind3n7df0.01'
dataset = 'AFG_w7h7_mind3n7df0.01'
dataset = 'RUS_w7h7_mind3n10df0.01'
d = []
with open(dataset+'/dyn_tf_2014-2015_900.pkl','rb') as f:
    d += pickle.load(f)

with open(dataset+'/dyn_tf_2015-2016_900.pkl','rb') as f:
    d += pickle.load(f)

with open(dataset+'/dyn_tf_2016-2017_900.pkl','rb') as f:
    d += pickle.load(f)


l = []
for g in d:
    u = g.edges['ww'].data['time'].unique()
    l.append(len(u))

l = np.array(l)

# THA 6.409209383145091
# EGY 6.37345003646973
# AFG 6.411987860394537
# RUS 6.23073611708997