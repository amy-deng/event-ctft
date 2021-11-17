import pandas as pd

import pickle
import numpy as np
import sys,os

try:
    old_dataset = sys.argv[1]
    new_dataset = sys.argv[2]
    thred = int(sys.argv[3])
except:
    print('Usage: old_dataset, new_dataset, thred=7')
    pass

with open('{}/static_tf_2016-2017_800.pkl'.format(old_dataset),'rb') as f:
    static14 = pickle.load(f)
#
with open('{}/dynamic_tf_2016-2017_800.pkl'.format(old_dataset),'rb') as f:
    dynamic14 = pickle.load(f)
with open('{}/attr_tf_2016-2017_800.pkl'.format(old_dataset),'rb') as f:
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

print('c=',)
exit()
os.makedirs(new_dataset, exist_ok=True)
with open('{}/static_tf_2016-2017_800.pkl'.format(new_dataset),'wb') as f:
    pickle.dump(new_static14,f)
#
with open('{}/dynamic_tf_2016-2017_800.pkl'.format(new_dataset),'wb') as f:
    pickle.dump(new_dynamic14,f)

with open('{}/attr_tf_2016-2017_800.pkl'.format(new_dataset),'wb') as f:
    pickle.dump(new_attr14,f)

