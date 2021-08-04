# import glob
import sys
import numpy as np
import pandas as pd
import os
import time
import pickle
import collections
import dgl
from dgl.data.utils import save_graphs,load_graphs
import torch
from datetime import date,timedelta
print(os.getcwd())


'''
graph_labels = {"label": torch.Tensor(graph_label).int(),'time':torch.Tensor(graph_time).int()}
save_graphs(path+"data.bin", graph_list, graph_labels)
'''
try:
    RAWDATA = sys.argv[1]
    # DATASET = sys.argv[1]
    # STARTTIME = str(sys.argv[2])
    # ENDTIME= str(sys.argv[3])
    # DELTA = int(sys.argv[2])
    WINDOW = int(sys.argv[2])
    HORIZON = int(sys.argv[3])
    PREDWINDOW = int(sys.argv[4])
except:
    print("Usage: RAWDATA, WINDOW=14, HORIZON=1, PREDWINDOW=3")
    exit()


DATASET = RAWDATA.split('-')[7][:-4]
print(DATASET,'dataset_name')

filename = RAWDATA.split('/')[-1]
print('path',RAWDATA,'filename',filename)
start_year = int(filename.split('-')[0])
start_month = int(filename.split('-')[1])
start_day = int(filename.split('-')[2])
end_year = int(filename.split('-')[3])
end_month = int(filename.split('-')[4])
end_day = int(filename.split('-')[5])


path = '../data/{}/'.format(DATASET)
os.makedirs(path, exist_ok=True)
print('path',path)



df = pd.read_csv(RAWDATA,sep=';')
df = df.drop_duplicates(subset=['data_id'], keep='first')
df['event_date'] = pd.to_datetime(df['event_date'])



DELTA = 1
# date and get protest count
start = date(start_year,start_month,start_day)
end = date(end_year,end_month,end_day)
delta = timedelta(days=DELTA)
date_table = {}
i = 0
while start <= end:
    start_date_str = start.strftime("%Y-%m-%d")
    date_table[start_date_str] = i
    i += 1
    start += delta




subevents_df = pd.read_csv(path + "subevent2id.txt",names=['id','name'],sep='\t')
subevents = subevents_df['name'].unique()
subevents.sort()
subevent_count_dict = {}
for v in subevents:
    subevent_count_dict[v] = np.array([0 for i in range(len(date_table))])
start = date(start_year,start_month,start_day)
end = date(end_year,end_month,end_day)
dayi = 0
while start <= end:
    start_date_str = start.strftime("%Y-%m-%d")
    df_day = df.loc[df['event_date'] == start_date_str]
    df_count = df_day['sub_event_type'].value_counts().rename_axis('unique_values').reset_index(name='counts')
    for i,row in df_count.iterrows():
        subevent_count_dict[row['unique_values']][dayi] = row['counts']
    dayi += 1
    start += delta
Protests_count = subevent_count_dict['Protest with intervention'] + subevent_count_dict['Peaceful protest'] + subevent_count_dict['Excessive force against protesters']

# build sequence data
subevent_count_seq = []
for k in subevents:
    v = subevent_count_dict[k].tolist()
    subevent_count_seq.append(v)
subevent_count_seq = np.array(subevent_count_seq)
subevent_count_seq = np.swapaxes(subevent_count_seq,0,1)
# print(subevent_count_seq.shape,'subevent_count_seq')
 

# get label and Y, and corresponding time
date_ids = list(date_table.values())
data_time = []
data_Y = []
data_treat = []
data_X = []
for i in range(WINDOW,len(date_ids),HORIZON+PREDWINDOW-1): # no overlap of pre_window
    last = subevent_count_seq[i-WINDOW:i]
#     print(i-WINDOW,i,'---',i,i+WINDOW,'   yyy',i+WINDOW,i+WINDOW+PREDWINDOW-1)
    curr = subevent_count_seq[i:i+WINDOW]
    data_X.append(curr)
    treat = curr.sum(0) - last.sum(0)
    data_treat.append(list(np.where(treat>0,1,0)))
    # print(i+WINDOW,i+WINDOW+PREDWINDOW-1)
    protest = Protests_count[i+WINDOW:i+WINDOW+PREDWINDOW].sum()
    data_Y.append(1 if protest > 0 else 0)
    data_time.append(date_ids[i+WINDOW])
    if i+WINDOW >=len(date_ids) or i+WINDOW+PREDWINDOW-1 >= len(date_ids):
        break

# to build counter factual data
data_X = np.stack(data_X) # t,window,#subevent


# get text for each day
data_text = []
date_ids = list(date_table.values())
date_name = list(date_table.keys())
date_table_rev = dict(zip(date_ids,date_name))
for i in range(WINDOW,len(date_table),HORIZON+PREDWINDOW-1): # no overlap of pre_window
    date_list = [date_table_rev[j] for j in range(i,i+WINDOW)]
    # print(date_list)
    df_window = df.loc[df['event_date'].isin(date_list)]['notes']
    # curr = subevent_count_seq[i:i+WINDOW]
    data_text.append(' '.join(df_window.values))
    if i+WINDOW >=len(date_ids) or i+WINDOW+PREDWINDOW-1 >= len(date_ids):
        break
 
print(len(data_time),len(data_Y),data_X.shape, len(data_time))