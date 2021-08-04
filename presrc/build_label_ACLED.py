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


subevent2id_file = path + "subevent2id.txt"
subevents = subevent2id_file.read().split(',')
print('subevents',subevents)
subevent_count_dict = {}
for v in subevents:
    subevent_count_dict[v] = np.array([0 for i in range(len(date_table))])

i = 0
while start <= end:
    start_date_str = start.strftime("%Y-%m-%d")

    df_day = df.loc[df['event_date'] == start_date_str]
    df_count = df_day['sub_event_type'].value_counts().rename_axis('unique_values').reset_index(name='counts')
    for i,row in df_count.iterrows():
        subevent_count_dict[row['unique_values']][i] = row['counts']
    i += 1
    start += delta

# subevent_count_dict

exit()


start_date = date(start_year, start_month, start_day)
end_date = date(end_year, end_month, end_day)
delta = timedelta(days=DELTA)
day_i = 0
last_date = start_date - delta
while start_date <= end_date:
    last_date_str = last_date.strftime("%Y-%m-%d") #("%d %B %Y")
    date_str = start_date.strftime("%Y-%m-%d")
    df_day = df.loc[(df['event_date'] > last_date_str) & (df['event_date'] <= date_str)]
    if day_i%300==0:
        print('#',day_i,len(df_day),len(df))
    df_count = df_day[event_type_column].value_counts().rename_axis('unique_values').reset_index(name='counts')
    for i,row in df_count.iterrows():
        subevent_count_dict[row['unique_values']][day_i] = row['counts']
    last_date = start_date
    start_date += delta
    day_i += 1
print('day_i =',day_i)



date_ids = date_table.values()
date_name = date_table.keys()
for i in range(len)
data_X = []
data_Y = []
for i in range(0,len(date_table),HORIZON+PREDWINDOW-1): # no overlap of pre_window
    i+WINDOW, :i+WINDOW+PREDWINDOW
    pass