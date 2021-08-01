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
    DELTA = int(sys.argv[2])
except:
    print("Usage: RAWDATA, DELTA=1")
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
print('path',path)

# map actors
# map event types (sub)
# save data_id

df = pd.read_csv(RAWDATA,sep=';')
df = df.drop_duplicates(subset=['data_id'], keep='first')
df['event_date'] = pd.to_datetime(df['event_date'])


# entity
df['assoc_actor_1'] = df['assoc_actor_1'].fillna('Undefined')
df['assoc_actor_2'] = df['assoc_actor_2'].fillna('Undefined')
sub = df['actor1'].unique()
ob = df['actor2'].unique()
entity = list(set(sub + ob))
entity.sort()
entity_idx_path = path + 'entity2id.txt'
entity_idx_f = open(entity_idx_path, 'w')
for i in range(len(entity)):
    entity_idx_f.write("{}\t{}\n".format(i,entity[i]))
entity_idx_f.close()
print(entity_idx_path, 'saved!')
entity_table = dict(zip(entity,list(range(len(entity)))))

# event types
df['event_type'] = df['event_type'].fillna('Undefined') # if nan
df['sub_event_type'] = df['sub_event_type'].fillna('Undefined')
event_type = df['event_type'].unique()
sub_event_type = df['sub_event_type'].unique()
event_idx_path = path + 'event2id.txt'
event_idx_f = open(event_idx_path, 'w')
for i in range(len(event_type)):
    event_idx_f.write("{}\t{}\n".format(i,event_type[i]))
print(event_idx_path, 'saved!')
event_table = dict(zip(event_type,list(range(len(event_type)))))

sub_event_idx_path = path + 'subevent2id.txt'
sub_event_idx_f = open(sub_event_idx_path, 'w')
for i in range(len(sub_event_type)):
    sub_event_idx_f.write("{}\t{}\n".format(i,sub_event_type[i]))
print(sub_event_idx_path, 'saved!')
sub_event_table = dict(zip(sub_event_type,list(range(len(sub_event_type)))))

# date
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



def comp_deg_norm(g):
    # print(g.in_degrees())
    in_deg = g.in_degrees(range(g.num_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0, as_tuple=False).view(-1)] = 1
    norm = 1.0 / in_deg  
    return norm


# TODO
# do I have to save quadruples.txt file?????

def build_dglgraphs():
    graph_list = []
    time_list = []
    for k in date_table:
        date_id = date_table[k]
        time_list.append(date_id)

        curr_df = df.loc[df['event_date'] == k]
        src = curr_df['actor1']
        dst = curr_df['actor2']
        rel = curr_df['sub_event_type']
        edge_id = curr_df['data_id']
        uniq_v, edges = np.unique((src, dst), return_inverse=True)  
        src, dst = np.reshape(edges, (2, -1))
        g = dgl.graph((src,dst))

        norm = comp_deg_norm(g)
        g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)}) #TODO
        g.edata['r'] = torch.LongTensor(rel).view(-1, 1)
        g.edata['eid'] = torch.from_numpy(edge_id).view(-1,1)
        # g.edata['main_r'] = torch.LongTensor(rel).view(-1, 1) 
        # TODO should I store the event ID? for abstracting cases?
        # print(g) 
        g.ids = {}
        idx = 0
        for id in uniq_v:
            g.ids[id] = idx
            idx += 1
        graph_list.append(g)
    save_graphs(path + "dglgraph.bin", graph_list, {"time":torch.Tensor(time_list).int()})

    # other information like label, text can be process later.

'''
def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)

def build_graphs(total_data):
    _, _, _, tim = total_data.transpose()
    code_idx_map, idx_main_map = get_event_map()

    total_times = np.unique(tim)
    graph_list = []
    graph_time = []
    graph_label = []# what is the target? main category of events # outbreak? greater than average?
    graph_sublabel = []
    graph_label_dup = []
    graph_sublabel_dup = []
    # graph_labels = {"glabel": th.tensor([0, 1])}
    # save_graphs("./data.bin", [g1, g2], graph_labels)
    print('total_times',total_times)
    for time in total_times:
        if time % 500 == 0:
            print('time',time)
        # if time > 100:
        #     break
        data = get_data_with_t(total_data, time)
        edge_indices = get_indices_with_t(total_data, time) # same order

        # print('time=',time,len(data))
        src, rel, dst = data.transpose()
        # print(rel)
        label_dup = [idx_main_map[v]-1 for v in rel]
        sublabel_dup = list(rel)
        label = [v-1 for v in sorted(list(set(label_dup)))]
        sublabel = sorted(list(set(rel)))
        # print('label',label)
        # print('sublabel',sublabel)
        # exit()
        # label = np.zeros(20)
        # main_cnt = collections.Counter(main_rel)
        # for k in main_cnt:
        #     label[k-1] = main_cnt[k]
        # sub rel
        # print(label)
        # exit()
        # print(src[:10],dst[:10])
        uniq_v, edges = np.unique((src, dst), return_inverse=True)  
        src, dst = np.reshape(edges, (2, -1))
        g = dgl.graph((src,dst))
        # g = dgl.graph()
        # g.add_nodes(len(uniq_v))
        # g.add_edges(src, dst, {'t': torch.from_numpy(tim)}) # array list
        norm = comp_deg_norm(g)
        # print('===========',len(uniq_v),len(norm),g.num_nodes(),g.num_edges())
        g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
        g.edata['r'] = torch.LongTensor(rel).view(-1, 1)
        g.edata['eid'] = torch.from_numpy(edge_indices).view(-1,1)

        # g.edata['main_r'] = torch.LongTensor(rel).view(-1, 1) 
        # TODO should I store the event ID? for abstracting cases?
        # print(g) 
        g.ids = {}
        idx = 0
        for id in uniq_v:
            g.ids[id] = idx
            idx += 1
        # graph_dict[time] = g
        graph_time.append(time)
        graph_list.append(g)
        graph_label.append(label)
        graph_sublabel.append(sublabel)
        graph_label_dup.append(label_dup)
        graph_sublabel_dup.append(sublabel_dup)
        # print(g,g.ids)
        # return g

    # graph_labels = {"label": torch.Tensor(graph_label).int(),
    #                 "sublabel": torch.Tensor(graph_sublabel).int(),
    #                 "time":torch.Tensor(graph_time).int()} 
    # save_graphs(path+"data.bin", graph_list, graph_labels)
    save_graphs(path + "data.bin", graph_list, {"time":torch.Tensor(graph_time).int()})
    print(path + 'data.bin', 'saved!')
    label_dict = {"label": graph_label, 
                "sublabel": graph_sublabel, 
                "label_dup":graph_label_dup, 
                "sublabel_dup":graph_sublabel_dup}
    with open(path + 'label.pkl','wb') as f:
        pickle.dump(label_dict, f)
    print(path + 'label.pkl', 'saved!')
'''