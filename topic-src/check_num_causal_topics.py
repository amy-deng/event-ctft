import numpy as np
import pickle
import torch
from torch.utils import data
# import utils
import pickle
import pandas as pd
import collections

datafile_list = ['dyn_tf_2014-2015_900','dyn_tf_2015-2016_900','dyn_tf_2016-2017_900']
path = '../data'
dataset = 'RUS_w7h7_mind3n10df0.01'
dataset = 'THA_w7h7_mind3n7df0.01'
dataset = 'AFG_w7h7_mind3n7df0.01'

g_data = []
t_data = []
for datafile in datafile_list:
    with open('{}/{}/{}.pkl'.format(path, dataset,datafile),'rb') as f:
        graph_list = pickle.load(f)
    g_data += graph_list
    tmp = datafile.split('_')
    tmp[0] = 'attr'
    attr_file = '_'.join(tmp)
    with open('{}/{}/{}.pkl'.format(path, dataset, attr_file),'rb') as f:
        attr_dict = pickle.load(f)
    t_data +=  attr_dict['date']


splitted_date_lists = [
'2013-01-01','2013-04-01','2013-07-01','2013-10-01',
'2014-01-01','2014-04-01','2014-07-01','2014-10-01',
'2015-01-01','2015-04-01','2015-07-01','2015-10-01',
'2016-01-01','2016-04-01','2016-07-01','2016-10-01',
'2017-01-01','2017-04-01'
]
causalfiles = 'causal_topics_0.01'
with open('{}/{}/{}.pkl'.format(path, dataset,causalfiles),'rb') as f:
    causal_time_dict = pickle.load(f)

pos_list = []
neg_list = []
for index in range(len(g_data)):
    g = g_data[index]
    date = t_data[index]
    for end_date in splitted_date_lists: # check date in which range
        if date < end_date:
            cur_end_date = end_date
            break
    causal_weight = causal_time_dict[cur_end_date]
    causal_weight_tensor = torch.from_numpy(causal_weight)#.to_sparse() 
    effect = causal_weight_tensor[g.nodes('topic').numpy()]
    effect = (effect >0)*1. + (effect < 0)*(-1.)
    effect = effect.sum(-1)
    pos = (effect>0).sum()
    neg = (effect<0).sum()
    pos_list.append(pos)
    neg_list.append(neg)
pos_list = np.array(pos_list)
neg_list = np.array(neg_list)
print(pos_list.mean(),neg_list.mean())