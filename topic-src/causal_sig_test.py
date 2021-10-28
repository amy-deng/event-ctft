import pandas as pd
import numpy as np
import sys, os, json, time, collections, pickle
import glob
# from sklearn.linear_model import LogisticRegression
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.preprocessing import StandardScaler
# from matplotlib import pyplot as plt
import pickle
import scipy
import csv
from scipy import stats
'''
python causal_sig_test.py ../data THA_topic check_topic_causal_data_w7h7 effect_dict_pw5_biy0
for each event find causes
'''
try:
    out_path = sys.argv[1]
    dataset_name = sys.argv[2] # THA_topic
    raw_data_name = sys.argv[3] 
    effect_dict_name = sys.argv[4] 
    sig_level = float(sys.argv[5])
except:
    print("usage: <out_path> <dataset_name `THA_topic`> <raw_data_name `check_topic_causal_data_w7h7`> <effect_dict_name> <sig_level 0.05 0.01>")
    exit()

file_path = "{}/{}/{}/causal_effect/{}_.pkl".format(out_path,dataset_name,raw_data_name,effect_dict_name,sig_level)

with open(file_path,'rb') as f:
    effect_dict = pickle.load(f)


# for each type of events find significant causes topic >0.01?


keys = effect_dict.keys() # for each time
splitted_date_lists = [
    '2010-07-01',
    '2011-01-01','2011-07-01','2012-01-01','2012-07-01','2013-01-01','2013-07-01',
    '2014-01-01','2014-07-01','2015-01-01','2015-07-01','2016-01-01','2016-07-01',
    '2017-01-01','2017-07-01'
]

event_types = ['statement', 'appeal','express cooperate','consult','diplomatic cooperation','material cooperation','provide aid','yield','investigate','demand','disapprove','reject','threaten','protest','minitary','reduce relation','coerce','assault','fight','mass violence']

f = open('{}/{}/{}/causal_effect/{}.csv'.format(out_path, dataset_name, raw_data_name,effect_dict_name),'a')
wrt = csv.writer(f)
wrt.writerow(["event-idx", "event-type", 'rank', "topic-id","effect","z-score","p-value","end-date"])

for end_date in splitted_date_lists:
    res = []
    for topic in range(50):
        key = (topic,end_date)
        if key not in effect_dict:
            print('key',key)
            # exit()
            tmp = np.zeros(20)
        else:
            tmp = effect_dict[key]
        # print(tmp.shape)
        res.append(tmp)
    res = np.stack(res)

    # print('end_date={} \t res {}'.format(end_date,res.shape))

    # z_list = []
    for j in range(20):
        # z_list.append(stats.zscore(res[:,j]))
        effect = res[:,j]
        z_scores = stats.zscore(effect)
        # print('z_scores',z_scores.shape)
        # p_values = scipy.stats.norm.cdf(z_scores)
        p_values = scipy.stats.norm.sf(abs(z_scores))*2
        sorted_idx = np.argsort(p_values)
        sig_idx = np.where(p_values<sig_level,1,0)
        len_nonzero = len(np.nonzero(sig_idx)[0])
        topic_idx = sorted_idx[:len_nonzero]
        top_p = p_values[topic_idx]
        top_z = z_scores[topic_idx]
        top_effect = effect[topic_idx]
        # print(event_types[j],len_nonzero,top_p,topic_idx)
        for i in range(len(topic_idx)):
            r = [j,event_types[j],i,topic_idx[i],round(top_effect[i],5),round(top_z[i],5),round(top_p[i],5),end_date]
            # print(r)
            wrt.writerow(r)

f.close()

    # res


# with open('{}/{}/{}/plot/significance.csv'.format(out_path, dataset_name, raw_data_name), 'w', newline='') as outcsv:
#     writer = csv.writer(outcsv)
#     writer.writerow(["event-idx", "event-type", 'rank', "topic-id","p-value"])

