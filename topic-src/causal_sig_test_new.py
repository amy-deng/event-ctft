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
python causal_sig_test.py ../data THA_topic check_topic_causal_data_w7h7 effect_dict_pw5_biy0 0.05
python causal_sig_test.py ../data THA_topic check_topic_causal_data_w7h14 effect_dict_pw14_biy1_nocheck 0.1
python causal_sig_test.py ../data THA_topic check_topic_causal_data_w7h14 effect_dict_pw7_biy1_nocheck 0.1
python causal_sig_test.py ../data THA_topic check_topic_causal_data_w7h14 effect_dict_pw3_biy1_nocheck 0.1

python causal_sig_test.py ../data THA_topic check_topic_causal_data_w7h14 effect_dict_pw14_biy1 0.
python causal_sig_test.py ../data THA_topic check_topic_causal_data_w7h14 effect_dict_pw7_biy1 0.1
python causal_sig_test.py ../data THA_topic check_topic_causal_data_w7h14 effect_dict_pw3_biy1 0.1


python causal_sig_test.py ../data THA_topic check_topic_causal_data_w7h14 effect_dict_pw14_biy1 0.05
python causal_sig_test.py ../data THA_topic check_topic_causal_data_w7h14 effect_dict_pw7_biy1 0.05
python causal_sig_test.py ../data THA_topic check_topic_causal_data_w14h14_from2013_minprob0.05 effect_dict_pw3714_biy1_nocheck 0.05
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

file_path = "{}/{}/{}/causal_effect/{}.pkl".format(out_path,dataset_name,raw_data_name,effect_dict_name)
print(file_path)
with open(file_path,'rb') as f:
    effect_dict = pickle.load(f)

print(effect_dict.keys())
# for each type of events find significant causes topic >0.01?
# exit()

keys = effect_dict.keys() # for each time
 
splitted_date_lists = [
    '2013-01-01','2013-04-01','2013-07-01','2013-10-01',
    '2014-01-01','2014-04-01','2014-07-01','2014-10-01',
    '2015-01-01','2015-04-01','2015-07-01','2015-10-01',
    '2016-01-01','2016-04-01','2016-07-01','2016-10-01',
    '2017-01-01','2017-04-01'
]

event_types = ['statement', 'appeal','express cooperate','consult','diplomatic cooperation','material cooperation','provide aid','yield','investigate','demand','disapprove','reject','threaten','protest','minitary','reduce relation','coerce','assault','fight','mass violence']

ignored_key = []
for end_date in splitted_date_lists:
    res = [] # res is a list
    res3 = []
    res7 = []
    res14 = []
    for topic in range(50):
        key = (topic,end_date)
        if key not in effect_dict:
            print('key',key)
            ignored_key.append(key)
            # exit()
            tmp3 = np.zeros(20) # remember this topic, TODO should not be causal
            tmp7 = np.zeros(20)
            tmp14 = np.zeros(20)
        else:
            [tmp3, tmp7, tmp14] = effect_dict[key]
        # print(tmp.shape)
        res3.append(tmp3)
        res7.append(tmp7)
        res14.append(tmp14)
    res3 = np.stack(res3)
    res7 = np.stack(res7)
    res14 = np.stack(res14)

    # print('end_date={} \t res {}'.format(end_date,res.shape))
    res_dict = {'3':res3, '7':res7, '14':res14}
    for h in res_dict:
        res = res_dict[h]
        f = open('{}/{}/{}/causal_effect/{}_{}_{}.csv'.format(out_path, dataset_name, raw_data_name,effect_dict_name,sig_level,h),'a')
        wrt = csv.writer(f)
        wrt.writerow(["event-idx", "event-type", 'rank', "topic-id","effect","z-score","p-value","end-date"])

        for j in range(20):
            # z_list.append(stats.zscore(res[:,j]))
            effect = res[:,j]
            try:
                z_scores = stats.zscore(effect)
            except:
                print(effect)
                exit()
            # print('z_scores',z_scores.shape)
            # p_values = scipy.stats.norm.cdf(z_scores)
            p_values = scipy.stats.norm.sf(abs(z_scores))*2
            # print(p_values,'p_values')
            sorted_idx = np.argsort(p_values)
            # print(sorted_idx,'sorted_idx')
            sig_idx = np.where(p_values<sig_level,1,0)
            # print(sig_idx,'sig_idx')
            # print(np.nonzero(sig_idx),'np.nonzero(sig_idx)')
            len_nonzero = len(np.nonzero(sig_idx)[0])
            # print(len_nonzero,'len_nonzero')
            topic_idx = sorted_idx[:len_nonzero]
            # print(topic_idx,'topic_idx')
            top_p = p_values[topic_idx]
            # print(top_p,'top_p')
            top_z = z_scores[topic_idx]
            # print(top_z,'top_z')
            top_effect = effect[topic_idx]
            # print(top_effect,'top_effect')
            # print(event_types[j],len_nonzero,top_p,topic_idx)
            for i in range(len(topic_idx)):
                if (topic_idx[i],end_date) in ignored_key:
                    print('error',(topic_idx[i],end_date))
                r = [j,event_types[j],i,topic_idx[i],round(top_effect[i],5),round(top_z[i],5),round(top_p[i],5),end_date]
                # print(r)
                wrt.writerow(r)

        f.close()
print('ignored_key',ignored_key)
    # res


# with open('{}/{}/{}/plot/significance.csv'.format(out_path, dataset_name, raw_data_name), 'w', newline='') as outcsv:
#     writer = csv.writer(outcsv)
#     writer.writerow(["event-idx", "event-type", 'rank', "topic-id","p-value"])

# finally decide to use this one [effect_dict_pw7_biy1_0.05.csv]. 
# binary outcome remove the noise that if repeated events are reforted 
# pred_window=7, if historical window=7,then causes still count