import pandas as pd
import numpy as np
import sys, os, json, time, collections, pickle
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

import scipy
'''
python PSM_w_time_new.py ../data THA_topic check_topic_causal_data_w14h14_from2013_minprob0.05 14 1 0
for each event find causes
'''
# out_path='../data'
# dataset_name='THA_topic'
# raw_data_name='check_topic_causal_data_w14h14_from2013_minprob0.05'
# pred_window=14
# target_binary=1
# check=0
try:
    # event_path = sys.argv[1] # /home/sdeng/data/icews/detailed_event_json/THA_2010_w14h7_city.json
    out_path = sys.argv[1]
    dataset_name = sys.argv[2] # THA_topic
    raw_data_name = sys.argv[3] 
    pred_window = int(sys.argv[4])
    target_binary = int(sys.argv[5])
    check = int(sys.argv[6])
    # event_code = int(sys.argv[4])
except:
    print("usage: <out_path> <dataset_name `THA_topic`> <raw_data_name `check_topic_causal_data_w7h7`> <pred_window 5> <target_binary 0> <check 1/0>")
    exit()

if check == 1:
    file_list = glob.glob('{}/{}/{}/check_topic*.pkl'.format(out_path, dataset_name, raw_data_name))
else:
    file_list = glob.glob('{}/{}/{}/nocheck_topic*.pkl'.format(out_path, dataset_name, raw_data_name))


splitted_date_lists = [
    '2013-01-01','2013-04-01','2013-07-01','2013-10-01',
    '2014-01-01','2014-04-01','2014-07-01','2014-10-01',
    '2015-01-01','2015-04-01','2015-07-01','2015-10-01',
    '2016-01-01','2016-04-01','2016-07-01','2016-10-01',
    '2017-01-01','2017-04-01'
]
save_path = '{}/{}/{}/causal_effect'.format(out_path, dataset_name, raw_data_name)
os.makedirs(save_path, exist_ok=True)


date_collection={}
for t in range(4,len(splitted_date_lists)): 
    tmp = splitted_date_lists[t-4:t]
    # print(tmp,len(tmp))
    l = []
    for j in range(1,len(tmp)):
        l.append('{}_{}'.format(tmp[j-1],tmp[j]))
    l.append('{}_{}'.format(tmp[-1],splitted_date_lists[t]))
    date_collection[splitted_date_lists[t]] = l

effect_dict = {}


for topic_id in range(50):
    print('loading topic {} data'.format(topic_id))
    for k in date_collection:
        treatment_list = []
        covariate_list = []
        outcome_list = []
        for tim in date_collection[k]:
            file_name = '{}/{}/{}/nocheck_topic_{}_{}.pkl'.format(out_path, dataset_name, raw_data_name, topic_id, tim)
            try:
                with open(file_name,'rb') as f:
                    dataset = pickle.load(f)
                treatment = dataset['treatment']
                treatment = np.where(treatment > 0, 1, 0)
                covariate = dataset['covariate']
                covariate = np.concatenate([v.toarray() for v in covariate],0)
                outcome = dataset['outcome'][:,:pred_window,].sum(1) 
                # print(type(treatment),len(treatment),'treatment',treatment.shape)
                # print(type(covariate),len(covariate),'covariate',covariate.shape)
                # print(type(outcome),len(outcome),'outcome',outcome.shape)
                treatment_list.append(treatment)
                covariate_list.append(covariate)
                outcome_list.append(outcome)
            except:
                print(file_name)
            # break
        if len(treatment_list) < 30:
            ATE = np.zeros(20)
            effect_dict[(int(topic_id),k)] = ATE
            print('empty')
            continue
        treatment_list = np.concatenate(treatment_list,0)
        covariate_list = np.concatenate(covariate_list,0)
        outcome_list = np.concatenate(outcome_list,0)
        print(treatment_list.shape,covariate_list.shape,outcome_list.shape)
        if target_binary == 1:
            print('Convert outcome to binary')
            outcome_list = np.where(outcome_list > 0, 1, 0)
        treatment = treatment_list
        covariate = covariate_list
        outcome = outcome_list
        scaler = StandardScaler()
        X = scaler.fit_transform(covariate)
        cls = LogisticRegression(random_state=42,max_iter=2000)
        cls = CalibratedClassifierCV(cls)
        cls.fit(X, treatment)
        print('propensity scoring model trained')
        propensity = cls.predict_proba(covariate)
        propensity = propensity[:,1]
        # caliper = propensity.std()*0.2
        propensity_logit = scipy.special.logit(propensity)
        caliper = propensity_logit.std()* 0.2
        # get pairs and calculate average treatment effect 
        # for each treatment ele, find a control, most similar
        controlled_indices = np.where(treatment == 0)[0]
        treatment_idices = treatment.nonzero()[0]
        np.random.shuffle(treatment_idices)
        # treatment_idices
        eff_list = [] 
        used_control_indices = []
        n_pairs = 0
        for i in treatment_idices:
            curr = propensity_logit[controlled_indices]
            diff = np.abs(curr-propensity_logit[i])
            min_idx = np.argmin(diff, axis=0)
            min_diff = diff[min_idx]
            if min_diff < caliper:
                # get treatment effect?
                outcome_control = outcome[controlled_indices[min_idx]]
                outcome_treatment = outcome[i]
                eff = outcome_treatment-outcome_control
                eff_list.append(eff) 
                n_pairs += 1
                used_control_indices.append(controlled_indices[min_idx])
            else:
                print('min diff is larger than the caliper {:.5f}; skip'.format(caliper))
        eff_list = np.stack(eff_list,0) 
        ATE = eff_list.mean(0)
        print('eff_list',eff_list.shape,'ATE',ATE.shape)
        effect_dict[(int(topic_id),k)] = ATE

outfile = '{}/effect_dict_pw{}_biy{}_nocheck.pkl'.format(save_path,pred_window,target_binary)
with open(outfile,'wb') as f:
    pickle.dump(effect_dict,f)
print(outfile,' saved')
 

