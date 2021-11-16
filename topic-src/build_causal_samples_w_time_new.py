import pandas as pd
import numpy as np
import sys, os, json, time, collections, pickle
# import glob
from gensim import corpora, models, similarities
from gensim.models.ldamulticore import LdaMulticore,LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_corpus, common_dictionary
from text_utils import *
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

'''
python build_causal_samples_w_time_new.py ../data THA_topic check_topic_causal_data_w7h7
python build_causal_samples_w_time_new.py ../data THA_topic check_topic_causal_data_w7h14 '' '' 0

'''
try:
    out_path = sys.argv[1]
    dataset = sys.argv[2] # THA_topic
    raw_data_name = sys.argv[3]  
    start_date = sys.argv[4]
    end_date = sys.argv[5]
    check = int(sys.argv[6])
except:
    print("usage: <out_path> <dataset `THA_topic`> <raw_data_name `raw_w10h7`> <start_date> <end_date> <no_check check 1/0>")
    exit()


save_path = '{}/{}/{}'.format(out_path,dataset,raw_data_name)
os.makedirs(save_path, exist_ok=True)


with open('{}/{}/{}.pkl'.format(out_path,dataset,raw_data_name),'rb') as f:
    data = pickle.load(f)
treatment = data['treatment'] # np.array
outcome = data['outcome'] # np.array
treatment_check = data['treatment_check'] # np.array
covariate = data['covariate'] # list of scipy.sparse.csr.csr_matrix
dates = data['date'] # np.array

 
# 2017-01-01
# splitted_date_lists = [
#     '2010-07-01',
#     '2011-01-01','2011-07-01','2012-01-01','2012-07-01','2013-01-01','2013-07-01',
#     '2014-01-01','2014-07-01','2015-01-01','2015-07-01','2016-01-01','2016-07-01',
#     '2017-01-01','2017-07-01'
# ]

splitted_date_lists = [
    '2013-01-01','2013-04-01','2013-07-01','2013-10-01',
    '2014-01-01','2014-04-01','2014-07-01','2014-10-01',
    '2015-01-01','2015-04-01','2015-07-01','2015-10-01',
    '2016-01-01','2016-04-01','2016-07-01','2016-10-01',
    '2017-01-01','2017-04-01'
]


def save_samples(treatment, outcome, covariate, outpath):
    treatment = np.array(treatment)
    n_smaple = len(treatment)
    n_treat =len(treatment.nonzero()[0])
    n_control = n_smaple-n_treat
    print('topic_id',topic_id,'n_treat =',n_treat,'n_control =',n_control,round(treatment.mean(),4))
    if n_treat < 30 or n_control < 30:
        print('n_treat',n_treat,'n_control',n_control,'skip')
        return
        
    outcome = np.stack(outcome,0)
    # print('n_smaple',n_smaple,'outcome',outcome.shape)
    # out_file = "{}/topic_{}_{}.pkl".format(save_path,topic_id,splitted_date_lists[cur_time_split_idx])
    with open(outpath,'wb') as f:
        pickle.dump({'treatment':treatment,
                    'covariate':covariate,
                    'outcome':outcome},f)
    return

# sorted_indices = np.argsort(dates)

samples_by_time = {}

# save samples for each time intervals

for topic_id in range(50):
    treatment_assign_by_topic = []
    covariate_by_topic = []
    outcome_by_topic = []

    for t in range(1,len(splitted_date_lists)):
        start = splitted_date_lists[t-1]
        end = splitted_date_lists[t]
        
        treatment_assign_by_topic = []
        covariate_by_topic = []
        outcome_by_topic = []
        for i in range(len(dates)):
            if dates[i] >= start and dates[i] < end:
                curr_treament = treatment[i]
                past_treatment = treatment_check[i]
                if check == 0:
                    if curr_treament[topic_id] > 0: # treated sample
                        treatment_assign_by_topic.append(1)
                    else: # control sample
                        treatment_assign_by_topic.append(0)
                    covariate_by_topic.append(covariate[i])
                    outcome_by_topic.append(outcome[i])
                else:
                    if past_treatment[topic_id] <= 0:
                        if curr_treament[topic_id] > 0: # treated sample
                            treatment_assign_by_topic.append(1)
                        else: # control sample
                            treatment_assign_by_topic.append(0)
                        covariate_by_topic.append(covariate[i])
                        outcome_by_topic.append(outcome[i])
                    else:
                        pass
        print(start,'-',end, len(treatment_assign_by_topic),'treatment_assign_by_topic')
        if check == 0: 
            out_path = "{}/nocheck_topic_{}_{}_{}_{}.pkl".format(save_path,topic_id,start,end)
        else:
            out_path = "{}/check_topic_{}_{}_{}_{}.pkl".format(save_path,topic_id,start,end)
        save_samples(treatment_assign_by_topic,outcome_by_topic,covariate_by_topic,out_path)

#         pass
# for topic_id in range(50):
#     treatment_assign_by_topic = []
#     covariate_by_topic = []
#     outcome_by_topic = []

#     cur_time_split_idx = 0
#     for i in sorted_indices:

#         date = dates[i]
#         if date < start_date or date >= start_date:
#             continue
#         if date > splitted_date_lists[cur_time_split_idx]:
            
#             out_path = "{}/nocheck_topic_{}_{}.pkl".format(save_path,topic_id,splitted_date_lists[cur_time_split_idx])
#             save_samples(treatment_assign_by_topic,outcome_by_topic,covariate_by_topic,out_path)
#             cur_time_split_idx += 1
            
#         curr_treament = treatment[i]
#         past_treatment = treatment_check[i]
#         # if past_treatment[topic_id] <= 0:
#         if curr_treament[topic_id] > 0: # treated sample
#             treatment_assign_by_topic.append(1)
#         else: # control sample
#             treatment_assign_by_topic.append(0)
#         covariate_by_topic.append(covariate[i])
#         outcome_by_topic.append(outcome[i])
#         # else:
#         #     pass
    
#     out_path = "{}/nocheck_topic_{}_{}_{}_{}.pkl".format(save_path,topic_id,start_date, end_date)
#     save_samples(treatment_assign_by_topic,outcome_by_topic,covariate_by_topic,out_path)

        