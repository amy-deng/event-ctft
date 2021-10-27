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
python build_causal_samples_w_time.py ../data THA_topic check_topic_causal_data_w7h7

'''
try:
    out_path = sys.argv[1]
    dataset = sys.argv[2] # THA_topic
    raw_data_name = sys.argv[3]  
except:
    print("usage: <out_path> <dataset `THA_topic`> <raw_data_name `raw_w10h7`> ")
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
splitted_date_lists = [
    '2010-07-01',
    '2011-01-01','2011-07-01','2012-01-01','2012-07-01','2013-01-01','2013-07-01',
    '2014-01-01','2014-07-01','2015-01-01','2015-07-01','2016-01-01','2016-07-01',
    '2017-01-01','2017-07-01'
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

sorted_indices = np.argsort(dates)

samples_by_time = {}
cur_time_split_idx = 0
for topic_id in range(2):
    treatment_assign_by_topic = []
    covariate_by_topic = []
    outcome_by_topic = []

    out_file = "{}/topic_{}.pkl".format(save_path,topic_id)

    for i in sorted_indices:

        date = dates[i]

        if date > splitted_date_lists[cur_time_split_idx]:
            
            out_path = "{}/topic_{}_{}.pkl".format(save_path,topic_id,splitted_date_lists[cur_time_split_idx])
            save_samples(treatment_assign_by_topic,outcome_by_topic,covariate_by_topic,out_path)
            cur_time_split_idx += 1
            
        curr_treament = treatment[i]
        past_treatment = treatment_check[i]
        if past_treatment[topic_id] <= 0:
            if curr_treament[topic_id] > 0: # treated sample
                treatment_assign_by_topic.append(1)
            else: # control sample
                treatment_assign_by_topic.append(0)
            covariate_by_topic.append(covariate[i])
            outcome_by_topic.append(outcome[i])
        else:
            pass
    
    out_path = "{}/topic_{}_{}.pkl".format(save_path,topic_id,splitted_date_lists[cur_time_split_idx])
    save_samples(treatment_assign_by_topic,outcome_by_topic,covariate_by_topic,out_path)

        