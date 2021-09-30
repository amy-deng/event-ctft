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
python build_causal_samples.py ../data THA_topic raw_w7h7

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
covariate = data['covariate'] # scipy.sparse.csr.csr_matrix

# all treatment-outcome pair
 
for topic_id in range(50):
    treatment_assign_by_topic = []
    covariate_by_topic = []
    outcome_by_topic = []
    out_file = "{}/topic_{}.pkl".format(save_path,topic_id)
#     print('out_file',out_file)
    for i in range(len(treatment)):
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
    treatment_assign_by_topic = np.array(treatment_assign_by_topic)
    n_smaple = len(treatment_assign_by_topic)
    n_treat =len(treatment_assign_by_topic.nonzero()[0])
    n_control = n_smaple-n_treat
    print('topic_id',topic_id,'n_treat =',n_treat,'n_control =',n_control,round(treatment_assign_by_topic.mean(),4))
    if n_treat < 30 or n_control < 30:
        continue
        
    outcome_by_topic = np.stack(outcome_by_topic,0)
    print('n_smaple',n_smaple,'outcome_by_topic',outcome_by_topic.shape)
    with open(out_file,'wb') as f:
        pickle.dump({'treatment':treatment_assign_by_topic,
                    'covariate':covariate_by_topic,
                    'outcome':outcome_by_topic},f)
 
