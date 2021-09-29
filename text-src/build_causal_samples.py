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
python build_causal_data.py /home/sdeng/data/icews/detailed_event_json/THA_2010_w14h7_city.json ../data 10 7 THA_50 /home/sdeng/data/icews/corpus/ngrams/THA_1gram_tfidf.txt 15000

'''
try:
    # event_path = sys.argv[1] # /home/sdeng/data/icews/detailed_event_json/THA_2010_w14h7_city.json
    out_path = sys.argv[1]
    dataset = sys.argv[2] # THA_topic
    raw_data_name = sys.argv[3] 
    # start_year = int(sys.argv[3])
    # end_year = int(sys.argv[4])
    # window = int(sys.argv[3])
    # horizon = int(sys.argv[4])
    # lda_name = sys.argv[5]
    # ngram_path = sys.argv[6]
    # top_k_ngram = int(sys.argv[7])
except:
    print("usage: <event_path> <out_path> <window <=13 > <horizon <=7 > <lda_name `THA_50`> <ngram_path> <top_k_ngram `16000`>")
    exit()


os.makedirs('{}/{}/{}'.format(out_path,dataset,), exist_ok=True)


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
    out_file = "{}_topic_{}.pkl".format(raw_data_name.split('_')[1],topic_id)
    print('out_file',out_file)
    for i in range(len(treatment)):
        curr_treament = treatment[i]
        past_treatment = treatment_check[i]
#         print(curr_treament.shape,past_treatment.shape,past_treatment)
        if past_treatment[topic_id] <= 0:
            if curr_treament[topic_id] > 0: # treated sample
#                 print('treatment',topic_id,i,'yes',curr_treament[topic_id], past_treatment[topic_id])
                treatment_assign_by_topic.append(1)
            else: # control sample
#                 print('controlled')
                treatment_assign_by_topic.append(0)
            covariate_by_topic.append(covariate[i])
            outcome_by_topic.append(outcome[i])
        else:
            pass
    if len(treatment_assign_by_topic) <= 0:
        continue
    treatment_assign_by_topic = np.array(treatment_assign_by_topic)
    outcome_by_topic = np.stack(outcome_by_topic,0)
    print('treatment_assign_by_topic',treatment_assign_by_topic.shape,'outcome_by_topic',outcome_by_topic.shape)
    with open(out_file,'wb') as f:
        pickle.dump({'treatment':treatment_assign_by_topic,
                    'covariate':covariate_by_topic,
                    'outcome':outcome_by_topic},f)
    
    print('treatment_assign_by_topic',treatment_assign_by_topic.mean())
 
 