# import glob
import sys
import numpy as np
import pandas as pd
import os
import time
import pickle
import collections
import dgl
from scipy import stats
from dgl.data.utils import save_graphs,load_graphs
import torch 
from datetime import date,timedelta
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial


# from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS, original_scorer
# import nltk
# import re
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import wordnet
# from nltk.corpus import stopwords


'''
generate counterfactual labels
'''
try:
    # RAWDATA = sys.argv[1]
    # # DATASET = sys.argv[1]
    # # STARTTIME = str(sys.argv[2])
    # # ENDTIME= str(sys.argv[3])
    # # DELTA = int(sys.argv[2])
    # WINDOW = int(sys.argv[2])
    # HORIZON = int(sys.argv[3])
    # PREDWINDOW = int(sys.argv[4])
    
    DATASET = sys.argv[1]
    TMPFILE = sys.argv[2]
    TREAT = int(sys.argv[3]) # following subevent2id.txt
except:
    print("Usage: DATASET, TMPFILE, TREAT=24(23 AFG)")
    exit()

tmp = TMPFILE.split('_')
WINDOW = int(tmp[2][1:])
HORIZON = int(tmp[3][1:])
PREWINDOW = int(tmp[4][1:2])
print('WINDOW={}  HORIZON={}  PREWINDOW={}'.format(WINDOW,HORIZON,PREWINDOW))
path = '../data/{}/'.format(DATASET)

with open(path+TMPFILE,'rb') as f:
    [data_time,data_Y,data_X_smooth,data_treat,tfidf] = pickle.load(f)

print('data_time',type(data_time),len(data_time))
print('data_Y',type(data_Y),len(data_Y))
print('data_X_smooth',type(data_X_smooth),data_X_smooth.shape)
print('data_treat',type(data_treat),data_treat.shape)
print('tfidf',type(tfidf),tfidf.shape)


# TODO assume it is not biased
# for each sample, find a cf sample, different treatment, see what is the outcome
data_Y_cf_all = []
for treat_id in range(data_treat.shape[-1]):
    treatment = data_treat[:,treat_id] 
    data_X_smooth_flat = data_X_smooth.reshape(data_X_smooth.shape[0],-1)
    data_Y_cf = []
    for i in range(len(data_Y)):
        y = data_Y[i]
        t = treatment[i]

        x = data_X_smooth_flat[i] 
        s = tfidf[i]
        # find samples with treatment 1-t
        sel_idx_list = (treatment!=t).nonzero()[0]
        sel_idx = None
        sel_idx_similarity_score = float('-inf')
        for j in sel_idx_list:
            can_x = data_X_smooth_flat[j]
            can_s = tfidf[j]
            v, p = stats.pearsonr(x, can_x)
            cos_sim = 1 - spatial.distance.cosine(s, can_s)
            if cos_sim + abs(v) > sel_idx_similarity_score:
                sel_idx_similarity_score = cos_sim + abs(v) 
                sel_idx = j
        # get cf label
        data_Y_cf.append(data_Y[sel_idx])
    data_Y_cf_all.append(data_Y_cf)

data_treat_cf = 1 - data_treat
# treatment_cf = 1 -  treatment
data_Y_cf_all = np.array(data_Y_cf_all)
data_Y_cf_all = np.swapaxes(data_Y_cf_all, 0,1)
print('data_Y_cf_all',data_Y_cf_all.shape, 'data_treat_cf',data_treat_cf.shape)

with open(path+'cf_data.pkl','wb') as f:
    pickle.dump({'data_time':data_time,'Y':data_Y,'C':data_treat,'CF_Y':data_Y_cf_all, 'CF_C':data_treat_cf},f)
print(path+'cf_data.pkl', 'saved!')

