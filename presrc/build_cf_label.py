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
    print("Usage: DATASET, TMPFILE, TREAT=24")
    exit()

tmp = TMPFILE.split('_')
WINDOW = int(tmp[2][1:])
HORIZON = int(tmp[3][1:])
PREWINDOW = int(tmp[4][1:2])
print('WINDOW={}  HORIZON={}  PREWINDOW={}'.format(WINDOW,HORIZON,PREWINDOW))
path = '../data/{}/'.format(DATASET)

with open(path+TMPFILE,'rb') as f:
    [data_time,data_Y,data_X_smooth,data_treat,tfidf] = pickle.load(f)

print('data_time',type(data_time))
print('data_Y',type(data_Y))
print('data_X_smooth',type(data_X_smooth))
print('data_treat',type(data_treat))
print('tfidf',type(tfidf))


# TODO
# for each sample, find a cf sample, different treatment, see what is the outcome
 
