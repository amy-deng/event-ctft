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
from scipy import sparse

#### build datasets
### TODO

# only select samples that have enough historical news, different locations
# {
#     'date':'',# future events
#     'story_id': '',# historical news'
#     'event_dict':
#     'city':
#     'state':
# }

# use data generated from build_detailed_event_json_data.py
# 
try:
    event_path = sys.argv[1] # /home/sdeng/data/icews/detailed_event_json/THA_2010_w14h7_city.json
    out_path = sys.argv[2]
    # dataset = sys.argv[4] # THA
    # start_year = int(sys.argv[3])
    # end_year = int(sys.argv[4])
    window = int(sys.argv[3])
    horizon = int(sys.argv[4])
    lda_name = sys.argv[5]
    ngram_path = sys.argv[6]
    top_k_ngram = int(sys.argv[7])
except:
    print("usage: <event_path> <out_path> <window <=13 > <horizon <=7 > <lda_name `THA_50`> <ngram_path> <top_k_ngram `16000`>")
    exit()

country = event_path.split('/')[-1][:3]
dataset = country + '_topic'
dataset_path = "{}/{}".format(out_path,dataset)
os.makedirs(dataset_path, exist_ok=True)
print('dataset_path',dataset_path)

out_file = "raw_w{}h{}.pkl".format(window,horizon)
print('out_file',out_file)

df = pd.read_json(event_path,lines=True)

news_df = pd.read_json('/home/sdeng/data/icews/news.1991.201703.country/icews_news_{}.json'.format(country), lines=True)

loaded_dict = corpora.Dictionary.load('/home/sdeng/data/icews/topic_models/{}.dict'.format(country))
loaded_lda =  models.LdaModel.load('/home/sdeng/data/icews/topic_models/{}.lda'.format(lda_name))
print('topic model and dictionary loaded')


event_path = '/home/sdeng/data/icews/detailed_event_json/THA_2010_w14h7_city.json'
df = pd.read_json(event_path,lines=True)

# at lease 1 day?