import imp
import pandas as pd
import numpy as np
from text_utils import *
from nltk.util import pr
import pandas as pd
import numpy as np
import sys, os, json, time, collections, pickle, math
# import glob
from gensim import corpora, models, similarities
from gensim.models.ldamulticore import LdaMulticore,LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_corpus, common_dictionary
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from scipy import sparse
from scipy.spatial.distance import cdist
import dgl
from dgl.data.utils import save_graphs,load_graphs
from numpy import linalg


news_f = '/home/sdeng/data/icews/THA-2012-2016/icews_news_THA.json'
news_df = pd.read_json(news_f,lines=True)


event_f = '/home/sdeng/data/icews/THA-2012-2016/icews_event_THA.json'
event_df = pd.read_json(event_f,lines=True)

def getRoot(x):
    x = int(x)
    if len(str(x)) == 4: # 1128
        return x // 100
    elif len(str(x)) == 3:
        if x // 10 < 20: # 190
            return x // 10
        else:
            return x // 100
    else:
        return x // 10

event_df['root'] = event_df['CAMEO Code'].apply(lambda x: getRoot(x) )

protests = event_df.loc[event_df['root']==14]
protest_news_ids = protests['Story ID'].unique() 
# get texts
protest_news = news_df.loc[news_df['StoryID'].isin(protest_news_ids)]['Text'].values
protest_date = news_df.loc[news_df['StoryID'].isin(protest_news_ids)]['Date'].values 
tokens_list = clean_document_list(protest_news) # [['thailand', 'coup'],[...]]

lda_name = 'THA_2012_50'
dict_name = '_'.join(lda_name.split('_')[:2])
# dict_name = 'THA_2012'
loaded_dict = corpora.Dictionary.load('/home/sdeng/data/icews/topic_models/{}.dict'.format(dict_name))
loaded_lda =  models.LdaModel.load('/home/sdeng/data/icews/topic_models/{}.lda'.format(lda_name))

corpus_bow = [loaded_dict.doc2bow(text) for text in tokens_list]
topic_dists =  loaded_lda.get_document_topics(corpus_bow,per_word_topics=False,minimum_probability=0.01)

doc_node, topic_node, weight = [], [], []
topic_freq_by_month = []

date_range = pd.date_range(start='1/1/2014', periods=36, freq='M')
date_range_dict = {}
for month in date_range:
    date_range_dict[month.strftime('%Y-%m')] = []
    # print(month.strftime('%Y-%m-%d'),str(month),type(month))


def getTimeKey(date_range,date):
    start = '2013-12-31'
    for month in date_range:
        curr = month.strftime('%Y-%m-%d')
        if start < date < curr:
            return  month.strftime('%Y-%m')
        start = curr


for doc_id in range(len(topic_dists)):
    date = protest_date[doc_id]
    time_key = getTimeKey(date_range,date)
    topic_weights = topic_dists[doc_id]
    # print('topic_weights',topic_weights)
    # break
    tmp = []
    for t,w in topic_weights:
        # doc_node.append(doc_id)
        tmp.append(t)
        if len(tmp) == 3:
            break
        # weight.append(w)
    # if time_key in date_range_dict:
    date_range_dict[time_key] += tmp
    # else:
    #     date_range_dict[time_key] = tmp

new_date_range_dict = dict()
for key in date_range_dict:
    v = date_range_dict[key]
    new_date_range_dict[key] = collections.Counter(v)
# dist = collections.Counter(topic_node)
# plot freq of some topics
# 27 33 44 37 10 6 9
# for topic in [3,27, 33, 44, 37, 10, 6, 9]:
for topic in [6, 27,39,46]:
    protest_topic_freq = []
    for key in new_date_range_dict:
        v = new_date_range_dict[key]
        protest_topic_freq.append(v.get(topic,0))
    print(topic,protest_topic_freq)