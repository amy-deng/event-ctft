import pandas as pd
import numpy as np
import sys, os, json, time, collections
# import pickle
# import glob
from gensim import corpora, models, similarities
from gensim.models.ldamulticore import LdaMulticore,LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_corpus, common_dictionary


try:
    event_path = sys.argv[1] # /home/sdeng/data/icews/detailed_event_json/THA_2010_w14h7_city.json
    out_path = sys.argv[2]
    # dataset = sys.argv[4] # THA
    # start_year = int(sys.argv[3])
    # end_year = int(sys.argv[4])
    window = int(sys.argv[4])
    horizon = int(sys.argv[5])
    lda_name = sys.argv[6]
except:
    print("usage: <event_path> <out_path>  <window <=14 > <horizon <=7 > <lda_name >")
    exit()

country = event_path.split('/')[-1][:3]
dataset = country + '_topic'
dataset_path = "{}/{}".format(out_path,dataset)
os.makedirs(dataset_path, exist_ok=True)

out_file = "w{}h{}.pkl".format(window,horizon)


df = pd.read_json(event_path,lines=True)

news_df = pd.read_json('/home/sdeng/data/icews/news.1991.201703.country/icews_news_{}.json'.format(country), lines=True)

loaded_dict = corpora.Dictionary.load('/home/sdeng/data/icews/topic_models/{}.dict'.format(country))
loaded_lda =  models.LdaModel.load('/home/sdeng/data/icews/topic_models/{}.lda'.format(lda_name))


# TODO
# def clean_document(text):
#     text = re.sub(r"''", " ",text) 
#     text = re.sub(r"\\n", " ",text) 
#     return sentence_tokenize(text)

# def clean_document_list(texts):
#     l = []
#     for t in texts:
#         l.append(clean_document(t))
#     return l 


for i,row in df:
    story_list = row['story_list'][14-window:]
    story_list = [item for sublist in story_list for item in sublist]
    story_list = list(set(story_list))
    news_df.loc[news_df['StoryID'].isin(story_list)]
