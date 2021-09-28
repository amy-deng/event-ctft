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
'''
python build_causal_data.py
'''
try:
    event_path = sys.argv[1] # /home/sdeng/data/icews/detailed_event_json/THA_2010_w14h7_city.json
    out_path = sys.argv[2]
    # dataset = sys.argv[4] # THA
    # start_year = int(sys.argv[3])
    # end_year = int(sys.argv[4])
    window = int(sys.argv[3])
    horizon = int(sys.argv[4])
    lda_name = sys.argv[5]
except:
    print("usage: <event_path> <out_path>  <window <=14 > <horizon <=7 > <lda_name >")
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


# list(set(story_list))
# np
raw_covariates = []
raw_outcomes = []
for i,row in df.iterrows():
    story_list = row['story_list'][14-window:]
    story_list = [item for sublist in story_list for item in sublist]
    story_list = list(set(story_list))
    if len(story_list) <= 0:
        continue # no story id

    text_df = news_df.loc[news_df['StoryID'].isin(story_list)]
    if text_df.empty:
        continue # no text
    text_list = text_df['Text'].values
#     print(text_list)
    processed_tokens = clean_document_list(text_list)
    # processed_tokens = [['crime', 'business','transnational','crime','suppression','csd','stepping','effort','seek','cooperation','foreign','embassy','embassy','criminal','coming','country'],
    #                     ['hitman', 'business', 'transnational', 'crime', 'suppression','csd', 'stepping', 'effort', 'seek','cooperation', 'foreign', 'embassy', 'embassy','criminal','coming', 'country']]
    corpus_bow = [loaded_dict.doc2bow(text) for text in processed_tokens]
    r =  loaded_lda.get_document_topics(corpus_bow,per_word_topics=False,minimum_probability=0.01)
    topic_ids = []
    topic_vec = np.zeros(50)
    for j in range(len(r)):
        topic_id = [a_tuple[0] for a_tuple in r[j]]
        topic_ids += topic_id
    topic_count = collections.Counter(topic_ids)
    for k in topic_count:
        topic_vec[k] = topic_count[k]
    
    # output 
    event_vec = np.zeros(20)
    event_count = row['event_count']
    for k in event_count:
        event_vec[int(k)-1] = event_count[k]
    
    # print(event_vec)
    # print(topic_vec)
    raw_covariates.append(topic_vec)
    raw_outcomes.append(event_vec)
 
    
raw_covariates = np.stack(raw_covariates,0)
raw_outcomes = np.stack(raw_outcomes,0)
print('raw_outcomes',raw_outcomes.shape, 'raw_outcomes',raw_outcomes.shape)
with open("{}/{}".format(dataset_path,out_file),'wb') as f:
    pickle.dump({'covariate':raw_covariates,'outcome':raw_outcomes},f)

print(out_file,'saved')