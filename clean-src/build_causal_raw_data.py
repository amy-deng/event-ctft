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
'''
python build_causal_raw_data_w_time2.py /home/sdeng/data/icews/detailed_event_json/THA_2010_w21h14_city.json ../data 14 14 THA_50 /home/sdeng/data/icews/corpus/ngrams/THA_1gram_tfidf.txt 15000 2013 0.1
python build_causal_raw_data_w_time2.py /home/sdeng/data/icews/detailed_event_json/EGY_2010_w21h14_city.json ../data 14 14 EGY_50 /home/sdeng/data/icews/corpus/ngrams/EGY_1gram_tfidf.txt 15000 2013 0.1
python build_causal_raw_data_w_time2.py /home/sdeng/data/icews/detailed_event_json/RUS_2010_w21h14_city.json ../data 14 14 RUS_50 /home/sdeng/data/icews/corpus/ngrams/RUS_1gram_tfidf.txt 5000 2013 0.1
python build_causal_raw_data_w_time2.py /home/sdeng/data/icews/detailed_event_json/AFG_2010_w21h14_city.json ../data 14 14 AFG_50 /home/sdeng/data/icews/corpus/ngrams/AFG_1gram_tfidf.txt 25000 2013 0.1
python build_causal_raw_data_w_time2.py /home/sdeng/data/icews/detailed_event_json/IND_2010_w21h14_city.json ../data 14 14 IND_50 /home/sdeng/data/icews/corpus/ngrams/IND_1gram_tfidf.txt 25000 2012 0.2

python build_causal_raw_data.py /home/sdeng/data/icews/detailed_event_json/THA_2010_w21h14_city.json ../data 14 14 THA_2012_50 /home/sdeng/data/icews/corpus/ngrams/THA_from2012_1gram_tfidf.txt -1 2013 0.1 THA_2012_50
python build_causal_raw_data.py /home/sdeng/data/icews/detailed_event_json/EGY_2010_w21h14_city.json ../data 14 14 EGY_2012_60 /home/sdeng/data/icews/corpus/ngrams/EGY_from2012_1gram_tfidf.txt -1 2013 0.1 EGY_2012_60
python build_causal_raw_data.py /home/sdeng/data/icews/detailed_event_json/AFG_2010_w21h14_city.json ../data 14 14 AFG_2012_60 /home/sdeng/data/icews/corpus/ngrams/AFG_from2012_1gram_tfidf.txt -1 2013 0.1 AFG_2012_60
python build_causal_raw_data.py /home/sdeng/data/icews/detailed_event_json/RUS_2010_w21h14_city.json ../data 14 14 RUS_2012_60 /home/sdeng/data/icews/corpus/ngrams/RUS_from2012_1gram_tfidf.txt -1 2013 0.15 RUS_2012_60

'''
try:
    event_path = sys.argv[1] # /home/sdeng/data/icews/detailed_event_json/THA_2010_w14h7_city.json
    out_path = sys.argv[2]
    window = int(sys.argv[3])
    horizon = int(sys.argv[4])
    lda_name = sys.argv[5]
    ngram_path = sys.argv[6]
    top_k_ngram = int(sys.argv[7])
    start_year = sys.argv[8]
    min_prob = float(sys.argv[9])
    dataset = sys.argv[10]
except:
    print("usage: <event_path> <out_path> <window <=13 > <horizon <=7 > <lda_name `THA_50`> <ngram_path> <top_k_ngram `15000`> <start_year> <min_prob> <dataset>")
    exit()

country = event_path.split('/')[-1][:3]
dataset = dataset + '_topic'
dataset_path = "{}/{}".format(out_path,dataset)
os.makedirs(dataset_path, exist_ok=True)
print('dataset_path',dataset_path)

out_file = "raw_topic_causal_data_w{}h{}_from{}_minprob{}_ngram{}.pkl".format(window,horizon,start_year,min_prob,top_k_ngram)
print('out_file',out_file)

df = pd.read_json(event_path,lines=True)
print('# of event (sample) recored',len(df))
df = df.loc[df['date'] >= str(start_year)+'-01-01']
print('# of event (sample) recored after {}-01-01'.format(start_year),len(df))

news_df = pd.read_json('/home/sdeng/data/icews/news.1991.201703.country/icews_news_{}.json'.format(country), lines=True)
dict_name = '_'.join(lda_name.split('_')[:2])
loaded_dict = corpora.Dictionary.load('/home/sdeng/data/icews/topic_models/{}.dict'.format(dict_name))
loaded_lda =  models.LdaModel.load('/home/sdeng/data/icews/topic_models/{}.lda'.format(lda_name))
num_topics = int(lda_name.split('_')[-1])
print('topic model and dictionary loaded', num_topics,'topics')

with open(ngram_path,'r') as f:
    ngram = f.read().splitlines()
if top_k_ngram > 0:
    ngram = ngram[:top_k_ngram]
print('ngram loaded',len(ngram))

c_vec = CountVectorizer(ngram_range=(1, 1),stop_words='english',vocabulary=ngram,binary=False)

raw_covariates = []
raw_treatments = []
raw_treatments_check = []
raw_outcomes = []
date_list = []

for i,row in df.iterrows():
    story_list = row['story_list']
    past_story_list = story_list[-window-1:-1]
    current_story_list = story_list[-1]

    past_story_list = [item for sublist in past_story_list for item in sublist]
    past_story_list = list(set(past_story_list))

    # current_story_list = [item for sublist in current_story_list for item in sublist]
    current_story_list = list(set(current_story_list))
    if len(current_story_list) <= 0 or len(past_story_list) <= 0 :
        continue # no story id

    past_text_df = news_df.loc[news_df['StoryID'].isin(past_story_list)]
    current_text_df = news_df.loc[news_df['StoryID'].isin(current_story_list)]

    if current_text_df.empty or past_text_df.empty:
        continue # no text

    '''topic as treatments at current time'''
    current_text_list = current_text_df['Text'].values
    processed_tokens = clean_document_list(current_text_list)
    corpus_bow = [loaded_dict.doc2bow(text) for text in processed_tokens]
    r =  loaded_lda.get_document_topics(corpus_bow,per_word_topics=False,minimum_probability=min_prob)
    topic_ids = []
    topic_vec = np.zeros(num_topics)
    for j in range(len(r)):
        topic_id = [a_tuple[0] for a_tuple in r[j]]
        topic_ids += topic_id
    topic_count = collections.Counter(topic_ids)
    for k in topic_count:
        topic_vec[k] = topic_count[k]
    raw_treatments.append(topic_vec)
    
    
    '''output (agg all future days)''' 
    '''
    event_vec = np.zeros(20)
    event_count = row['event_count']
    for k in event_count:
        event_vec[int(k)-1] = event_count[k]
    raw_outcomes.append(event_vec)
    '''

    event_vec = np.zeros((horizon,20))
    event_count_list = row['event_count_list']
    # print(len(event_count_list),event_vec.shape,'event_vec')
    # for i_ in range(len(event_count_list)):
    for i_ in range(horizon):
        event_count = event_count_list[i_]
        for k in event_count:
            event_vec[i_][int(k)-1] = event_count[k]
    raw_outcomes.append(event_vec)


    '''covariates'''
    past_text_list = past_text_df['Text'].values
    processed_str = ' '.join(clean_document_list_str(past_text_list))
    ngrams_vec = c_vec.fit_transform([processed_str])
    # print(ngrams_vec.shape) # scipy.sparse.csr.csr_matrix
    raw_covariates.append(ngrams_vec)
    # raw_covariates.append(ngrams_vec.toarray())

    # topic in past, used to check if the treatment topic is the first time appear
    processed_tokens = clean_document_list(past_text_list)
    corpus_bow = [loaded_dict.doc2bow(text) for text in processed_tokens]
    r =  loaded_lda.get_document_topics(corpus_bow,per_word_topics=False,minimum_probability=min_prob)
    topic_ids = []
    topic_vec = np.zeros(num_topics)
    for j in range(len(r)):
        topic_id = [a_tuple[0] for a_tuple in r[j]]
        topic_ids += topic_id
    topic_count = collections.Counter(topic_ids)
    for k in topic_count:
        topic_vec[k] = topic_count[k]
    raw_treatments_check.append(topic_vec)

    ''' date '''
    date_list.append(str(row['date'])[:10])

    if i % 100 == 0:
        print('processing i =',i,time.ctime())
    # if i > 30:
    #     print('testing...break')
    #     break
 
raw_treatments_check = np.stack(raw_treatments_check,0)
raw_treatments = np.stack(raw_treatments,0)
raw_outcomes = np.stack(raw_outcomes,0)
raw_covariates = np.stack(raw_covariates,0)
date_list = np.array(date_list)
print('raw_outcomes',raw_outcomes.shape,'raw_treatments',raw_treatments.shape,'raw_treatments_check',raw_treatments_check.shape)
print('raw_covariates',type(raw_covariates),raw_covariates.shape,'date_list',date_list.shape)

with open("{}/{}".format(dataset_path,out_file),'wb') as f:
    pickle.dump(
    {'covariate':raw_covariates,
    'outcome':raw_outcomes,
    'treatment':raw_treatments,
    'treatment_check':raw_treatments_check,
    'date':date_list},f)

print(out_file,'saved')