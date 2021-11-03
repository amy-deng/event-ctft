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
from text_utils import *
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from scipy import sparse
from scipy.spatial.distance import cdist
import dgl
from dgl.data.utils import save_graphs,load_graphs

#### build datasets
### testing
 
'''
python build_word_topic_graph_data.py /home/sdeng/data/icews/detailed_event_json/THA_2010_w21h7_city.json ../data THA_50 /home/sdeng/data/icews/corpus/ngrams/THA_1gram_tfidf.txt 15000 7 7 3 ../data/THA_topic/check_topic_causal_data_w7h7/causal_effect/effect_dict_pw7_biy1_0.05.csv
'''
# event_path = '/home/sdeng/data/icews/detailed_event_json/THA_2010_w21h7_city.json'
# lda_name = 'THA_50'
# ngram_path = '/home/sdeng/data/icews/corpus/ngrams/THA_1gram_tfidf.txt'
# country = 'THA'
# top_k_ngram = 15000
# window=7
# horizon=7
# his_days_threshold=3
# causal_file = '../data/THA_topic/check_topic_causal_data_w7h7/causal_effect/effect_dict_pw7_biy1_0.05.csv'
try:
    event_path = sys.argv[1] # /home/sdeng/data/icews/detailed_event_json/THA_2010_w21h7_city.json
    out_path = sys.argv[2]
    lda_name = sys.argv[3]
    ngram_path = sys.argv[4]
    top_k_ngram = int(sys.argv[5])
    window = int(sys.argv[6])
    horizon = int(sys.argv[7])
    his_days_threshold = int(sys.argv[8])
    causal_file = sys.argv[9] # ../data/THA_topic/check_topic_causal_data_w7h7/causal_effect/effect_dict_pw7_biy1_0.05.csv
    start_date = sys.argv[10]
    stop_date = sys.argv[11]
except:
    print("usage: <event_path> <out_path> <lda_name `THA_50`> <ngram_path> <top_k_ngram `15000`> <window 7> <horizon 7> <his_days_threshold 3> <causal_file> <start_date 2010-01-01> <stop_date 2017-01-01>")
    exit()

country = event_path.split('/')[-1][:3]
dataset = '{}_w{}h{}_minday{}'.format(country,window,horizon,his_days_threshold)
dataset_path = "{}/{}".format(out_path,dataset)
os.makedirs(dataset_path, exist_ok=True)
print('dataset_path',dataset_path)
 

'''event and news'''
df = pd.read_json(event_path,lines=True)
news_df = pd.read_json('/home/sdeng/data/icews/news.1991.201703.country/icews_news_{}.json'.format(country), lines=True)
'''topic model'''
loaded_dict = corpora.Dictionary.load('/home/sdeng/data/icews/topic_models/{}.dict'.format(country))
loaded_lda =  models.LdaModel.load('/home/sdeng/data/icews/topic_models/{}.lda'.format(lda_name))
print('topic model and dictionary loaded')
'''vocabulary'''
# /home/sdeng/data/icews/corpus/ngrams/THA_1gram_tfidf.txt
with open(ngram_path,'r') as f:
    vocab = f.read().splitlines()
vocab = vocab[:top_k_ngram]
print('vocab loaded',len(vocab))

outf = dataset_path + '/data_{}_{}_tt85_sentpmi.pkl'.format(start_date,stop_date)

print(outf)

word_id_map = {}
for i in range(len(vocab)):
    word_id_map[vocab[i]] = i

# 2017-01-01
splitted_date_lists = [
    '2010-07-01',
    '2011-01-01','2011-07-01','2012-01-01','2012-07-01','2013-01-01','2013-07-01',
    '2014-01-01','2014-07-01','2015-01-01','2015-07-01','2016-01-01','2016-07-01',
    '2017-01-01','2017-07-01'
]

'''causal_file'''
causal_df = pd.read_csv(causal_file,sep=',')
causal_df = causal_df.loc[causal_df['event-type']=='protest']
causal_time_dict = {}
for end_date in splitted_date_lists:
    tmp = causal_df.loc[causal_df['end-date']==end_date]
    causal_topic_effect = tmp[['topic-id','effect']].values
    effect_all_topic = np.zeros(50)#[0. for i in range(50)]
    for topic_id, eff in causal_topic_effect:
        effect_all_topic[int(topic_id)] = round(eff,5)
    causal_time_dict[end_date] = effect_all_topic

'''non-causal since we defined the significance level, 
then middle parts are considered random, 
maybe just randomly denoise those topics in approach design'''


def word_word_pmi_sent(tokens_list, window_size=20):
    '''
    tokens_list = [['thailand', 'district', 'injury', 'reported', 'explosion', 'damaged'],['thailand','bomb', 'patrolman']]
    '''
    windows = tokens_list
    # windows = [] # get all moving windows
    # for tokens in tokens_list:
    #     length = len(tokens)
    #     if length <= window_size:
    #         windows.append(tokens)
    #     else:
    #         for j in range(length - window_size + 1):
    #             window = tokens[j: j + window_size]
    #             windows.append(window)
    # print(len(windows),windows[:3])
    word_window_freq = {} # get word freq in windows
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])
    # print(len(appeared))
    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_j = window[j]
                if word_i not in word_id_map or word_j not in word_id_map:
                    continue
                word_i_id = word_id_map[word_i]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
    # pmi as weight
    row, col, weight = [], [], []
    num_window = len(windows)
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = math.log((1.0 * count * num_window) / (1.0 * word_freq_i * word_freq_j))
        if pmi <= 0:
            continue
        row.append(i)
        col.append(j)
        weight.append(pmi)
    return row, col, weight
    

def doc_word_tfidf(tokens_list):
    word_doc_list = {} # document frequency DF
    for i in range(len(tokens_list)):
        words = tokens_list[i]
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)
    # return word_doc_list,word_doc_freq
    # doc word frequency TF
    doc_word_freq = {}
    for doc_id in range(len(tokens_list)):
        words = tokens_list[doc_id]
        for word in words:
            if word not in word_id_map:
                continue
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1
    # return doc_word_freq
    doc_node, word_node, weight = [], [], []
    for i in range(len(tokens_list)):
        words = tokens_list[i]
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            if word not in word_id_map:
                continue 
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            doc_node.append(i)
            word_node.append(j) 
            idf = math.log(1.0 * len(tokens_list) / word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)
    return doc_node, word_node, weight

def doc_topic_dist(tokens_list):
    corpus_bow = [loaded_dict.doc2bow(text) for text in tokens_list]
    topic_dists =  loaded_lda.get_document_topics(corpus_bow,per_word_topics=False,minimum_probability=0.01)
    # r =  loaded_lda.get_document_topics(corpus_bow,per_word_topics=False,minimum_probability=0.00)
    '''
    r[0] = [(3, 0.20297068), (5, 0.1559293), (13, 0.5837211), (46, 0.035471357)]
    r[1] = [(3, 0.54063946), (5, 0.23690455), (11, 0.1461465), (42, 0.043448117)]
    '''
    doc_node, topic_node, weight = [], [], []
    for doc_id in range(len(topic_dists)):
        topic_weights = topic_dists[doc_id]
        # print('topic_weights',topic_weights)
        for t,w in topic_weights:
            doc_node.append(doc_id)
            topic_node.append(t)
            weight.append(w)
    return doc_node, topic_node, weight
    

def topic_topic_sim(percent=95):
    term_topic_mat = loaded_lda.get_topics()
    num_topics = len(term_topic_mat)
    cosine_similarity = 1 - cdist(term_topic_mat, term_topic_mat, metric='cosine')
    np.fill_diagonal(cosine_similarity, 0)
    threshold = np.percentile(cosine_similarity,percent)
    cosine_similarity_thr = np.where(cosine_similarity>threshold, cosine_similarity, 0)
    topic_i, topic_j, weight = [], [], []
    for i in range(num_topics):
        for j in range(num_topics):
            if i == j or cosine_similarity_thr[i][j] <= 0:
                continue 
            topic_i.append(i)
            topic_j.append(j)
            weight.append(cosine_similarity_thr[i][j])
            # no threshold, will 
    return topic_i, topic_j, weight


def topic_word_conn(sample_words,num_words=20):
    term_topic_mat = loaded_lda.get_topics()
    num_topics = len(term_topic_mat)
    topic_node, word_node, weight = [], [], []
    for topic_id in range(num_topics):
        top_word_weights = loaded_lda.get_topic_terms(topic_id,num_words)
        for word,w in top_word_weights:
            word_str = loaded_dict[word]
            if word_str in sample_words:
                topic_node.append(topic_id)
                word_node.append(word_id_map[word_str])
                weight.append(w)
    return topic_node, word_node, weight

# his_days_threshold=3
num_sample, num_pos_sample = 0, 0
all_g_list, y_list, city_list, date_list = [], [], [], []

# topic---topic
topic_i, topic_j, weight = topic_topic_sim(percent=85)
edge_tt = torch.tensor(weight).float()
print('# topic nodes',len(set(topic_i)),len(set(topic_j)),'weight',len(weight))

for i,row in df.iterrows():
    city = row['city']
    date = str(row['date'])[:10]
    if date < start_date or date >= stop_date: #<2015-01-01 or >= 2017-01-01]
        continue
    day_has_data = 0
    story_list = row['story_list'][-window:]
    for v in story_list:
        if len(v) > 0:
            day_has_data += 1
    if day_has_data < his_days_threshold:
        continue
    # print(date,type(date),str(date))
    event_count_list = row['event_count_list'][:horizon] # event_count = row['event_count']
    event_count = {}
    ys = []
    for ii in range(len(event_count_list)):
        curr_event_count = event_count_list[ii]
        if len(curr_event_count) > 0: 
            for key in curr_event_count:
                event_count[key] = event_count.get(key,0)+curr_event_count[key]
        if event_count and '14' in event_count:
            ys.append(1)
        else:
            ys.append(0)
    

    # 1. get causal topic
    for end_date in splitted_date_lists: # check date in which range
        if date < end_date:
            cur_end_date = end_date
            break
    causal_weight = causal_time_dict[cur_end_date]
    # used to set topic nodes [just hightlight topic] TODO
    
    # continue
    # 2. build hetero graph for each day
    g_list = []
    for story_ids_day in story_list:
        if len(story_ids_day) <= 0:
            # how to deal with it
            continue
        story_text_lists = news_df.loc[news_df['StoryID'].isin(story_ids_day)]['Text'].values
        if len(story_text_lists) <= 0:
            # print('story_ids_day',len(story_ids_day),'story_text_lists',len(story_text_lists))
            continue
        # tokens_list = clean_document_list(story_text_lists)
        tokens_list, sent_token_list = document_sent_tokenize(story_text_lists)
        # words appeared in this example
        sample_words = list(set([item for sublist in tokens_list for item in sublist]))
        sample_words = [w for w in sample_words if w in vocab]

        graph_data = {}
        # doc---word
        doc_node, word_node, weight = doc_word_tfidf(tokens_list)
        # print('# doc nodes',len(set(doc_node)),len(set(word_node)))
        words_in_curr_sample = list(set(word_node))
        # print('words_in_curr_sample',len(words_in_curr_sample))
        words_in_curr_sample.sort()
        vocab_graph_node_map = dict(zip(words_in_curr_sample,range(len(words_in_curr_sample))))
        word_graph_node = [vocab_graph_node_map[v] for v in word_node]
        graph_data[('word','wd','doc')]=(torch.tensor(word_graph_node),torch.tensor(doc_node))
        edge_dw = torch.tensor(weight)

        # word---word
        word_i, word_j, weight = word_word_pmi_sent(sent_token_list, window_size=10) # window-size=20
        word_graph_node_i = [vocab_graph_node_map[v] for v in word_i]
        word_graph_node_j = [vocab_graph_node_map[v] for v in word_j]
        graph_data[('word','ww','word')]=(torch.tensor(word_graph_node_i),torch.tensor(word_graph_node_j))
        # print('# word nodes',len(set(word_i)),len(set(word_j)))
        # vocab_ids, edges = np.unique((word_i, word_j), return_inverse=True)  
        # src, dst = np.reshape(edges, (2, -1))
        # graph_data[('word','ww','word')]=(torch.tensor(src),torch.tensor(dst))
        edge_ww = torch.tensor(weight)
        # vocab_graph_node_map = dict(zip(vocab_ids,range(len(vocab_ids))))

        # doc---topic
        doc_node, topic_node, weight = doc_topic_dist(tokens_list)
        # print('# topic nodes',len(set(topic_node)),len(set(doc_node)))
        graph_data[('topic','td','doc')]=(torch.tensor(topic_node),torch.tensor(doc_node))
        edge_dt = torch.tensor(weight)
        '''# topic---topic
        topic_i, topic_j, weight = topic_topic_sim(percent=85)
        # print('# topic nodes',len(set(topic_i)),len(set(topic_j)))
        graph_data[('topic','tt','topic')]=(torch.tensor(topic_i),torch.tensor(topic_j))
        edge_tt = torch.tensor(weight)'''
        graph_data[('topic','tt','topic')]=(torch.tensor(topic_i),torch.tensor(topic_j))
        # topic---word
        topic_node, word_node, weight = topic_word_conn(sample_words,num_words=20) #need check words existed in topics
        # print('# word nodes',len(set(word_node)),len(set(topic_node)))
        word_graph_node = [vocab_graph_node_map[v] for v in word_node]
        graph_data[('word','wt','topic')]=(torch.tensor(word_graph_node),torch.tensor(topic_node))
        edge_tw = torch.tensor(weight)

        g = dgl.heterograph(graph_data)
        # g.nodes['word'].data['id'] = torch.from_numpy(vocab_ids).long()
        g.nodes['word'].data['id'] = torch.tensor(words_in_curr_sample).long()
        g.nodes['topic'].data['id'] = g.nodes('topic').long()

        topic_graph_nodes = g.nodes('topic').numpy()
        curr_causal_weight = torch.from_numpy(causal_weight[topic_graph_nodes])
        g.nodes['topic'].data['effect'] = curr_causal_weight
        g.edges['ww'].data['weight'] = edge_ww
        g.edges['wd'].data['weight'] = edge_dw
        g.edges['td'].data['weight'] = edge_dt
        g.edges['tt'].data['weight'] = edge_tt
        g.edges['wt'].data['weight'] = edge_tw
        g.ids = {}
        idx = 0
        for id in words_in_curr_sample:
            g.ids[id] = idx
            idx += 1
        # print(g)
        g_list.append(g)
    if len(g_list) < his_days_threshold:
        continue
    all_g_list.append(g_list)
    y_list.append(ys)  
    city_list.append(city)
    date_list.append(date)
    print('i={} \t {} \t {} \t {} day_has_data \t cur_end_date:{} {}'.format(i,date,city,len(g_list),cur_end_date,time.ctime()))
    # if len(all_g_list) >= 3:
        # break

y_list = torch.tensor(y_list)
# save_graphs(dataset_path + "/data.bin", all_g_list, {"y":y_list})
print('g',len(all_g_list),'y',len(y_list), 'date',len(date_list), 'city',len(city_list))
attr_dict = {"graphs_list":all_g_list,"y":y_list,"date":date_list,"city":city_list}

with open(outf,'wb') as f:
    pickle.dump(attr_dict, f)
print(outf, 'saved!')

 
# data = np.array([
#     [3,0,2],
#     [2,0,3],
#     [1,1,3]
# ])
# src, rel, dst = data.transpose()
# uniq_v, edges = np.unique((src, dst), return_inverse=True)  
# src, dst = np.reshape(edges, (2, -1))
# uniq_v
 