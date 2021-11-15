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
from numpy import linalg

#### build datasets
### testing
 
'''
python build_hetero_graph_notime.py /home/sdeng/data/icews/detailed_event_json/THA_2010_w21h7_city.json ../data THA_50 /home/sdeng/data/icews/corpus/ngrams/THA_1gram_tfidf.txt 15000 7 7 6 2014 2015
'''
try:
    event_path = sys.argv[1] # /home/sdeng/data/icews/detailed_event_json/THA_2010_w21h7_city.json
    out_path = sys.argv[2]
    lda_name = sys.argv[3]
    ngram_path = sys.argv[4]
    top_k_ngram = int(sys.argv[5])
    window = int(sys.argv[6])
    horizon = int(sys.argv[7])
    news_threshold = int(sys.argv[8])
    # causal_file = sys.argv[9] # ../data/THA_topic/check_topic_causal_data_w7h7/causal_effect/effect_dict_pw7_biy1_0.05.csv
    start_year = sys.argv[9]
    stop_year = sys.argv[10]
    vocab_size = int(sys.argv[11])
    
except:
    print("usage: <event_path> <out_path> <lda_name `THA_50`> <ngram_path> <top_k_ngram `15000`> <window 7> <horizon 7> <news_threshold 3> <start_year 2010> <stop_year 2017> <vocab_size>")
    exit()

country = event_path.split('/')[-1][:3]
dataset = '{}_w{}h{}_minday{}'.format(country,window,horizon,news_threshold)
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

if vocab_size > 0:
    outf = dataset_path + '/static_tf_{}-{}_{}.pkl'.format(start_year,stop_year,vocab_size)
else:
    outf = dataset_path + '/static_tf_{}-{}.pkl'.format(start_year,stop_year)
print(outf)

start_date = '{}-01-01'.format(start_year)
if stop_year == '2017':
    stop_date = '{}-03-20'.format(stop_year)
else:
    stop_date = '{}-01-01'.format(stop_year)

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
 
# https://github.com/dmlc/dgl/blob/ddc2faa547da03e0b791648677ed06ce1daf3e0d/examples/pytorch/gcn/gcn_spmv.py
def norm_edges(g,ntype,etype):
    # in_deg = g.in_degrees(etype='ww',range(g.number_of_nodes('word'))).float()
    degs = g.in_degrees(etype=etype).float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.nodes[ntype].data['norm'] = norm

def get_topwords(docs, top_n=800, use_tfidf=True):
    if use_tfidf:
        vectorizer = TfidfVectorizer(
                    analyzer='word',
                    tokenizer=lambda x: x,
                    preprocessor=lambda x: x,
                    token_pattern=None,
                    min_df = 1) # ignore terms that appear in less than 5 documents, default is 1
        X = vectorizer.fit_transform(docs)
        indices = np.argsort(vectorizer.idf_)[::-1]
        features = vectorizer.get_feature_names()
        top_features = [features[i] for i in indices[:top_n]]
    else:
        vectorizer = CountVectorizer(
                    analyzer='word',
                    tokenizer=lambda x: x,
                    preprocessor=lambda x: x,
                    token_pattern=None,
                    min_df = 1) # ignore terms that appear in less than 5 documents, default is 1
        X = vectorizer.fit_transform(docs)
        freqs = zip(vectorizer.get_feature_names(), X.sum(axis=0).tolist()[0])    
        # sort from largest to smallest
        top_features = sorted(freqs, key=lambda x: -x[1])[:top_n]
        top_features = [v[0] for v in top_features]
    return top_features


def word_word_pmi_norm(tokens_list, sample_words, window_size=20): # , window_size=20
    '''
    tokens_list = [['thailand', 'district', 'injury', 'reported', 'explosion', 'damaged'],['thailand','bomb', 'patrolman']]
    '''

    # unique_tokens_list = []
    # for l in tokens_list:
    #     unique_tokens_list.append(list(set(l)))# unique

    windows = [] # get all moving windows
    for tokens in tokens_list:
        length = len(tokens)
        if length <= window_size:
            windows.append(tokens)
        else:
            for j in range(length - window_size + 1):
                window = tokens[j: j + window_size]
                windows.append(window)
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
        word_pair_str_appeared = set()
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_j = window[j]
                # if word_i not in word_id_map or word_j not in word_id_map:
                #     continue
                # if word_i not in sample_words or word_j not in sample_words:
                #     continue
                # word_i_id = word_id_map[word_i]
                # word_j_id = word_id_map[word_j]
                word_i_id = word_i
                word_j_id = word_j
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_str_appeared:
                    # print('skip')
                    continue
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                word_pair_str_appeared.add(word_pair_str)
                word_pair_str = str(word_j_id) + ',' + str(word_i_id) # two orders
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                word_pair_str_appeared.add(word_pair_str)
    row, col, weight = [], [], [] # pmi as weight
    num_window = len(windows)
    for key in word_pair_count:
        temp = key.split(',')
        i = temp[0]
        j = temp[1]
        count = word_pair_count[key]
        word_freq_i = word_window_freq[i]
        word_freq_j = word_window_freq[j]
        # https://towardsdatascience.com/word2vec-for-phrases-learning-embeddings-for-more-than-one-word-727b6cf723cf
        pmi = math.log((1.0 * count * num_window) / (1.0 * word_freq_i * word_freq_j)) 
        if pmi <= 0:
            continue
        try:
            npmi = pmi / (-math.log(count/num_window))
            # print('count=',count,'num_window=',num_window,'word_freq_i=',word_freq_i,'word_freq_j=',word_freq_j,'pmi=',pmi,'npmi=',npmi)
        except:
            print('count=',count,'num_window=',num_window,'word_freq_i=',word_freq_i,'word_freq_j=',word_freq_j,'pmi=',pmi)
            # print('npmi=',npmi)
            exit()
        row.append(i)
        col.append(j)
        weight.append(npmi)
    self_loop = set() # add self loop
    for node in row:
        if node in self_loop:
            continue
        row.append(node)
        col.append(node)
        weight.append(1.)
        self_loop.add(node)
    return row, col, weight
 
def doc_word_tfidf(tokens_list, sample_words):
    
    word_doc_list = {} # document frequency DF
    for i in range(len(tokens_list)):
        words = tokens_list[i]
        appeared = set()
        for word in words:
            # if word not in sample_words: 
                # continue
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
            # if word not in sample_words: 
                # continue
            # if word not in word_id_map:
            #     continue
            # word_id = word_id_map[word]
            word_id = word
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1
    # return doc_word_freq
    doc_node, word_node, weight = [], [], []
    for i in range(len(tokens_list)):
        words = tokens_list[i]
        tmp_doc_node, tmp_word_node, tmp_weight = [], [], []
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            # if word not in word_id_map:
            #     continue 
            # if word not in sample_words: 
            #     continue
            # j = word_id_map[word]
            j = word
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            idf = math.log(1.0 * len(tokens_list) / word_doc_freq[j])
            tfidf = freq * idf
            if tfidf <= 0:
                continue
            tmp_doc_node.append(i)
            tmp_word_node.append(j)
            tmp_weight.append(tfidf)
            doc_word_set.add(word)
        # normalization
        # tmp_weight = np.array(tmp_weight)
        sum_tfidf = linalg.norm(tmp_weight, ord=2)
        norm_weight = [round(v/sum_tfidf, 4) for v in tmp_weight]
        doc_node += tmp_doc_node
        word_node += tmp_word_node
        weight += norm_weight
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
 

def topic_topic_sim(thr=0.15):
    term_topic_mat = loaded_lda.get_topics()
    num_topics = len(term_topic_mat)
    cosine_similarity = 1 - cdist(term_topic_mat, term_topic_mat, metric='cosine')
    # np.fill_diagonal(cosine_similarity, 0)
    # threshold = np.percentile(cosine_similarity,percent)
    # cosine_similarity_thr = np.where(cosine_similarity>=0.1, cosine_similarity, 0)
    topic_i, topic_j, weight = [], [], []
    loop = set()
    for i in range(num_topics):
        for j in range(i, num_topics):
            # if i == j or cosine_similarity[i][j] < thr:
            if cosine_similarity[i][j] < thr:
                continue 
            topic_i.append(i)
            topic_j.append(j)
            weight.append(cosine_similarity[i][j])
            if i == j:
                continue
            topic_i.append(j)
            topic_j.append(i)
            weight.append(cosine_similarity[j][i])
    return topic_i, topic_j, weight


def topic_word_conn(sample_words,num_words=30):
    term_topic_mat = loaded_lda.get_topics()
    num_topics = len(term_topic_mat)
    topic_node, word_node, weight = [], [], []
    for topic_id in range(num_topics):
        top_word_weights = loaded_lda.get_topic_terms(topic_id,num_words)
        for word,w in top_word_weights:
            word_str = loaded_dict[word]
            if word_str in sample_words:
                topic_node.append(topic_id)
                # word_node.append(word_id_map[word_str])
                word_node.append(word_str)
                weight.append(w)
    return topic_node, word_node, weight


# news_threshold=3
num_sample, num_pos_sample = 0, 0
all_g_list, y_list, city_list, date_list = [], [], [], []

iii=0
# topic---topic
topic_i, topic_j, weight = topic_topic_sim(thr=0.15) # 85
edge_tt = torch.tensor(weight).float()
print('# topic nodes',len(set(topic_i)),len(set(topic_j)),'weight',len(weight))
for i,row in df.iterrows():
    city = row['city']
    date = str(row['date'])[:10]
    if date < start_date or date >= stop_date: #<2015-01-01 or >= 2017-01-01]
        continue

    # total num of news
    story_list = row['story_list'][-window:]
    story_list_flatten = list(set([item for sublist in story_list for item in sublist]))
    if len(story_list_flatten) < news_threshold:
        print(len(story_list_flatten),'articles; first skip')
        continue
    story_text_lists = news_df.loc[news_df['StoryID'].isin(story_list_flatten)]['Text'].values
    if len(story_text_lists) < news_threshold:
        print(len(story_text_lists),'articles; skip')
        continue

    # day_has_data = 0
    # story_list = row['story_list'][-window:]
    # n_doc = 0
    # for v in story_list:
    #     if len(v) > 0:
    #         day_has_data += 1
    #     n_doc += len(v)
    # if day_has_data < his_days_threshold or n_doc <= 5:  
    #     continue
    
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
    # for end_date in splitted_date_lists: # check date in which range
    #     if date < end_date:
    #         cur_end_date = end_date
    #         break
    # causal_weight = causal_time_dict[cur_end_date]
    
    # continue
    # 2. build hetero graph for each day
    g_list = []
    iii+=1
    # if iii < 813:
    #     print(iii,len(story_text_lists))
    #     continue
    tokens_list = clean_document_list(story_text_lists)
    # if len(tokens_list) <= 5:
    #     continue
    # tokens_list, sent_token_list = document_sent_tokenize(story_text_lists)
    # words appeared in this example
    sample_words = list(set([item for sublist in tokens_list for item in sublist]))
    if vocab_size > 0:
        if len(sample_words) > vocab_size:
            # sample_words = get_topwords(tokens_list,vocab_size)
            sample_words = get_topwords(tokens_list, vocab_size, False)

    sample_words = [w for w in sample_words if w in vocab]

    tokens_list_clean = []
    for l in tokens_list:
        tokens_list_clean.append([v for v in l if v in sample_words])
 
    words_in_curr_sample = [word_id_map[w] for w in sample_words] # [5,6,7,10,8,...]
    vocab_graph_node_map = dict(zip(sample_words,range(len(words_in_curr_sample))))


    graph_data = {}
    # doc---word
    doc_node, word_node, weight = doc_word_tfidf(tokens_list_clean,sample_words)
    word_graph_node = [vocab_graph_node_map[v] for v in word_node]
    graph_data[('word','wd','doc')]=(torch.tensor(word_graph_node),torch.tensor(doc_node))

    edge_dw = torch.tensor(weight)

    # word---word
    # word_i, word_j, weight = word_word_pmi_sent_norm(sent_token_list, sample_words) # window-size=20
    word_i, word_j, weight = word_word_pmi_norm(tokens_list_clean, sample_words, window_size=20)
    word_graph_node_i = [vocab_graph_node_map[v] for v in word_i]
    word_graph_node_j = [vocab_graph_node_map[v] for v in word_j]
    graph_data[('word','ww','word')]=(torch.tensor(word_graph_node_i),torch.tensor(word_graph_node_j))
    edge_ww = torch.tensor(weight)
    

    # doc---topic
    doc_node, topic_node, weight = doc_topic_dist(tokens_list)
    graph_data[('topic','td','doc')]=(torch.tensor(topic_node),torch.tensor(doc_node))
    edge_dt = torch.tensor(weight)
     
    graph_data[('topic','tt','topic')]=(torch.tensor(topic_i),torch.tensor(topic_j))
    # topic---word
    topic_node, word_node, weight = topic_word_conn(sample_words,num_words=30) #need check words existed in topics
    word_graph_node = [vocab_graph_node_map[v] for v in word_node]
    graph_data[('word','wt','topic')]=(torch.tensor(word_graph_node),torch.tensor(topic_node))
    edge_tw = torch.tensor(weight)
     
    g = dgl.heterograph(graph_data)
    # g.nodes['word'].data['id'] = torch.from_numpy(vocab_ids).long()
    g.nodes['word'].data['id'] = torch.tensor(words_in_curr_sample).long()
    g.nodes['topic'].data['id'] = g.nodes('topic').long()
    # topic_graph_nodes = g.nodes('topic').numpy()
    # # curr_causal_weight = torch.from_numpy(causal_weight[topic_graph_nodes])
    # # g.nodes['topic'].data['effect'] = curr_causal_weight
    # g.nodes['topic'].data['effect'] = torch.from_numpy(causal_weight[topic_graph_nodes]).to_sparse()
    g.edges['ww'].data['weight'] = edge_ww
    g.edges['wd'].data['weight'] = edge_dw
    g.edges['td'].data['weight'] = edge_dt
    g.edges['tt'].data['weight'] = edge_tt
    g.edges['wt'].data['weight'] = edge_tw
    norm_edges(g,ntype='word',etype='ww')
    norm_edges(g,ntype='topic',etype='tt')
    # g.ids = {}
    # idx = 0
    # for id in words_in_curr_sample:
    #     g.ids[id] = idx
    #     idx += 1
    # print(g)
    # g_list.append(g) 
    all_g_list.append(g)
    y_list.append(ys)  
    city_list.append(city)
    date_list.append(date)
    
    print('iii={} \t {} \t {} \t {} day_has_data \t  {} vocab {} doc {}'.format(iii,date,city,1,time.ctime(),len(sample_words),len(tokens_list)))
 

y_list = torch.tensor(y_list)
# save_graphs(dataset_path + "/data.bin", all_g_list, {"y":y_list})
print('g',len(all_g_list),'y',len(y_list), 'date',len(date_list), 'city',len(city_list))
attr_dict = {"graphs_list":all_g_list,"y":y_list,"date":date_list,"city":city_list}
with open(outf,'wb') as f:
    pickle.dump(attr_dict, f)
print(outf, 'saved!')

 