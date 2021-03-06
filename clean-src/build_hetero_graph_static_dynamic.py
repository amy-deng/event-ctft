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
 
'''
python build_hetero_graph_static_dynamic.py /home/sdeng/data/icews/detailed_event_json/THA_2010_w21h14_city.json ../data THA_2012_50 /home/sdeng/data/icews/corpus/ngrams/THA_from2012_1gram_tfidf.txt -1 7 7 7 3 2017 2017 900 0.01
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
    days_threshold = int(sys.argv[9])
    # causal_file = sys.argv[9] # ../data/THA_topic/check_topic_causal_data_w7h7/causal_effect/effect_dict_pw7_biy1_0.05.csv
    start_year = sys.argv[10]
    stop_year = sys.argv[11]
    vocab_size = int(sys.argv[12])
    mindf = float(sys.argv[13])
except:
    print("usage: <event_path> <out_path> <lda_name `THA_50`> <ngram_path> <top_k_ngram `15000`/-1> <window 7> <horizon 7> <news_threshold 7> <days_threshold> <start_year 2010> <stop_year 2017> <vocab_size> <mindf>")
    exit()

country = event_path.split('/')[-1][:3]
dataset = '{}_w{}h{}_mind{}n{}df{}'.format(country,window,horizon,days_threshold,news_threshold,mindf)
dataset_path = "{}/{}".format(out_path,dataset)
os.makedirs(dataset_path, exist_ok=True)
print('dataset_path',dataset_path)
 

'''event and news'''
df = pd.read_json(event_path,lines=True)
news_df = pd.read_json('/home/sdeng/data/icews/news.1991.201703.country/icews_news_{}.json'.format(country), lines=True)
news_df = news_df.loc[(news_df['Date']>str(int(start_year)-1)+'-12-15') & (news_df['Date']<str(int(start_year)+1)+'-01-10')]
print(len(news_df),'news_df')
'''topic model'''
dict_name = '_'.join(lda_name.split('_')[:2])
loaded_dict = corpora.Dictionary.load('/home/sdeng/data/icews/topic_models/{}.dict'.format(dict_name))
loaded_lda =  models.LdaModel.load('/home/sdeng/data/icews/topic_models/{}.lda'.format(lda_name))
print('topic model and dictionary loaded')
'''vocabulary'''
# /home/sdeng/data/icews/corpus/ngrams/THA_1gram_tfidf.txt
with open(ngram_path,'r') as f:
    vocab = f.read().splitlines()
if top_k_ngram > 0:
    vocab = vocab[:top_k_ngram]
print('vocab loaded',len(vocab))

if vocab_size > 0:
    outf_dynamic = dataset_path + '/dyn_tf_{}-{}_{}.pkl'.format(start_year,stop_year,vocab_size)
    outf_static = dataset_path + '/sta_tf_{}-{}_{}.pkl'.format(start_year,stop_year,vocab_size)
    outf_attr =  dataset_path + '/attr_tf_{}-{}_{}.pkl'.format(start_year,stop_year,vocab_size)
else:
    outf_dynamic = dataset_path + '/dyn_tf_{}-{}.pkl'.format(start_year,stop_year)
    outf_static = dataset_path + '/sta_tf_{}-{}.pkl'.format(start_year,stop_year)
    outf_attr =  dataset_path + '/attr_tf_{}-{}.pkl'.format(start_year,stop_year)

print(outf_dynamic)
print(outf_static)

start_date = '{}-01-01'.format(start_year)
if stop_year == '2017':
    stop_date = '{}-03-20'.format(stop_year)
    # stop_date = '{}-01-15'.format(stop_year)
else:
    stop_date = '{}-01-01'.format(stop_year)

word_id_map = {}
for i in range(len(vocab)):
    word_id_map[vocab[i]] = i

 
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
                    min_df = mindf) # ignore terms that appear in less than 5 documents, default is 1
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
                    min_df = 1,
                    max_df=0.9) # ignore terms that appear in less than 5 documents, default is 1
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
    windows = [] # get all moving windows
    for tokens in tokens_list:
        length = len(tokens)
        if length <= window_size:
            filtered = [w for w in tokens if w in sample_words]
            windows.append(filtered)
        else:
            for j in range(length - window_size + 1):
                filtered = [w for w in tokens[j: j + window_size] if w in sample_words]
                windows.append(filtered)
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
    
def topic_topic_sim(thr=0.2):
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


all_static_g_list, all_dynamic_g_list, y_list, city_list, date_list = [], [], [], [], []

iii=0
# topic---topic
topic_i, topic_j, weight = topic_topic_sim(thr=0.2) # 85
edge_tt = torch.tensor(weight).float()
print('# topic nodes',len(set(topic_i)),len(set(topic_j)),'weight',len(weight))
for i,row in df.iterrows():
    city = row['city']
    date = str(row['date'])[:10]
    if date < start_date or date >= stop_date: #<2015-01-01 or >= 2017-01-01]
        continue
    story_list = row['story_list'][-window:]

    story_len_day = [] # [0,0,1,1,3,6]
    num_nonzero_days = 0
    story_text_lists_all = [] 
    for sl in story_list:
        story_len = 0
        if len(sl) > 0:
            sl_text = news_df.loc[news_df['StoryID'].isin(sl)]['Text'].values
            if len(sl_text) > 0:
                # num_days_with_news += 1
                # num_news += len(sl_text)
                story_len = len(sl_text)
                num_nonzero_days += 1
                story_text_lists_all += sl_text.tolist()
        story_len_day.append(story_len)
    num_news = sum(story_len_day)
    
    if num_nonzero_days < days_threshold or num_news <= news_threshold:
        print('# days = {} \t # news = {} \t skip'.format(num_nonzero_days, num_news))
        continue
  
    tokens_list = clean_document_list(story_text_lists_all)
    # tokens_list, sent_token_list = document_sent_tokenize(story_text_lists)
    if len(tokens_list) <= news_threshold:
        print(len(tokens_list),'articles; third skip')
        continue

    iii+=1 
    sample_words = list(set([item for sublist in tokens_list for item in sublist]))
    if vocab_size > 0:
        if len(sample_words) > vocab_size:
            # sample_words = get_topwords(tokens_list,vocab_size, False)
            # print('[TF]',sample_words[:80])
            sample_words = get_topwords(tokens_list,vocab_size, True)
            # print('[TFIDF]',tfidf_sample_words[:80])
            # common = list(set(sample_words) & set(tfidf_sample_words))
    # continue
    sample_words = [w for w in sample_words if w in vocab and w]
    # print(sample_words)
    words_in_curr_sample = [word_id_map[w] for w in sample_words] # [5,6,7,10,8,...]
    vocab_graph_node_map = dict(zip(sample_words,range(len(words_in_curr_sample))))
    vocab_graph_node_map_reverse = dict(zip(range(len(words_in_curr_sample)),words_in_curr_sample))

    '''static graph'''
    graph_data = {}
    tokens_list_clean = []
    for l in tokens_list:
        tokens_list_clean.append([v for v in l if v in sample_words])
    
    # doc---word
    doc_node, word_node, weight = doc_word_tfidf(tokens_list_clean,sample_words)
    word_graph_node = [vocab_graph_node_map[v] for v in word_node]
    graph_data[('word','wd','doc')]=(torch.tensor(word_graph_node),torch.tensor(doc_node))
    graph_data[('doc','dw','word')]=(torch.tensor(doc_node),torch.tensor(word_graph_node))
    edge_wd = torch.tensor(weight)

    # word---word
    word_i, word_j, weight = word_word_pmi_norm(tokens_list, sample_words, window_size=20)
    word_graph_node_i = [vocab_graph_node_map[v] for v in word_i]
    word_graph_node_j = [vocab_graph_node_map[v] for v in word_j]
    graph_data[('word','ww','word')]=(torch.tensor(word_graph_node_i),torch.tensor(word_graph_node_j))
    edge_ww = torch.tensor(weight)

    # doc---topic
    doc_node, topic_node, weight = doc_topic_dist(tokens_list)
    graph_data[('topic','td','doc')]=(torch.tensor(topic_node),torch.tensor(doc_node))
    graph_data[('doc','dt','topic')]=(torch.tensor(doc_node),torch.tensor(topic_node))
    edge_td = torch.tensor(weight)

    graph_data[('topic','tt','topic')]=(torch.tensor(topic_i),torch.tensor(topic_j))
    # topic---word
    topic_node, word_node, weight = topic_word_conn(sample_words, num_words=30) #need check words existed in topics
    word_graph_node = [vocab_graph_node_map[v] for v in word_node]
    graph_data[('word','wt','topic')]=(torch.tensor(word_graph_node),torch.tensor(topic_node))
    graph_data[('topic','tw','word')]=(torch.tensor(topic_node),torch.tensor(word_graph_node))
    edge_wt = torch.tensor(weight)

    g = dgl.heterograph(graph_data)
    # print(g.num_nodes('word'),'words static')
    nodes1 = g.nodes('word')
    # print(nodes1)

    g.nodes['word'].data['id'] = torch.tensor(words_in_curr_sample).long()
    g.nodes['topic'].data['id'] = g.nodes('topic').long()
    g.edges['ww'].data['weight'] = edge_ww
    g.edges['wd'].data['weight'] = edge_wd
    g.edges['td'].data['weight'] = edge_td
    g.edges['tt'].data['weight'] = edge_tt
    g.edges['wt'].data['weight'] = edge_wt
    
    # inverse edge
    g.edges['dw'].data['weight'] = edge_wd
    g.edges['dt'].data['weight'] = edge_td
    g.edges['tw'].data['weight'] = edge_wt

    norm_edges(g,ntype='word',etype='ww')
    # norm_edges(g,ntype='topic',etype='tt') 
    # g = g.int()
    all_static_g_list.append(g)

    '''dynamic graph'''

    graph_data = {}
    split_indices = np.cumsum(story_len_day)
    story_text_lists_day = np.split(tokens_list, split_indices)
    # doc_ids_list_day = np.split(doc_ids, split_indices)

    ww_src, ww_dst, ww_time, ww_weight = [], [], [], []
    wd_src, wd_dst, wd_time, wd_weight = [], [], [], []
    wt_src, wt_dst, wt_time, wt_weight = [], [], [], []
    td_src, td_dst, td_time, td_weight = [], [], [], []
    
    doc_id = 0
    for day_i in range(len(story_text_lists_day)-1):
        # print(' - day ',day_i, '-')
        tokens_list_day = story_text_lists_day[day_i]
        if len(tokens_list_day) <= 0:
            continue

        tokens_list_clean = []
        for l in tokens_list_day:
            tokens_list_clean.append([v for v in l if v in sample_words])
        
        sample_words_day = list(set([item for sublist in tokens_list_clean for item in sublist]))
        # sample_words_day = [w for w in sample_words_day if w in sample_words]  
        # doc_ids_day = doc_ids_list_day[day_i] # 0,1,2
        # docidx_id_map = dict(zip(range(len(tokens_list_day)),doc_ids_day))
        if len(sample_words_day) <= 3:
            continue
        # print('docidx_id_map',docidx_id_map)
        if len(tokens_list_day) == 1:
            doc_node = [doc_id] * len(sample_words_day)
            wd_src += sample_words_day 
            wd_dst += doc_node
            wd_weight += [1.0/len(sample_words_day)] * len(sample_words_day)
            wd_time += [day_i*1.0]*len(doc_node)
        else:
            doc_node, word_node, weight = doc_word_tfidf(tokens_list_clean, sample_words_day)
            doc_node = [v+doc_id for v in doc_node]
            wd_src += word_node 
            wd_dst += doc_node
            wd_weight += weight
            wd_time += [day_i*1.0]*len(weight)

         # [word - word]
        word_i, word_j, weight = word_word_pmi_norm(tokens_list_day, sample_words_day, window_size=20)
        ww_src += word_i
        ww_dst += word_j
        ww_weight += weight
        ww_time += [day_i*1.0]*len(weight)

        # [topic - doc]
        doc_node, topic_node, weight = doc_topic_dist(tokens_list_day)
        doc_node = [v+doc_id for v in doc_node]
        td_src += topic_node
        td_dst += doc_node
        td_weight += weight
        td_time += [day_i*1.0]*len(weight)

        # word - topic
        topic_node, word_node, weight = topic_word_conn(sample_words_day,num_words=30) #need check words existed in topics
        wt_src += word_node
        wt_dst += topic_node
        wt_weight += weight
        wt_time += [day_i*1.0]*len(weight)

        doc_id += len(tokens_list_day)

    ww_src = [vocab_graph_node_map[v] for v in ww_src]
    ww_dst = [vocab_graph_node_map[v] for v in ww_dst]
    ww_src = torch.tensor(ww_src).view(-1)
    ww_dst = torch.tensor(ww_dst).view(-1)
    graph_data[('word','ww','word')] = (ww_src, ww_dst)
    ww_time = torch.tensor(ww_time).view(-1)
    ww_weight = torch.tensor(ww_weight).view(-1).float()
    

    wd_src = [vocab_graph_node_map[v] for v in wd_src]
    wd_src = torch.tensor(wd_src).view(-1)
    wd_dst = torch.tensor(wd_dst).view(-1)
    graph_data[('word','wd','doc')] = (wd_src, wd_dst)
    graph_data[('doc','dw','word')] = (wd_dst, wd_src)

    wd_time = torch.tensor(wd_time).view(-1)
    wd_weight = torch.tensor(wd_weight).view(-1).float()

    td_src = torch.tensor(td_src).view(-1)
    td_dst = torch.tensor(td_dst).view(-1)
    graph_data[('topic','td','doc')] = (td_src, td_dst)
    graph_data[('doc','dt','topic')] = (td_dst, td_src)

    td_time = torch.tensor(td_time).view(-1)
    td_weight = torch.tensor(td_weight).view(-1).float()

    graph_data[('topic','tt','topic')] = (torch.tensor(topic_i),torch.tensor(topic_j))
    
    wt_src = [vocab_graph_node_map[v] for v in wt_src]
    wt_src = torch.tensor(wt_src).view(-1)
    wt_dst = torch.tensor(wt_dst).view(-1)
    graph_data[('word','wt','topic')] = (wt_src, wt_dst)
    graph_data[('topic','tw','word')] = (wt_dst, wt_src)

    wt_time = torch.tensor(wt_time).view(-1)
    wt_weight = torch.tensor(wt_weight).view(-1).float()
    g = dgl.heterograph(graph_data)
    nodes2 = g.nodes('word')
    # print(g.num_nodes('word'),'words dynamic')
    # print(nodes2)
    # combined = torch.cat((nodes1, nodes2))
    # uniques, counts = combined.unique(return_counts=True)
    # difference = uniques[counts == 1]
    # intersection = uniques[counts > 1]
    # print(difference,difference.shape,'difference',intersection.shape,'intersection')
    if len(nodes1) != len(nodes2):
        words_in_curr_sample = [vocab_graph_node_map_reverse[v] for v in nodes2.numpy()]
        print('nodes1',len(nodes1), 'nodes2',len(nodes2), 'not the same',words_in_curr_sample)
    # exit()
    g.nodes['word'].data['id'] = torch.tensor(words_in_curr_sample).long()
    g.nodes['topic'].data['id'] = g.nodes('topic').long()

    g.edges['ww'].data['weight'] = ww_weight
    g.edges['ww'].data['time'] = ww_time
    g.edges['tt'].data['weight'] = edge_tt

    g.edges['wd'].data['weight'] = wd_weight
    g.edges['wd'].data['time'] = wd_time
    g.edges['td'].data['weight'] = td_weight
    g.edges['td'].data['time'] = td_time
    g.edges['wt'].data['weight'] = wt_weight
    g.edges['wt'].data['time'] = wt_time
    # inverse edges
    g.edges['dw'].data['weight'] = wd_weight
    g.edges['dw'].data['time'] = wd_time
    g.edges['dt'].data['weight'] = td_weight
    g.edges['dt'].data['time'] = td_time
    g.edges['tw'].data['weight'] = wt_weight
    g.edges['tw'].data['time'] = wt_time

    # norm_edges(g,ntype='word',etype='ww')
    # norm_edges(g,ntype='topic',etype='tt')
    # g.ids = {}
    # idx = 0
    # for id in words_in_curr_sample:
    #     g.ids[id] = idx
    #     idx += 1
    # print(g)
    # g = g.int()
    all_dynamic_g_list.append(g)
    #####
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
            ys.append(0) # TODO if involve other events to predict
    y_list.append(ys)  
    city_list.append(city)
    date_list.append(date)
    
    print('iii={} \t {} \t {} \t {} day_has_data \t  {} vocab {} doc {} ww sta-{} dyn-{}'.format(iii,date,city,num_nonzero_days,time.ctime(),len(sample_words),len(tokens_list),len(edge_ww),len(ww_weight)))
    # if iii >= 15:
    #     break

y_list = torch.tensor(y_list)
# save_graphs(dataset_path + "/data.bin", all_g_list, {"y":y_list})
print('static',len(all_static_g_list),'dynamic',len(all_dynamic_g_list),'y',len(y_list), 'date',len(date_list), 'city',len(city_list))
# exit()
attr_dict = {"y":y_list,"date":date_list,"city":city_list}

with open(outf_static,'wb') as f:
    pickle.dump(all_static_g_list, f)
print(outf_static, 'saved!')

with open(outf_dynamic,'wb') as f:
    pickle.dump(all_dynamic_g_list, f)
print(outf_dynamic, 'saved!')

with open(outf_attr,'wb') as f:
    pickle.dump(attr_dict, f)
print(outf_attr, 'saved!')

 