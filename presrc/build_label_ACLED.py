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
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS, original_scorer
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
print(os.getcwd())


'''
graph_labels = {"label": torch.Tensor(graph_label).int(),'time':torch.Tensor(graph_time).int()}
save_graphs(path+"data.bin", graph_list, graph_labels)
'''
try:
    RAWDATA = sys.argv[1]
    # DATASET = sys.argv[1]
    # STARTTIME = str(sys.argv[2])
    # ENDTIME= str(sys.argv[3])
    # DELTA = int(sys.argv[2])
    WINDOW = int(sys.argv[2])
    HORIZON = int(sys.argv[3])
    PREDWINDOW = int(sys.argv[4])
except:
    print("Usage: RAWDATA, WINDOW=14, HORIZON=1, PREDWINDOW=3")
    exit()


DATASET = RAWDATA.split('-')[7][:-4]
print(DATASET,'dataset_name')

filename = RAWDATA.split('/')[-1]
print('path',RAWDATA,'filename',filename)
start_year = int(filename.split('-')[0])
start_month = int(filename.split('-')[1])
start_day = int(filename.split('-')[2])
end_year = int(filename.split('-')[3])
end_month = int(filename.split('-')[4])
end_day = int(filename.split('-')[5])


path = '../data/{}/'.format(DATASET)
os.makedirs(path, exist_ok=True)
print('path',path)



df = pd.read_csv(RAWDATA,sep=';')
df = df.drop_duplicates(subset=['data_id'], keep='first')
df['event_date'] = pd.to_datetime(df['event_date'])



DELTA = 1
# date and get protest count
start = date(start_year,start_month,start_day)
end = date(end_year,end_month,end_day)
delta = timedelta(days=DELTA)
date_table = {}
i = 0
while start <= end:
    start_date_str = start.strftime("%Y-%m-%d")
    date_table[start_date_str] = i
    i += 1
    start += delta




subevents_df = pd.read_csv(path + "subevent2id.txt",names=['id','name'],sep='\t')
subevents = subevents_df['name'].unique()
subevents.sort()
subevent_count_dict = {}
for v in subevents:
    subevent_count_dict[v] = np.array([0 for i in range(len(date_table))])
start = date(start_year,start_month,start_day)
end = date(end_year,end_month,end_day)
dayi = 0
while start <= end:
    start_date_str = start.strftime("%Y-%m-%d")
    df_day = df.loc[df['event_date'] == start_date_str]
    df_count = df_day['sub_event_type'].value_counts().rename_axis('unique_values').reset_index(name='counts')
    for i,row in df_count.iterrows():
        subevent_count_dict[row['unique_values']][dayi] = row['counts']
    dayi += 1
    start += delta
Protests_count = subevent_count_dict['Protest with intervention'] + subevent_count_dict['Peaceful protest'] + subevent_count_dict['Excessive force against protesters']

# build sequence data
subevent_count_seq = []
for k in subevents:
    v = subevent_count_dict[k].tolist()
    subevent_count_seq.append(v)
subevent_count_seq = np.array(subevent_count_seq)
subevent_count_seq = np.swapaxes(subevent_count_seq,0,1)
# print(subevent_count_seq.shape,'subevent_count_seq')
 

# get label and Y, and corresponding time
date_ids = list(date_table.values())
date_name = list(date_table.keys())
date_table_rev = dict(zip(date_ids,date_name))
data_time = []
data_Y = []
data_treat = []
data_X = []
data_text = []

for i in range(WINDOW,len(date_ids),PREDWINDOW): # no overlap of pre_window
    y_start = i+WINDOW+HORIZON-1
    y_end = i+WINDOW+PREDWINDOW+HORIZON-1
    if y_start >=len(date_ids):
        break
    # treat
    last = subevent_count_seq[i-WINDOW:i]
    curr = subevent_count_seq[i:i+WINDOW]
    data_X.append(curr)
    treat = curr.sum(0) - last.sum(0)
    data_treat.append(list(np.where(treat>0,1,0)))
    
    # label
    protest = Protests_count[y_start:y_end].sum()
    data_Y.append(1 if protest > 0 else 0)
    # time
    data_time.append(date_ids[i+WINDOW])
    # text
    date_list = [date_table_rev[j] for j in range(i,i+WINDOW)]
    df_window = df.loc[df['event_date'].isin(date_list)]['Event Sentence']
    data_text.append(' '.join(df_window.values))



# for i in range(WINDOW,len(date_ids),HORIZON+PREDWINDOW-1): # no overlap of pre_window
#     if i+WINDOW >=len(date_ids) or i+WINDOW+PREDWINDOW-1 >= len(date_ids):
#         break
#     # treat
#     last = subevent_count_seq[i-WINDOW:i]
# #     print(i-WINDOW,i,'---',i,i+WINDOW,'   yyy',i+WINDOW,i+WINDOW+PREDWINDOW-1)
#     curr = subevent_count_seq[i:i+WINDOW]
#     data_X.append(curr)
#     treat = curr.sum(0) - last.sum(0)
#     data_treat.append(list(np.where(treat>0,1,0)))
#     # label
#     # print(i+WINDOW,i+WINDOW+PREDWINDOW-1)
#     protest = Protests_count[i+WINDOW:i+WINDOW+PREDWINDOW].sum()
#     data_Y.append(1 if protest > 0 else 0)
#     # time
#     data_time.append(date_ids[i+WINDOW])
#     # text
#     date_list = [date_table_rev[j] for j in range(i,i+WINDOW)]
#     df_window = df.loc[df['event_date'].isin(date_list)]['notes']
#     data_text.append(' '.join(df_window.values))

    

# to build counter factual data
data_X = np.stack(data_X) # t,window,#subevent
data_treat = np.array(data_treat)
 
print('data_time',len(data_time),'data_Y',len(data_Y),data_X.shape,data_treat.shape, len(data_text))


######
# for all samples,
# 1. make time series smooth, moving average
# 2. get tfidf matrix using text data

def movingaverage(a, n=3) :
    padding = []
    for i in range(n-1):
        padding.append(a[:i+1].mean())
    padding = np.array(padding)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.concatenate((padding, ret[n - 1:] / n),0)
data_X_smooth = [] # 549,10,24
d1,d2,d3 = data_X.shape
for i in range(d1):
    tmp = []
    for j in range(d2):
        tmp.append(list(movingaverage(data_X[i,j,:],WINDOW//2)))
    data_X_smooth.append(tmp)
data_X_smooth = np.array(data_X_smooth)
print('data_X_smooth',data_X_smooth.shape,'data_X',data_X.shape)



def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def get_sep_token_list(sentences):
    sentences = re.sub(r'\[[^)]*\]', '',sentences)
    sentences = re.sub(r"[-()*$\"#%/@;:'<>{}`+=~|]", '', sentences)
    sentences = sentences.strip().lower()
    # print(sentences)
    tokens = nltk.word_tokenize(sentences)
    tagged = nltk.pos_tag(tokens)
    # print('tagged:',tagged)
    wordnet_lemmatizer = WordNetLemmatizer()
    out_tokens = []
    sents,sent = [], []
    for (word,tag) in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:
            w = wordnet_lemmatizer.lemmatize(word)
        else:
            w = wordnet_lemmatizer.lemmatize(word, pos=wntag)
        if w not in [',','.','?','!','']:# and w not in stopwords.words('english'):
            sent.append(w)
        elif len(sent) > 0:
            sents.append(sent)
            sent = []
    return sents

def get_lem_token_list(sentences):
    sentences = re.sub(r'\[[^)]*\]', '',sentences)
    sentences = re.sub(r"[-()*$\"#%/@;:'<>{}`+=~|]", '', sentences)
    sentences = sentences.strip().lower()
    out_tokens = []
    # print(sentences)
    tokens = nltk.word_tokenize(sentences)
    tagged = nltk.pos_tag(tokens)
    # print('tagged:',tagged)
    wordnet_lemmatizer = WordNetLemmatizer()
    for (word,tag) in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:
            w = wordnet_lemmatizer.lemmatize(word)
        else:
            w = wordnet_lemmatizer.lemmatize(word, pos=wntag)
        if w not in [',','.','?','!'] and w not in stopwords.words('english') and w.isalpha() : # did notremove adv and other words
            out_tokens.append(w)
    return out_tokens

# all_sent_list = []
# for v in data_text:
#     sent_list = get_sep_token_list(v)
#     all_sent_list += sent_list
# ignore_set = ENGLISH_CONNECTOR_WORDS
# phrases = Phrases(all_sent_list, min_count=5, threshold=0.2, scoring="npmi", connector_words=ignore_set)

with open('/home/sdeng/data/stopwords-en-basic.txt','r') as f:
    stop_words = f.read().splitlines()
stop_words += ['aren', 'can', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 'placeholder', 're', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']
all_tokens = []
k = 0
for sentences in data_text:
    tokens = get_lem_token_list(sentences)
    all_tokens.append(tokens)
    if k % 100 == 0:
        print('processing text k = {}'.format(k))
    k+=1

vectorizer = TfidfVectorizer(tokenizer=(lambda x:x), lowercase=False, stop_words=stop_words,min_df=5,max_df=0.99)#token_pattern=r'(?u)\b\w*[a-zA-Z]\w*\b') # stopwords='english' u'(?u)\b\w*[a-zA-Z]\w*\b
tfidf = vectorizer.fit_transform(all_tokens)
print('tfidf',tfidf.shape)

feature_names = vectorizer.get_feature_names()
tfidf_vocab = path + 'tfidf_vocab.txt'
tfidf_vocab_f = open(tfidf_vocab, 'w')
for i in range(len(feature_names)):
    tfidf_vocab_f.write("{}\n".format(feature_names[i]))
tfidf_vocab_f.close()


with open(path+'tmp_label_w{}_h{}_p{}.pkl'.format(WINDOW,HORIZON,PREDWINDOW),'wb') as f:
    pickle.dump([data_time,data_Y,data_X,data_X_smooth,data_treat,tfidf],f)
print('tmp data saved')

# for each samples, find a cf data, 
# TODO
# for i in range(data_treat)
# multiple treatment..... TODO