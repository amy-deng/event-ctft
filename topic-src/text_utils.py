# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, os
import torch
# import nltk
import re
import string
import torch
import torch.nn.functional as F
# from ignite.metrics import Precision, Recall, Accuracy
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
# from sklearn.feature_extraction import stop_words
# from gensim.models import Word2Vec
# from gensim.models import KeyedVectors

 
# text process functions

def nltk_lemmatize(word_l): 
    ''' full-form '''
    return [WordNetLemmatizer().lemmatize(i) for i in word_l]


def nltk_stem(word_l):
    return [PorterStemmer().stem(i) for i in word_l]


def get_stopwords():
    file = '/home/sdeng/data/stopwords-en.txt'
    assert check_exist(file), "can not find stopwords file {}".format(file)
    return open(file).read().split('\n')


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def text_to_phrases_pipeline(text):
    pass

def text_to_words_pipeline(text):
    stop_words = get_stopwords()
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    stripped = [w.translate(string.punctuation) for w in tokens]
    words = [word for word in stripped if word.isalpha() and is_english(word)]
    words = [w for w in words if not w in stop_words]
    # words = nltk_stem(words) #nltk_lemmatize
    words = nltk_lemmatize(words) #nltk_lemmatize
    
    # words = [w for w in words if not w in stop_words] # final remove
    return words

def process_texts(texts):
    ''' process texts list '''
    l = []
    for t in texts:
        l.append(text_to_words_pipeline(t))
    return l

def remove_noise(word):
    if set(word) == '-': return False
    if '-' in word:
        strs = word.split('-')
        for s in strs:
            if not s.isalpha() or not is_english(word):
                return False
        return True
    return word.isalpha() or not is_english(word)


def sentence_tokenize(text):
    stop_words = get_stopwords()
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    stripped = [w.translate(string.punctuation) for w in tokens]
    words = [word for word in stripped if not word.isnumeric() and remove_noise(word)] # word.isalpha() and is_english(word)]
    # words = [word for word in stripped if not word.isnumeric() and word not in "(){}[]'':.;,*&#@-_?"]#['(',')','[',']',':','.',"''"]]
    words = [w for w in words if not w in stop_words]
    # words = nltk_stem(words) #nltk_lemmatize
    words = nltk_lemmatize(words) #nltk_lemmatize
    # words = [w for w in words if not w in stop_words] # final remove
    return words
    
def text_tokenize(text): # list of list
    lists = []
    text = re.sub(r"''", "\\n",text) 
    lists += text.split("\\n")
    token_lists = []
    for sent in lists:
        token_lists.append(sentence_tokenize(sent))
    return token_lists

def process_texts_phrases(texts): # list of list
    l = []
    for t in texts:
        l += text_tokenize(t)
    return l

def clean_document(text):
    text = re.sub(r"''", " ",text) 
    text = re.sub(r"\\n", " ",text) 
    return sentence_tokenize(text)

def clean_document_list(texts):
    l = []
    for t in texts:
        l.append(clean_document(t))
    return l 


def clean_document_str(text):
    text = re.sub(r"''", " ",text) 
    text = re.sub(r"\\n", " ",text) 
    return ' '.join(sentence_tokenize(text))

def clean_document_list_str(texts):
    l = []
    for t in texts:
        l.append(clean_document_str(t))
    return l 

def check_exist(outf):
    return os.path.isfile(outf)


 