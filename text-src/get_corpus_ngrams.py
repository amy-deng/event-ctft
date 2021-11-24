import pandas as pd
import numpy as np
import sys, os
import pickle
import glob
from gensim import corpora, models, similarities
from gensim.models.ldamulticore import LdaMulticore,LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_corpus, common_dictionary
# from nltk import ngrams, FreqDist
from nltk.util import ngrams
from nltk.lm import NgramCounter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


'''
python get_corpus_ngrams.py '/home/sdeng/data/icews/corpus/THA_doc_tokens_from_2010.txt' 2 /home/sdeng/data/icews/corpus/ngrams year
'''

try:
    corpus_path = sys.argv[1]
    # topic_model_name = sys.argv[2] # THA
    ngram = int(sys.argv[2])
    out_path = sys.argv[3]
    year = sys.argv[4]
except:
    print("usage: <corpus_path (abs)> <ngram> <out_path `/home/sdeng/data/icews/corpus/ngrams`> <year>")
    exit()

if not os.path.exists(out_path):
    print(out_path, 'not exist')
    exit()

# load corpus
with open(corpus_path,'r') as f:
    corpus = f.read().splitlines()

country = corpus_path.split('/')[-1][:3]
print('country',country)
# out_file = "{}/{}_{}gram_tfidf.txt".format(out_path,country,ngram)
out_file = "{}/{}_from{}_{}gram_tfidf.txt".format(out_path,country,year,ngram)
print('out_file',out_file)
c_vec = TfidfVectorizer(ngram_range=(1, ngram),stop_words='english', min_df=20) # before add from{} is 20, RUS 20

ngrams = c_vec.fit_transform(corpus)

vocab = c_vec.vocabulary_

count_values = ngrams.toarray().sum(axis=0)
sorted_tfidf_ngrams_tuple = sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
sorted_tfidf_ngrams = [a_tuple[1] for a_tuple in sorted_tfidf_ngrams_tuple]
print('sorted_tfidf_ngrams',len(sorted_tfidf_ngrams),sorted_tfidf_ngrams[:10])

with open(out_file, "w") as outfile:
    outfile.write("\n".join(sorted_tfidf_ngrams))