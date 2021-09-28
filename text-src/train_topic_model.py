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

'''
python train_topic_model.py '/home/sdeng/data/icews/corpus/THA_doc_tokens_from_2010.txt' THA 50 /home/sdeng/data/icews/topic_models
'''

try:
    corpus_path = sys.argv[1]
    topic_model_name = sys.argv[2] # THA
    num_topic = int(sys.argv[3])
    out_path = sys.argv[4]
except:
    print("usage: <corpus_path (abs)> <topic_model_name `THA`> <num_topic> <out_path `/home/sdeng/data/icews/topic_models`> ")
    exit()

if not os.path.exists(out_path):
    print(out_path, 'not exist')
    exit()

# load corpus
with open(corpus_path,'r') as f:
    corpus = f.read().splitlines()

corpus_tokenized = []
for text in corpus:
    corpus_tokenized.append(text.split(' '))
print('corpus_tokenized processed')

dictionary_file = '{}/{}.dict'.format(out_path,topic_model_name)
if not os.path.exists(dictionary_file):
    corpus_dictionary = Dictionary(corpus_tokenized)
    corpus_dictionary.save('{}/{}.dict'.format(out_path,topic_model_name))
    print(dictionary_file,'saved')
else:
    corpus_dictionary = Dictionary.load(dictionary_file)
    print(dictionary_file,'existed')


corpus_bow = [corpus_dictionary.doc2bow(text) for text in corpus_tokenized]
lda = LdaModel(corpus_bow, num_topics=num_topic)
lda_file = '{}/{}_{}.lda'.format(out_path,topic_model_name, num_topic)
lda.save(lda_file)
print(lda_file,'saved')