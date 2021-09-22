# -*- coding: utf-8 -*-
# from gensim.models import Word2Vec

from nltk.util import pr
import pandas as pd
import argparse
import time
from text_utils import *
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import os, string, re
# from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS, original_scorer
import nltk
'''
get tokenized sentences
'''
 
 
    
def main(args):
     
    story_file = os.path.join(args.story_path,"icews_news_{}.json".format(args.country))
    df = pd.read_json(story_file,lines=True)
    print('# stories ',len(df))
    start_date = args.start_year + '-01-01'
    df = df.loc[df['Date']>start_date]
    print('# stories (after {})'.format(start_date),len(df))
    story_l = df['Text'].values

    processed_tokens = clean_document_list(story_l)
    # processed_tokens = process_texts(story_l)
    print('# processed_tokens',len(processed_tokens))
    '''
    ignore_set = ENGLISH_CONNECTOR_WORDS

    # phrases = Phrases(token_lists, min_count=5, threshold=0.2, scoring="npmi", connector_words=ignore_set)
    phrases = Phrases(processed_tokens, connector_words=ignore_set, delimiter='-',threshold=15)
    frozen_model = phrases.freeze()
    phrase_model_name = "{}/{}_phrases2_from_{}_model.pkl".format(args.result_path,args.country,args.start_year)
    frozen_model.save(phrase_model_name)
    print('phrases done saved')
    model_reloaded = Phrases.load(phrase_model_name)
    print('phrases done load')
    print(model_reloaded['united','state','tear','gas','middle','class'])
    # exit()
    '''
    with open("{}/{}_doc_tokens_from_{}.txt".format(args.result_path,args.country,args.start_year), 'w') as f:
        for sentence in processed_tokens:
            # phrase_tokens = phrases[sentence]
            f.write(' '.join(sentence)+'\n')
            # sentence_str = ' '.join(sentence)
            # f.write(sentence_str+'\n')
    print(time.ctime(), 'done')
    # model = Word2Vec(processed_tokens, size=args.dim, min_count=5, sg=1, workers=24, iter=10) # sg:1 for skip-gram,otherwise CBOW.
    # model.save(outf)
    # print(time.ctime(), "word_vectors file saved ")
    # load model
    # new_model = Word2Vec.load('model.bin')
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_path", default="/home/sdeng/data/icews/corpus", help="path of result")
    ap.add_argument("--story_path", default="/home/sdeng/data/icews/news.1991.201703.country/", help="path of story")
    # ap.add_argument('--story_file', type=str, default='icews_news_THA.json', help="news file")
    ap.add_argument("--country", default='THA', help='country code')
    # ap.add_argument("--dim", type=int, default=100, help='word embedding dimension')
    # ap.add_argument("--ids_file", default='IND_storyids_2012_2016.txt', help='story ids file name')
    ap.add_argument("--start_year", type=str, default='2010', help='start and end year, default 2012')
    args = ap.parse_args()                                              
    print (args)
    main(args)
    # different country, different language, so train w2v for each one


   