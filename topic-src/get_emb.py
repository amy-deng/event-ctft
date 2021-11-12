# use env xgraph
import sent2vec
import pickle
model = sent2vec.Sent2vecModel()
name='icews-full-cbow-c+w-ngrams-300.bin'
model.load_model(name)

ngram_path = '/home/sdeng/data/icews/corpus/ngrams/RUS_1gram_tfidf.txt'
with open(ngram_path,'r') as f:
    vocab = f.read().splitlines()
top_k_ngram=15000
vocab = vocab[:top_k_ngram]
print('vocab loaded',len(vocab))
uni_embs = model.embed_unigrams(vocab)

with open('/home/sdeng/workspace/event-ctft/data/RUS_w7h7_minday3/word_emb_300.pkl','wb') as f:
    pickle.dump(uni_embs,f)