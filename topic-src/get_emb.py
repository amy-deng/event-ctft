# use env xgraph
import sent2vec
import pickle
model = sent2vec.Sent2vecModel()
# cd /home/sdeng/data/word_embbedding
name='icews-full-cbow-c+w-ngrams-300.bin'
model.load_model(name)

ngram_path = '/home/sdeng/data/icews/corpus/ngrams/AFG_1gram_tfidf.txt'
ngram_path = '/home/sdeng/data/icews/corpus/ngrams/RUS_from2012_1gram_tfidf.txt'
with open(ngram_path,'r') as f:
    vocab = f.read().splitlines()
print(len(vocab))
top_k_ngram=15000

vocab = vocab[:top_k_ngram]
print('vocab loaded',len(vocab))
uni_embs = model.embed_unigrams(vocab)
print(uni_embs.shape)
print(uni_embs.nonzero()[0].shape[0] // 300)
with open('/home/sdeng/workspace/event-ctft/data/AFG_w7h7_minday7/word_emb_300.pkl','wb') as f:
    pickle.dump(uni_embs,f)


with open('/home/sdeng/workspace/event-ctft/data/RUS_w7h7_mind3n10df0.01/word_emb_300.pkl','wb') as f:
    pickle.dump(uni_embs,f)