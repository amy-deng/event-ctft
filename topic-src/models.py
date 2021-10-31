import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import os, math
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import torch
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
import time
# from layers import *
# from utils import *
# from sparsemax import Sparsemax
# from tcn import *
try:
    import dgl
    import dgl.nn.pytorch as dglnn
except:
    print("<<< dgl are not imported >>>")
    pass


# a static graph model
class static_heto_graph(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, seq_len=7, num_topic=50, num_word=15000,dropout=0.5):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        self.hconv = dglnn.HeteroGraphConv({
                        'wt' : dglnn.GraphConv(h_inp,h_dim),
                        'ww' : dglnn.GraphConv(h_inp,h_dim),
                        'wd' : dglnn.GraphConv(h_inp,h_dim),
                        'td' : dglnn.GraphConv(h_dim,h_dim),
                        'tt' : dglnn.GraphConv(h_dim,h_dim)},
                        # 'dw' : dglnn.SAGEConv(10,10),
                        # 'dt' : dglnn.SAGEConv(10,10),
                        # 'tt' : dglnn.SAGEConv(10,10)},
                        aggregate='sum')
        self.maxpooling  = nn.MaxPool1d(3)# 
        # self.maxpooling  = dglnn.MaxPooling()
        self.out_layer = nn.Linear(h_dim,1)
        # self.word_embeds = None
        # self.global_emb = None  
        # self.ent_map = None
        # self.rel_map = None
        # self.word_graph_dict = None
        # self.graph_dict = None
        # self.aggregator= aggregator_event(h_dim, dropout, num_ents, num_rels, seq_len, maxpool, attn)
        # if use_gru:
        #     self.encoder = nn.GRU(3*h_dim, h_dim, batch_first=True)
        # else:
        #     self.encoder = nn.RNN(3*h_dim, h_dim, batch_first=True)
        # self.linear_r = nn.Linear(h_dim, self.num_rels)

        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = F.binary_cross_entropy_with_logits #soft_cross_entropy
        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, g_list, y_data): 
        bg = dgl.batch(g_list)
        # unbatch???? get emb of lists
        # print(bg,'bg =====')
        # print(bg.canonical_etypes)
        # h1 = {'word' : torch.randn((bg.number_of_nodes('word'), 5)),'topic' : torch.randn((bg.number_of_nodes('topic'), 5))}
        # h1 = {'word' : torch.randn((bg.number_of_nodes('word'), 5))}
        # h1 = {'topic' : torch.randn((bg.number_of_nodes('topic'), 5))}
        # bg.nodes['word'].data['h'] 
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        # topic_emb = self.topic_embeds[bg.nodes('topic')]
        print(bg.nodes['topic'].data['id'])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = torch.zeros((bg.number_of_nodes('doc'), self.h_dim))
        emb_dict = {
            'word':word_emb,
            'topic':topic_emb,
            'doc':doc_emb
        }
        # h1 = {'doc' : torch.randn((bg.number_of_nodes('doc'), 5)),
        #      'word' : torch.randn((bg.number_of_nodes('word'), 5)),
        #     'topic' : torch.randn((bg.number_of_nodes('topic'), 5))}
        r = self.hconv(bg,emb_dict)
        # print('rrrrr',r.keys())
        doc_emb = r['doc']
        print(doc_emb.shape,'doc_emb')
        doc_len = [g.num_nodes('doc') for g in g_list]
        doc_emb_split = torch.split(doc_emb, doc_len)
        print(len(doc_emb_split),'doc_emb_split',doc_emb_split[0].shape)
        # padding to same size  
        print(max(doc_len),'max(doc_len)')
        embed_pad_tensor = torch.zeros(len(doc_len), max(doc_len), self.h_dim)
        for i, embeds in enumerate(doc_emb_split): 
                embed_pad_tensor[i, torch.arange(0,len(embeds)), :] = embeds
        print(embed_pad_tensor.shape,'embed_pad_tensor') # batch,max # doc, f 

        # doc_pool = self.maxpooling(bg,doc_emb)
        # sub_g = dgl.edge_type_subgraph(bg, [('topic', 'td', 'doc')])
        # sub_g = dgl.edge_type_subgraph(bg, [('topic', 'tt', 'topic')])
        # doc_emb = r['topic']
        # doc_pool = self.maxpooling(dgl.to_homogeneous(sub_g), doc_emb)
        # print(doc_pool,doc_pool.shape,'doc_pool')

        # doc_emb_split = torch.split(doc_pool, [1 for i in range(len(g_list))])
        # print(doc_emb_split,'doc_emb_split')
        doc_pool = embed_pad_tensor.mean(1)
        # doc_pool = self.maxpooling(embed_pad_tensor)
        print(doc_pool.shape,'doc_pool')
        # doc_emb_mean = doc_emb.mean(0)
        y_pred = self.out_layer(doc_pool)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        # out layer
        # batch graphs
        # graph propagation
        # pred, idx, _ = self.__get_pred_embeds(t_list)
        # loss = self.criterion(pred, true_prob_r[idx])
        return loss, y_pred


    def __get_pred_embeds(self, t_list):
        sorted_t, idx = t_list.sort(0, descending=True)  
        embed_seq_tensor, len_non_zero = self.aggregator(sorted_t, self.ent_embeds, 
                                    self.rel_embeds, self.word_embeds, 
                                    self.graph_dict, self.word_graph_dict, 
                                    self.ent_map, self.rel_map)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               len_non_zero,
                                                               batch_first=True)
        _, feature = self.encoder(packed_input)
        feature = feature.squeeze(0)

        if torch.cuda.is_available():
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1)).cuda()), dim=0)
        else:
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1))), dim=0)
        
        pred = self.linear_r(feature)
        return pred, idx, feature
        
    def predict(self, t_list, true_prob_r): 
        pred, idx, feature = self.__get_pred_embeds(t_list)
        if true_prob_r is not None:
            loss = self.criterion(pred, true_prob_r[idx])
        else:
            loss = None
        return loss, pred, feature

    def evaluate(self, t, true_prob_r):
        loss, pred, _ = self.predict(t, true_prob_r)
        prob_rel = self.out_func(pred.view(-1))
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)  
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss

 

# a temporal graph model

