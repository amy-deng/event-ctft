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
    def __init__(self, h_dim, seq_len=7, num_topic=50, num_word=15000,dropout=0.5):
        super().__init__()
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        self.hconv = dglnn.HeteroGraphConv({
                        'wt' : dglnn.GraphConv(5,5),
                        'ww' : dglnn.GraphConv(5,5),
                        'wd' : dglnn.GraphConv(5,5),
                        'td' : dglnn.GraphConv(5,5),
                        'tt' : dglnn.GraphConv(5,5)},
                        # 'dw' : dglnn.SAGEConv(10,10),
                        # 'dt' : dglnn.SAGEConv(10,10),
                        # 'tt' : dglnn.SAGEConv(10,10)},
                        aggregate='sum')

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
        print(bg,'bg =====')
        print(bg.canonical_etypes)
        # h1 = {'word' : torch.randn((bg.number_of_nodes('word'), 5)),'topic' : torch.randn((bg.number_of_nodes('topic'), 5))}
        h1 = {'word' : torch.randn((bg.number_of_nodes('word'), 5))}
        # h1 = {'topic' : torch.randn((bg.number_of_nodes('topic'), 5))}
        h1 = {'doc' : torch.randn((bg.number_of_nodes('doc'), 5)),
        'word' : torch.randn((bg.number_of_nodes('word'), 5)),
        'topic' : torch.randn((bg.number_of_nodes('topic'), 5))}
        r = self.hconv(bg,h1)
        print(r,'rrrrr',r.keys())
        # batch graphs
        # graph propagation
        # pred, idx, _ = self.__get_pred_embeds(t_list)
        # loss = self.criterion(pred, true_prob_r[idx])
        return 


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

