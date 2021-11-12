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
    import dgl.function as fn
    import dgl.nn.pytorch as dglnn
except:
    print("<<< dgl are not imported >>>")
    pass
 
 
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)): 
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        # self.semantic_attention = SemanticAttention(out_size, out_size)
        self.semantic_attention = SemanticAttention(out_size * layer_num_heads,out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.norm = nn.LayerNorm(out_size* layer_num_heads,elementwise_affine=False)

    def forward(self, g, feat_dict):
        semantic_embeddings_doc = []
        semantic_embeddings_topic = []
        # print('self.meta_paths',self.meta_paths)
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                # print(meta_path,'meta_path')
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            # print(meta_path)
            # continue
            # print(h.shape,new_g.num_nodes())
            # v = self.gat_layers[i](new_g, h).flatten(1)
            # print(v.shape,'v')
            if meta_path[0] == 'ww':
                r = self.gat_layers[i](new_g, feat_dict['word']).flatten(1)
                # print(r.shape,'rrrrr')
                semantic_embeddings_word = r
            elif meta_path[0] == 'tt':
                r = self.gat_layers[i](new_g, feat_dict['topic']).flatten(1)
                # print(r.shape,'rrrrr')
                semantic_embeddings_topic.append(r)
                # semantic_embeddings.append(r)
            elif meta_path[0] == 'wt':
                r = self.gat_layers[i](new_g, (feat_dict['word'], feat_dict['topic'])).flatten(1)
                # print(r.shape,'rrrrr')
                semantic_embeddings_topic.append(r)
                # semantic_embeddings.append(r)
            elif meta_path[0] == 'td':
                r = self.gat_layers[i](new_g, (feat_dict['topic'], feat_dict['doc'])).flatten(1)
                # print(r.shape,'rrrrr')
                semantic_embeddings_doc.append(r)
                # semantic_embeddings.append(r)
            elif meta_path[0] == 'wd':
                r = self.gat_layers[i](new_g, (feat_dict['word'], feat_dict['doc'])).flatten(1)
                # print(r.shape,'rrrrr')
                # semantic_embeddings.append(r)
                semantic_embeddings_doc.append(r)
        semantic_embeddings_doc = torch.stack(semantic_embeddings_doc, dim=1)  
        semantic_embeddings_doc = self.semantic_attention(semantic_embeddings_doc) 
        # print(semantic_embeddings_doc.shape,'semantic_embeddings_doc')

        semantic_embeddings_topic = torch.stack(semantic_embeddings_topic, dim=1)  
        semantic_embeddings_topic = self.semantic_attention(semantic_embeddings_topic) 
        # print(semantic_embeddings_topic.shape,'semantic_embeddings_topic')
        # semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  
        # print(semantic_embeddings.shape,'semantic_embeddings====')
        feat_dict = {'word':self.norm((semantic_embeddings_word)), 
            'doc':self.norm((semantic_embeddings_doc)), 
            'topic':self.norm((semantic_embeddings_topic))}
        # print(feat_dict,'feat_dict')
        return feat_dict                          

# class HAN(nn.Module):
#     def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
#         super(HAN, self).__init__()

#         self.layers = nn.ModuleList()
#         self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
#         for l in range(1, len(num_heads)):
#             self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
#                                         hidden_size, num_heads[l], dropout))
#         self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

#     def forward(self, g, h):
#         for gnn in self.layers:
#             h = gnn(g, h)

#         return self.predict(h)

class HANNet(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
    # def __init__(self, in_size, hidden_size, num_heads, device, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
    # def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HANNet, self).__init__()
        # meta_paths = None
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size//num_heads[0], num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size ,
                                        hidden_size// num_heads[l-1], num_heads[l], dropout))
        # self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return h


class HAN(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, activation, device, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(HAN, self).__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1,n_hid))
        # meta_paths=[('topic', 'doc'), ('topic', 'topic'), ('word', 'doc'), ('word', 'topic'), ('word', 'word')]
        meta_paths = [['ww'],['wd'],['tt'],['td'],['wt']]
        self.adapt_ws = nn.Linear(n_inp, n_hid)
        self.han = HANNet(meta_paths, n_hid, n_hid, n_hid, [4,4,4,4,4], dropout)
        self.out_layer = nn.Sequential(
                # nn.Linear(n_hid*3, n_hid),
                # nn.BatchNorm1d(n_hid),
                nn.Linear(n_hid, 1) 
        )
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
        bg = dgl.batch(g_list).to(self.device) 
        # print(bg.metagraph().edges())
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        # torch.zeros((bg.number_of_nodes('doc'), self.n_hid)).to(self.device)
        # for i in range(len(self.gcn_topic_layers)):
        # print('word_emb',word_emb.shape,topic_emb.shape,'doc_emb',doc_emb.shape)
        word_emb = self.adapt_ws(word_emb)
        # print('word_emb',word_emb.shape,topic_emb.shape)
        # feat_dict = {'word':word_emb, 'topic':topic_emb}#, 'doc':doc_emb}
        # for layer in self.hetero_layers:
        #     feat_dict = layer(bg, feat_dict)
        feat_dict = {'word':word_emb, 'topic':topic_emb, 'doc':doc_emb}
        feat_dict = self.han(bg, feat_dict)
        # bg.nodes['word'].data['h'] = torch.tanh(self.adapt_ws(word_emb))
        bg.nodes['word'].data['h'] = feat_dict['word']
        bg.nodes['topic'].data['h'] = feat_dict['topic']
        bg.nodes['doc'].data['h'] = feat_dict['doc']
        # print(feat_dict['word'].shape, feat_dict['topic'].shape)
        # for i in range(self.n_layers):
        #     self.gcs[i](bg, 'h', 'h')
        # attention on words on topics on docs??
        
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='h',ntype='doc')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='h',ntype='doc')
        # print(global_doc_info,'global_doc_info')
        y_pred = self.out_layer(global_doc_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred


class HANAll(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, activation, device, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(HANAll, self).__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1,n_hid))
        # meta_paths=[('topic', 'doc'), ('topic', 'topic'), ('word', 'doc'), ('word', 'topic'), ('word', 'word')]
        meta_paths = [['ww'],['wd'],['tt'],['td'],['wt']]
        self.adapt_ws = nn.Linear(n_inp, n_hid)
        self.han = HANNet(meta_paths, n_hid, n_hid, n_hid, [4,4,4,4,4], dropout)
        self.out_layer = nn.Sequential(
                nn.Linear(n_hid*3, n_hid),
                nn.BatchNorm1d(n_hid),
                nn.Linear(n_hid, 1) 
        )
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
        bg = dgl.batch(g_list).to(self.device) 
        # print(bg.metagraph().edges())
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        # torch.zeros((bg.number_of_nodes('doc'), self.n_hid)).to(self.device)
        # for i in range(len(self.gcn_topic_layers)):
        # print('word_emb',word_emb.shape,topic_emb.shape,'doc_emb',doc_emb.shape)
        word_emb = self.adapt_ws(word_emb)
        # print('word_emb',word_emb.shape,topic_emb.shape)
        # feat_dict = {'word':word_emb, 'topic':topic_emb}#, 'doc':doc_emb}
        # for layer in self.hetero_layers:
        #     feat_dict = layer(bg, feat_dict)
        feat_dict = {'word':word_emb, 'topic':topic_emb, 'doc':doc_emb}
        feat_dict = self.han(bg, feat_dict)
        # bg.nodes['word'].data['h'] = torch.tanh(self.adapt_ws(word_emb))
        bg.nodes['word'].data['h'] = feat_dict['word']
        bg.nodes['topic'].data['h'] = feat_dict['topic']
        bg.nodes['doc'].data['h'] = feat_dict['doc']
        # print(feat_dict['word'].shape, feat_dict['topic'].shape)
        # for i in range(self.n_layers):
        #     self.gcs[i](bg, 'h', 'h')
        # attention on words on topics on docs??
        
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='h',ntype='doc')
            global_word_info = dgl.max_nodes(bg, feat='h',ntype='word')
            global_topic_info = dgl.max_nodes(bg, feat='h',ntype='topic')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='h',ntype='doc')
            global_word_info = dgl.mean_nodes(bg, feat='h',ntype='word')
            global_topic_info = dgl.mean_nodes(bg, feat='h',ntype='topic')
        global_info = torch.cat((global_doc_info, global_word_info, global_topic_info),-1)
        # print(global_info.shape,'global_info')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred


# Generally, every node in a heterogeneous graph contains multiple types of semantic information