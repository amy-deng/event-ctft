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

class HeteroLayer_orig(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroLayer_orig, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                # name : nn.Linear(in_size, out_size) for name in etypes
                'ww': nn.Linear(in_size, out_size),
                'wt': nn.Linear(out_size, out_size),
                'wd': nn.Linear(out_size, out_size),
                'td': nn.Linear(out_size, out_size),
                'tt': nn.Linear(out_size, out_size),
            }) 
    def forward(self, G, feat_dict):
        print(G,feat_dict,'G,feat_dict')
        # The input is a dictionary of node features for each type
        funcs={}
        for srctype, etype, dsttype in [['word','ww','word']]:
            # print('srctype, etype, dsttype',srctype, etype, dsttype)
            # edge_weight = G.edges['ww'].data['weight']
            # print('srctype',srctype,feat_dict.keys())
            # print(feat_dict[srctype].shape,'feat_dict[srctype]')
            # Wh = self.weight[etype](feat_dict[srctype])
            # print(Wh.shape,'Wh')
            # G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))
            # G.update_all(fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'), 'sum')
            # then other layer
            Wh = self.weight[etype](G.nodes['word'].data['h'])
            G.nodes[srctype].data['h'] = Wh
            funcs[etype] = (fn.u_mul_e('h', 'weight', 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        # feat_dict['word'] = G.nodes['word'].data['h']
        print()
        funcs = {}
        print(G.canonical_etypes)
        G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype == 'ww':
                continue

            print('srctype, etype, dsttype',srctype, etype, dsttype)
            # Compute W_r * h
            # edge
            # print(feat_dict[srctype].shape,'srctype')
            # edge_weight = G.edges[etype].data['weight']
            # Wh = self.weight[etype](feat_dict[srctype])
            # print(edge_weight.type(),Wh.type(),feat_dict[srctype].type())

            # # Save it in graph for message passing
            # G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # # Specify per-relation message passing functions: (message_func, reduce_func).
            # # Note that the results are saved to the same destination feature 'h', which
            # # hints the type wise reducer for aggregation.
            # funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))


            Wh = self.weight[etype](G.nodes[srctype].data['h'])

            # Save it in graph for message passing
            G.nodes[srctype].data['h'] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.u_mul_e('h', 'weight', 'm'), fn.mean('m', 'h'))
            # funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        # print({ntype : G.nodes[ntype].data['h'] for ntype in ['word']})
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class HeteroLayerG(nn.Module):
    def __init__(self, in_size, out_size):
        super(HeteroLayerG, self).__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(in_size, out_size),
                'wt': nn.Linear(out_size, out_size),
                'wd': nn.Linear(out_size, out_size),
                'td': nn.Linear(out_size, out_size),
                'tt': nn.Linear(out_size, out_size),
            }) 

    def forward(self, G):
        # print(G,feat_dict,'G,feat_dict')
        funcs={}
        for srctype, etype, dsttype in [['word','ww','word']]: 
            Wh = self.weight[etype](G.nodes['word'].data['h'])
            G.nodes[srctype].data['h'] = Wh
            funcs[etype] = (fn.u_mul_e('h', 'weight', 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        funcs = {}
        # print(G.canonical_etypes)
        G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype == 'ww':
                continue
            # print('srctype, etype, dsttype',srctype, etype, dsttype) 
            Wh = self.weight[etype](G.nodes[srctype].data['h'])
            G.nodes[srctype].data['h'] = Wh 
            funcs[etype] = (fn.u_mul_e('h', 'weight', 'm'), fn.mean('m', 'h')) 
        G.multi_update_all(funcs, 'sum')
        return G
        # return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}


class WordGraphLayerG(nn.Module):
    def __init__(self, in_size, out_size):
        super(WordGraphLayerG, self).__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(in_size, out_size),
            }) 
    def forward(self, G):
        # print(G,feat_dict,'G,feat_dict')
        funcs={}
        for srctype, etype, dsttype in [['word','ww','word']]: 
            Wh = self.weight[etype](G.nodes['word'].data['h'])
            G.nodes[srctype].data['h'] = Wh
            funcs[etype] = (fn.u_mul_e('h', 'weight', 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        return G
        # return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class static_heto_graph(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        # norm = dglnn.EdgeWeightNorm(norm='both')
        # norm_edge_weight = norm(g, edge_weight)
        # self.hconv = dglnn.HeteroGraphConv({
        #                 'wt' : dglnn.GraphConv(h_dim,h_dim),
        #                 # 'ww' : HeteroLayer(h_inp,h_dim),
        #                 'ww' : dglnn.GraphConv(h_inp,h_dim),
        #                 'wd' : dglnn.GraphConv(h_dim,h_dim),
        #                 'td' : dglnn.GraphConv(h_dim,h_dim),
        #                 'tt' : dglnn.GraphConv(h_dim,h_dim)},
        #                 # 'dw' : dglnn.SAGEConv(10,10),
        #                 # 'dt' : dglnn.SAGEConv(10,10),
        #                 # 'tt' : dglnn.SAGEConv(10,10)},
        #                 aggregate='sum')
        # self.hconv = nn.ModuleList([HeteroLayer(h_inp, h_dim),HeteroLayer(h_dim, h_dim)])
        self.hconv1 = HeteroLayerG(h_inp, h_dim) 
        self.hconv2 = HeteroLayerG(h_dim, h_dim)
        # self.maxpooling  = nn.MaxPool1d(3)# 
        # self.maxpooling  = dglnn.MaxPooling()
        self.out_layer = nn.Linear(h_dim,1) 
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
        # unbatch???? get emb of lists
        # print(bg,'bg =====')
        # print(bg.canonical_etypes)
        # h1 = {'word' : torch.randn((bg.number_of_nodes('word'), 5)),'topic' : torch.randn((bg.number_of_nodes('topic'), 5))}
        # h1 = {'word' : torch.randn((bg.number_of_nodes('word'), 5))}
        # h1 = {'topic' : torch.randn((bg.number_of_nodes('topic'), 5))}
        # bg.nodes['word'].data['h'] 
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = torch.zeros((bg.number_of_nodes('doc'), self.h_dim)).to(self.device)
        bg.nodes['word'].data['h'] = word_emb
        bg.nodes['topic'].data['h'] = topic_emb
        bg.nodes['doc'].data['h'] = doc_emb
        # print(doc_emb,'================')
        # emb_dict = {
        #     'word':word_emb,
        #     'topic':topic_emb,
        #     'doc':doc_emb
        # }
        bg = self.hconv1(bg)
        # print(bg.nodes['doc'].data['h'],'+++++++++++++++++')
        bg = self.hconv2(bg)
        # bg = self.hconv(bg)
        # print('r',r.keys())
        # print(bg.nodes['doc'].data['h'])
        doc_emb = bg.nodes['doc'].data['h']
        # exit()
        # h1 = {'doc' : torch.randn((bg.number_of_nodes('doc'), 5)),
        #      'word' : torch.randn((bg.number_of_nodes('word'), 5)),
        #     'topic' : torch.randn((bg.number_of_nodes('topic'), 5))}
        # x_src = {'word' : word_emb}
        # x_dst = x_src
        # r = self.hconv(bg, (x_src, x_dst))
        # # print('r word',r.keys())
        # h_word_emb = r['word']
        # x_src = {'word' : h_word_emb,'word' : h_word_emb,'topic' : topic_emb}
        # x_dst = {'topic' : topic_emb,'doc' : doc_emb,'doc' : doc_emb}
        # # r['topic'] = topic_emb
        # # r['doc'] = doc_emb
        # r = self.hconv(bg, (x_src, x_dst))
        # # r = self.hconv(bg,emb_dict)
        # # print('rrrrr',r.keys())
        # doc_emb = r['doc']
        # print(doc_emb.shape,'doc_emb')
        doc_len = [g.num_nodes('doc') for g in g_list]
        doc_emb_split = torch.split(doc_emb, doc_len)
        # print(len(doc_emb_split),'doc_emb_split',doc_emb_split[0].shape)
        # padding to same size  
        # print(max(doc_len),'max(doc_len)')
        embed_pad_tensor = torch.zeros(len(doc_len), max(doc_len), self.h_dim).to(self.device)
        for i, embeds in enumerate(doc_emb_split): 
                embed_pad_tensor[i, torch.arange(0,len(embeds)), :] = embeds
        # print(embed_pad_tensor.shape,'embed_pad_tensor') # batch,max # doc, f 

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
        # print(doc_pool.shape,'doc_pool')
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


class static_graph(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None 
        self.hconv1 = WordGraphLayerG(h_inp, h_dim) 
        self.hconv2 = WordGraphLayerG(h_dim, h_dim)
        # self.maxpooling  = nn.MaxPool1d(3)# 
        # self.maxpooling  = dglnn.MaxPooling()
        self.out_layer = nn.Linear(h_dim,1) 
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
        # unbatch???? get emb of lists
        # print(bg,'bg =====')
        # print(bg.canonical_etypes)
        # h1 = {'word' : torch.randn((bg.number_of_nodes('word'), 5)),'topic' : torch.randn((bg.number_of_nodes('topic'), 5))}
        # h1 = {'word' : torch.randn((bg.number_of_nodes('word'), 5))}
        # h1 = {'topic' : torch.randn((bg.number_of_nodes('topic'), 5))}
        # bg.nodes['word'].data['h'] 
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        # topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        # doc_emb = torch.zeros((bg.number_of_nodes('doc'), self.h_dim)).to(self.device)
        bg.nodes['word'].data['h'] = word_emb
        bg = self.hconv1(bg)
        bg = self.hconv2(bg)
        word_emb = bg.nodes['word'].data['h']
        word_len = [g.num_nodes('word') for g in g_list]
        word_emb_split = torch.split(word_emb, word_len)
        # print(len(doc_emb_split),'doc_emb_split',doc_emb_split[0].shape)
        # padding to same size  
        embed_pad_tensor = torch.zeros(len(word_len), max(word_len), self.h_dim).to(self.device)
        for i, embeds in enumerate(word_emb_split): 
                embed_pad_tensor[i, torch.arange(0,len(embeds)), :] = embeds
        # print(embed_pad_tensor.shape,'embed_pad_tensor') # batch,max # doc, f 

        word_pool = embed_pad_tensor.mean(1)
        y_pred = self.out_layer(word_pool)
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        # out layer
        # batch graphs
        # graph propagation
        # pred, idx, _ = self.__get_pred_embeds(t_list)
        # loss = self.criterion(pred, true_prob_r[idx])
        return loss, y_pred
