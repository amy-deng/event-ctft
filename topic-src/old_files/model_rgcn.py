import  time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
# from dgl import DGLGraph

# class HeteroConvLayer(nn.Module):
#     def __init__(self, word_in_size, topic_in_size, out_size):
#         super().__init__()
#         self.weight = nn.ModuleDict({
#                 'ww': nn.Linear(word_in_size, out_size),
#                 'wt': nn.Linear(word_in_size, out_size),
#                 'wd': nn.Linear(word_in_size, out_size),
#                 'td': nn.Linear(topic_in_size, out_size),
#                 'tt': nn.Linear(topic_in_size, out_size),
#             }) 

#     def forward(self, G, feat_dict):
#         funcs={}
#         G.edges['wd'].data['weight'] = G.edges['wd'].data['weight'].float()
#         for srctype, etype, dsttype in G.canonical_etypes:
#             # print('srctype, etype, dsttype',srctype, etype, dsttype,feat_dict[srctype].shape) 
#             Wh = self.weight[etype](feat_dict[srctype])
#             G.nodes[srctype].data['Wh_%s' % etype] = Wh
#             # print(etype,G.edges[etype].data['weight'].dtype,Wh.dtype)
#             funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

#         G.multi_update_all(funcs, 'sum')
#         return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
 

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout, etypes, ntypes):
        super().__init__()
        self.etypes = etypes
        self.ntypes = ntypes
        self.weight = nn.ModuleDict()
        for etype in etypes:
            self.weight[etype] = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.drop = nn.Dropout(dropout)

    def forward(self, G, feat_dict):

        for ntype in feat_dict:
            G.nodes[ntype].data['h'] = feat_dict[ntype]
        funcs={}
        G.edges['wd'].data['weight'] = G.edges['wd'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype not in self.etypes:
                continue
            Wh = self.weight[etype](G.nodes[srctype].data['h'])   #   feat_dict[srctype]
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')

        for ntype in self.ntypes:
            feat = G.nodes[ntype].data['h']
            G.nodes[ntype].data['h'] = self.drop(self.activation(feat))

        return {ntype : G.nodes[ntype].data['h'] for ntype in self.ntypes}

class Hetero(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, activation, device, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(Hetero, self).__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.device = device
        self.pool = pool
        # self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1,n_hid))
        self.hetero_layers = nn.ModuleList()
        # self.hetero_layers.append(HeteroConvLayer(n_inp, n_hid, activation, dropout, ['ww','wt','tt','wd','td'],['word','topic','doc']))
        self.adapt_ws = nn.Linear(n_inp, n_hid)
        for _ in range(n_layers):
            self.hetero_layers.append(HeteroRGCNLayer(n_hid, n_hid, activation, dropout, ['ww','wt','tt','wd','td'],['word','topic','doc']))
        self.out_layer = nn.Linear(n_hid, 1)  
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
        
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        # doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        # torch.zeros((bg.number_of_nodes('doc'), self.n_hid)).to(self.device)
        # for i in range(len(self.gcn_topic_layers)):
        # print('word_emb',word_emb.shape,topic_emb.shape,'doc_emb',doc_emb.shape)
        word_emb = self.adapt_ws(word_emb)
        feat_dict = {'word':word_emb, 'topic':topic_emb}#, 'doc':doc_emb}
        for layer in self.hetero_layers:
            feat_dict = layer(bg, feat_dict) 
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

class RGCNAll(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, activation, device, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(RGCNAll, self).__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.device = device
        self.pool = pool
        # self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1,n_hid))
        self.hetero_layers = nn.ModuleList()
        # self.hetero_layers.append(HeteroConvLayer(n_inp, n_hid, activation, dropout, ['ww','wt','tt','wd','td'],['word','topic','doc']))
        self.adapt_ws = nn.Linear(n_inp, n_hid)
        for _ in range(n_layers):
            self.hetero_layers.append(HeteroRGCNLayer(n_hid, n_hid, activation, dropout, ['ww','wt','tt','wd','td'],['word','topic','doc']))
        self.out_layer = nn.Linear(n_hid*3, 1)  
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        # doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        # torch.zeros((bg.number_of_nodes('doc'), self.n_hid)).to(self.device)
        # for i in range(len(self.gcn_topic_layers)):
        # print('word_emb',word_emb.shape,topic_emb.shape,'doc_emb',doc_emb.shape)
        word_emb = self.adapt_ws(word_emb)
        feat_dict = {'word':word_emb, 'topic':topic_emb}#, 'doc':doc_emb}
        for layer in self.hetero_layers:
            feat_dict = layer(bg, feat_dict) 
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='h',ntype='doc')
            global_word_info = dgl.max_nodes(bg, feat='h',ntype='word')
            global_topic_info = dgl.max_nodes(bg, feat='h',ntype='topic')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='h',ntype='doc')
            global_word_info = dgl.mean_nodes(bg, feat='h',ntype='word')
            global_topic_info = dgl.mean_nodes(bg, feat='h',ntype='topic')
        # print(global_doc_info,'global_doc_info')
        global_info = torch.cat((global_doc_info, global_word_info, global_topic_info),-1)
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

 