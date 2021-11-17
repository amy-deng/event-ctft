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
 

class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h, ntype, etype):
        
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * g.nodes[ntype].data['norm'].unsqueeze(1)
        g.nodes[ntype].data['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'),etype=etype)
        h = g.nodes[ntype].data.pop('h')
        # normalization by square root of dst degree
        h = h * g.nodes[ntype].data['norm'].unsqueeze(1)
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        return h
    def __repr__(self):
        return '{}(in_dim={}, out_dim={})'.format(
            self.__class__.__name__, self.in_feats, self.out_feats)

 
class HGTLayerModified(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.5, use_norm = False):
        super(HGTLayerModified, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.use_norm    = use_norm
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        for t in ['word','topic','doc']:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        
        self.skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
                'doc': nn.Parameter(torch.ones(1)),
        })
        self.relation_pri = nn.ParameterDict({
                # 'ww': nn.Parameter(torch.ones(self.n_heads)),
                'wt': nn.Parameter(torch.ones(self.n_heads)),
                'wd': nn.Parameter(torch.ones(self.n_heads)),
                'tt': nn.Parameter(torch.ones(self.n_heads)),
                'td': nn.Parameter(torch.ones(self.n_heads))
        })
        self.relation_att = nn.ParameterDict({
                # 'ww': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'wt': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'wd': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'tt': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'td': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
        })
        self.relation_msg = nn.ParameterDict({
                # 'ww': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'wt': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'wd': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'tt': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'td': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
        })
        self.drop           = nn.Dropout(dropout)
         
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def edge_attention(self, etype):
        def msg_func(edges):
            # print(etype,'etype')
            # etype = edges.data['id'][0]
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype]
            key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}
        
    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype == 'ww':
                continue
            edge_dict.append(etype)
            # print(srctype, etype, dsttype)
            # k_linear = self.k_linears[node_dict[srctype]]
            # v_linear = self.v_linears[node_dict[srctype]] 
            # q_linear = self.q_linears[node_dict[dsttype]]

            k_linear = self.k_linears[srctype]
            v_linear = self.v_linears[srctype] 
            q_linear = self.q_linears[dsttype]
            
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            
            G.apply_edges(func=self.edge_attention(etype), etype=etype)
           
        G.multi_update_all({etype : (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer = 'mean')
        for ntype in G.ntypes:
            if ntype == 'word':
                continue
            alpha = torch.sigmoid(self.skip[ntype])
            trans_out = self.a_linears[ntype](G.nodes[ntype].data['t'])
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)


    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)

class TempHGTLayerModified(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.5, use_norm = False):
        super(TempHGTLayerModified, self).__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.use_norm    = use_norm
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        for t in ['word','topic','doc']:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        
        self.skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
                'doc': nn.Parameter(torch.ones(1)),
        })
        self.relation_pri = nn.ParameterDict({
                'ww': nn.Parameter(torch.ones(self.n_heads)),
                'wt': nn.Parameter(torch.ones(self.n_heads)),
                'wd': nn.Parameter(torch.ones(self.n_heads)),
                'tt': nn.Parameter(torch.ones(self.n_heads)),
                'td': nn.Parameter(torch.ones(self.n_heads))
        })
        self.relation_att = nn.ParameterDict({
                'ww': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'wt': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'wd': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'tt': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'td': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
        })
        self.relation_msg = nn.ParameterDict({
                'ww': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'wt': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'wd': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'tt': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
                'td': nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k)),
        })
        self.drop           = nn.Dropout(dropout)
         
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def edge_attention(self, etype):
        def msg_func(edges):
            # print(etype,'etype')
            # etype = edges.data['id'][0]
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype]
            key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        if 'timeh' in edges.data:
            # print(edges.data['v'].shape,edges.data['timeh'].shape,'==')
            edges.data['v'] += edges.data['timeh'].unsqueeze(1)
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        # print("nodes.mailbox['v']",nodes.mailbox['v'].shape)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}
        
    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            edge_dict.append(etype)
            
            k_linear = self.k_linears[srctype]
            v_linear = self.v_linears[srctype] 
            q_linear = self.q_linears[dsttype]
            
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            
            G.apply_edges(func=self.edge_attention(etype), etype=etype)
           
        G.multi_update_all({etype : (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer = 'mean')
        for ntype in G.ntypes:
            alpha = torch.sigmoid(self.skip[ntype])
            trans_out = self.a_linears[ntype](G.nodes[ntype].data['t'])
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)


    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


# https://github.com/acbull/pyHGT/blob/f7c4be620242d8c1ab3055f918d4c082f5060e07/pyHGT/conv.py#L283
class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 7, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
    def forward(self, t):
        # return x + self.lin(self.emb(t))
         return self.lin(self.emb(t))

 
class HeteroCau(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(HeteroCau, self).__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1,n_hid))
        self.cau_embeds = nn.Parameter(torch.Tensor(3,n_hid))
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)

        node_dict = {'doc': 0, 'topic': 1, 'word': 2}
        edge_dict = {'td': 0, 'tt': 1, 'wd': 2, 'wt': 3, 'ww': 4}
        self.gcn_word_layers = nn.ModuleList()
        self.gcn_word_layers.append(GCNLayer(n_inp, n_hid, activation, dropout))
        for _ in range(n_layers-1):
            self.gcn_word_layers.append(GCNLayer(n_hid, n_hid, activation, dropout))
        # self.topic_layers = nn.ModuleList()
        # for _ in range(n_layers-1):
        #     self.topic_layers.append(GCNLayer(n_hid, n_hid, activation, dropout))

        self.atten_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.atten_layers.append(HGTLayerModified(n_hid, n_hid, len(node_dict), len(edge_dict), n_heads, use_norm = use_norm))

        self.out_layer = nn.Sequential(
                # nn.Linear(n_hid*3, n_hid),
                # nn.BatchNorm1d(n_hid),
                nn.Linear(n_hid*3, 1) 
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        # torch.zeros((bg.number_of_nodes('doc'), self.n_hid)).to(self.device)
        bg.nodes['word'].data['h'] = word_emb
        # bg.nodes['word'].data['h'] = self.adapt_ws(word_emb)
        bg.nodes['topic'].data['h'] = topic_emb
        bg.nodes['doc'].data['h'] = doc_emb # todo add attention to this?
        # for i in range(self.n_layers):
        for i in range(len(self.gcn_word_layers)):
            word_emb = self.gcn_word_layers[i](bg, word_emb, 'word','ww')
            bg.nodes['word'].data['h'] = word_emb
            self.atten_layers[i](bg, 'h', 'h')

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


class TempHGTAll2(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(TempHGTAll2, self).__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1,n_hid))
        self.time_emb = RelTemporalEncoding(n_hid//n_heads,seq_len)
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        node_dict = {'doc': 0, 'topic': 1, 'word': 2}
        edge_dict = {'td': 0, 'tt': 1, 'wd': 2, 'wt': 3, 'ww': 4}
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(TempHGTLayerModified(n_hid, n_hid, len(node_dict), len(edge_dict), n_heads, use_norm = use_norm))
        self.out_layer = nn.Sequential(
                # nn.Linear(n_hid*3, n_hid),
                # nn.BatchNorm1d(n_hid),
                nn.Linear(n_hid*3, 1) 
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        # torch.zeros((bg.number_of_nodes('doc'), self.n_hid)).to(self.device)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h'] = word_emb
        bg.nodes['topic'].data['h'] = topic_emb
        bg.nodes['doc'].data['h'] = doc_emb
        # bg.edges['ww'].data['time'] = torch.zeros(bg.edges['ww'].data['weight'].shape).int()
        # print('time',bg.edges['ww'].data['time'].shape)
        bg.edges['ww'].data['timeh'] = self.time_emb(bg.edges['ww'].data['time'].long())
        # print(bg.edges['ww'].data['timeh'].shape,'timeh',bg.edges['ww'].data['time'])
        bg.edges['wd'].data['timeh'] = self.time_emb(bg.edges['wd'].data['time'].long())
        bg.edges['wt'].data['timeh'] = self.time_emb(bg.edges['wt'].data['time'].long())
        bg.edges['td'].data['timeh'] = self.time_emb(bg.edges['td'].data['time'].long())
        # print(bg.edges['td'].data['timeh'])
        for i in range(self.n_layers):
            self.gcs[i](bg, 'h', 'h')

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
