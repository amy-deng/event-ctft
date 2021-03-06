from torch._C import device
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
from sparsemax import Sparsemax
# from tcn import *
try:
    import dgl
    import dgl.function as fn
    import dgl.nn.pytorch as dglnn
except:
    print("<<< dgl are not imported >>>")
    pass
 
 
class TempMessagePassingLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.5, use_norm = False):
        super(TempMessagePassingLayer, self).__init__()

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
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype]
            # print('etype =',etype)
            # print(edges.data['time'].shape,edges.data['time'])

            # curr_time = 6
            # edge_times = (edges.data['time']==curr_time).nonzero() # [n,1]
            # print(edge_times.shape,edge_times)

            # if for time=0, propagate
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
        return {'t': F.relu(h.view(-1, self.out_dim))}
        
    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            # if etype == 'ww':
            #     continue
            edge_dict.append(etype)
            # print(srctype, etype, dsttype)
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
            # if ntype == 'word':
            #     continue
            # print('ntype', ntype)
            alpha = torch.sigmoid(self.skip[ntype])
            trans_out = self.a_linears[ntype](G.nodes[ntype].data['t']) # TODO h? or ht
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            # recurrent unit
            # trans_out = 
            # trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)


    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


 
class TempMessagePassingLayer2(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.5, use_norm = False):
        super(TempMessagePassingLayer2, self).__init__()
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
        self.rnn = nn.RNNCell(out_dim, out_dim)
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
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype]
            # print('etype =',etype)
            # print(edges.data['time'].shape,edges.data['time'])

            # curr_time = 6
            # edge_times = (edges.data['time']==curr_time).nonzero() # [n,1]
            # print(edge_times.shape,edge_times)

            # if for time=0, propagate
            key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        # if 'timeh' in edges.data:
        #     # print(edges.data['v'].shape,edges.data['timeh'].shape,'==')
        #     edges.data['v'] += edges.data['timeh'].unsqueeze(1)
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': F.relu(h.view(-1, self.out_dim))}
        
    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            # if etype == 'ww':
            #     continue
            edge_dict.append(etype)
            # print(srctype, etype, dsttype)
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
            # if ntype == 'word':
            #     continue
            # print('ntype', ntype)
            # alpha = torch.sigmoid(self.skip[ntype])
            # trans_out = self.a_linears[ntype](G.nodes[ntype].data['t']) # TODO h? or ht
            # trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            attn_msg_t = G.nodes[ntype].data['t'] + G.time_emb
            # print(attn_msg_t.shape,'attn_msg_t',G.time_emb.shape,'G.time_emb',G.time_emb)
            hx = self.rnn(attn_msg_t, G.nodes[ntype].data['ht-1'])
            # recurrent unit
            # tanh(W*t+W*ht-1)
            trans_out = hx #+ G.nodes[ntype].data['h0']
            # trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)


    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)

# a little diff in forward
class TempMessagePassingLayer3(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.5, use_norm = False):
        super(TempMessagePassingLayer3, self).__init__()
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
        self.rnn = nn.RNNCell(out_dim, out_dim)
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
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype]
            # print('etype =',etype)
            # print(edges.data['time'].shape,edges.data['time'])

            # curr_time = 6
            # edge_times = (edges.data['time']==curr_time).nonzero() # [n,1]
            # print(edge_times.shape,edge_times)

            # if for time=0, propagate
            key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        # if 'timeh' in edges.data:
        #     # print(edges.data['v'].shape,edges.data['timeh'].shape,'==')
        #     edges.data['v'] += edges.data['timeh'].unsqueeze(1)
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': F.relu(h.view(-1, self.out_dim))}
        
    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            # if etype == 'ww':
            #     continue
            edge_dict.append(etype)
            # print(srctype, etype, dsttype)
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
            # if ntype == 'word':
            #     continue
            # print('ntype', ntype)
            alpha = torch.sigmoid(self.skip[ntype])
            trans_out = self.a_linears[ntype](G.nodes[ntype].data['t'] + G.time_emb) # TODO h? or ht
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            
            if ntype in ['word','topic']:
                trans_out = self.rnn(trans_out, G.nodes[ntype].data['ht-1'])

            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)

# very similar to 3
class SeqHGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.use_norm    = use_norm
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        # self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        # self.skip = nn.ParameterDict() 
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            # self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            # self.skip[t] = nn.Parameter(torch.ones(1))
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        self.relation_pri = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        # self.rnn = nn.RNNCell(out_dim, out_dim)
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
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype] 
            key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        # if 'timeh' in edges.data:
        #     # print(edges.data['v'].shape,edges.data['timeh'].shape,'==')
        #     edges.data['v'] += edges.data['timeh'].unsqueeze(1)
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}
        # return {'t': F.relu(h.view(-1, self.out_dim))}

    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            # if etype == 'ww':
            #     continue
            edge_dict.append(etype)
            # print(srctype, etype, dsttype)
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
            # alpha = torch.sigmoid(self.skip[ntype])
            # trans_out = self.a_linears[ntype](G.nodes[ntype].data.pop('t') + G.time_emb) # TODO h? or ht
            # trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            trans_out = G.nodes[ntype].data.pop('t') #+ G.time_emb
            trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_relations={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
# HGT with causal node
class SeqCauHGTLayer_w_rdm(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False, device=torch.device("cpu")):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.device        = device
        self.use_norm    = use_norm
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        # self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        # self.skip = nn.ParameterDict() 
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            # self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            # self.skip[t] = nn.Parameter(torch.ones(1))
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        self.relation_pri = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))

        self.relation_pri_cau = nn.ParameterDict()
        self.relation_att_cau = nn.ParameterDict()
        self.relation_msg_cau = nn.ParameterDict()
        for etype in ['tw','tt','td']:
            self.relation_pri_cau[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att_cau[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg_cau[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        '''    
        self.cau_filter = nn.ParameterDict()
        for cau_type in ['pos','neg','rdm']:
            self.cau_filter[cau_type] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        '''
        self.cau_filter = nn.Parameter(torch.Tensor(3, n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        # self.rnn = nn.RNNCell(out_dim, out_dim)
        self.sparsemax = Sparsemax(dim=1)
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
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype] 
            # print(relation_msg.shape,'relation_msg')
            key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            cau_att = None
            cau_val = None
            if etype in ['tw','tt','td']:
                # if src is causal node
                # causal edges
                # src node 
                cau_types = edges.src['cau_type'] # 0,1,2,3  learn and mask out 0 type
                # print('cau_types',cau_types.shape)
                relation_att_cau = self.relation_att_cau[etype]
                relation_pri_cau = self.relation_pri_cau[etype]
                relation_msg_cau = self.relation_msg_cau[etype]
                # print(relation_att_cau.shape,'relation_att_cau')
                # print(relation_pri_cau.shape,'relation_pri_cau')
                # print(relation_msg_cau.shape,'relation_msg_cau')
                # src_k = edges.src['k'].transpose(1,0)
                # print(edges.src['k'].transpose(1,0).shape,'==')
                cau_key = torch.bmm(edges.src['k'].transpose(1,0), relation_att_cau).transpose(1,0)
                # print(cau_key.shape,'cau_key') # [339, 4, 8]
                
                '''# mask or filter
                pos_effect = self.cau_filter['pos']
                neg_effect = self.cau_filter['neg']
                rdm_effect = torch.zeros(neg_effect.size()).to(self.device) # TODO
                # print(pos_effect.shape,neg_effect.shape,'neg_effect',rdm_effect.shape)
                effect = torch.stack((rdm_effect,pos_effect,neg_effect),dim=0)
                '''
                effect = self.cau_filter
                # print(effect.shape,'====')
                effect_mask = effect[cau_types]
                # print(effect_mask.shape,'effect_mask') # [339, 4, 8, 8]
                # print(effect_mask.nonzero(as_tuple=False).sum())
                # exit()
                n, n_head, d_k, _ = effect_mask.size()
                mul1 = cau_key.reshape(-1,1,d_k)
                mul2 = effect_mask.reshape(-1,d_k,d_k)
                # print(mul1.shape,mul2.shape,'========')
                masked_effect = torch.bmm(mul1,mul2)
                masked_effect = masked_effect.reshape(n,n_head,-1)
                # print(masked_effect.shape,'masked_effect',masked_effect)
                # r = r.squeeze(2)
                # print(r.shape,'r222')

                cau_att   = (edges.dst['q'] * masked_effect).sum(dim=-1) * relation_pri_cau / self.sqrt_dk
                cau_val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg_cau).transpose(1,0)
                # print(cau_key.shape,'cau_key',cau_att.shape,'cau_att',cau_val.shape,'cau_val')
                return {'a': att, 'v': val, 'ca':cau_att,'cv':cau_val}
            # return {'a': att, 'v': val, 'ca':cau_att,'cv':cau_val}
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        # if 'timeh' in edges.data:
        #     # print(edges.data['v'].shape,edges.data['timeh'].shape,'==')
        #     edges.data['v'] += edges.data['timeh'].unsqueeze(1)
        if 'ca' in edges.data:
            # print('=======')
            # print(edges.data['ca'])
            return {'v': edges.data['v'], 'a': edges.data['a'], 'ca':edges.data.pop('ca'),'cv':edges.data.pop('cv')}
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):

        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        # print(h.shape,'~~~~~~~~~~')
        if 'ca' in nodes.mailbox:
            cau_att = F.softmax(nodes.mailbox['ca'], dim=1) # spasemax TODO
            # cau_att = softmax_custom(nodes.mailbox['ca'], dim=1)
            cau_h   = torch.sum(cau_att.unsqueeze(dim = -1) * nodes.mailbox['cv'], dim=1)
            # print(cau_h.shape,'======')
            h += cau_h

        return {'t': h.view(-1, self.out_dim)}
        # return {'t': F.relu(h.view(-1, self.out_dim))}

    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            # if etype == 'ww':
            #     continue
            edge_dict.append(etype)
            # print(srctype, etype, dsttype)
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
            # alpha = torch.sigmoid(self.skip[ntype])
            # trans_out = self.a_linears[ntype](G.nodes[ntype].data.pop('t') + G.time_emb) # TODO h? or ht
            # trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            trans_out = G.nodes[ntype].data.pop('t') #+ G.time_emb
            trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

# with time
class SeqCauHGTLayer_w_rdm2(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False, device=torch.device("cpu")):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.device        = device
        self.use_norm    = use_norm
        self.time_emb    = None
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        # self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        # self.skip = nn.ParameterDict() 
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            # self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            # self.skip[t] = nn.Parameter(torch.ones(1))
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        self.relation_pri = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))

        self.relation_pri_cau = nn.ParameterDict()
        self.relation_att_cau = nn.ParameterDict()
        self.relation_msg_cau = nn.ParameterDict()
        for etype in ['tw','tt','td']:
            self.relation_pri_cau[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att_cau[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg_cau[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        '''    
        self.cau_filter = nn.ParameterDict()
        for cau_type in ['pos','neg','rdm']:
            self.cau_filter[cau_type] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        '''
        self.cau_filter = nn.Parameter(torch.Tensor(3, n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        # self.rnn = nn.RNNCell(out_dim, out_dim)
        self.sparsemax = Sparsemax(dim=1)
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
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype] 
            key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            cau_att = None
            cau_val = None
            if etype in ['tw','tt','td']:
                # if src is causal node
                # causal edges
                # src node 
                cau_types = edges.src['cau_type'] # 0,1,2,3  learn and mask out 0 type
                relation_att_cau = self.relation_att_cau[etype]
                relation_pri_cau = self.relation_pri_cau[etype]
                relation_msg_cau = self.relation_msg_cau[etype] 
                cau_key = torch.bmm(edges.src['k'].transpose(1,0), relation_att_cau).transpose(1,0)
                effect = self.cau_filter
                effect_mask = effect[cau_types]
                n, n_head, d_k, _ = effect_mask.size()
                mul1 = cau_key.reshape(-1,1,d_k)
                mul2 = effect_mask.reshape(-1,d_k,d_k)
                # print(mul1.shape,mul2.shape,'========')
                masked_effect = torch.bmm(mul1,mul2)
                # print(masked_effect.shape,'masked_effect')
                masked_effect = masked_effect.reshape(n,n_head,d_k) 
                cau_att   = (edges.dst['q'] * masked_effect).sum(dim=-1) * relation_pri_cau / self.sqrt_dk
                cau_val   = torch.bmm(edges.src['v'].transpose(1,0) + self.time_emb, relation_msg_cau).transpose(1,0)
                # print(cau_key.shape,'cau_key',cau_att.shape,'cau_att',cau_val.shape,'cau_val')
                return {'a': att, 'v': val, 'ca':cau_att,'cv':cau_val}
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        # if 'timeh' in edges.data:
        #     # print(edges.data['v'].shape,edges.data['timeh'].shape,'==')
        #     edges.data['v'] += edges.data['timeh'].unsqueeze(1)
        if 'ca' in edges.data:
            # print('=======')
            # print(edges.data['ca'])
            return {'v': edges.data['v'], 'a': edges.data['a'], 'ca':edges.data.pop('ca'),'cv':edges.data.pop('cv')}
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):

        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        if 'ca' in nodes.mailbox:
            cau_att = F.softmax(nodes.mailbox['ca'], dim=1) # spasemax TODO
            cau_h   = torch.sum(cau_att.unsqueeze(dim = -1) * nodes.mailbox['cv'], dim=1)
            h += cau_h

        return {'t': h.view(-1, self.out_dim)}

    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        self.time_emb = G.time_emb
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype not in self.etypes:
                continue 
            edge_dict.append(etype)
            # print(srctype, etype, dsttype)
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
            # alpha = torch.sigmoid(self.skip[ntype])
            # trans_out = self.a_linears[ntype](G.nodes[ntype].data.pop('t') + G.time_emb) # TODO h? or ht
            # trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            trans_out = G.nodes[ntype].data.pop('t') #+ G.time_emb
            trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

 
class SeqCauHGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False, device=torch.device("cpu")):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.device        = device
        self.use_norm    = use_norm
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        # self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        # self.skip = nn.ParameterDict() 
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            # self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            # self.skip[t] = nn.Parameter(torch.ones(1))
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        self.relation_pri = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))

        self.relation_pri_cau = nn.ParameterDict()
        self.relation_att_cau = nn.ParameterDict()
        self.relation_msg_cau = nn.ParameterDict()
        for etype in ['tw','tt','td']:
            self.relation_pri_cau[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att_cau[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg_cau[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.cau_filter = nn.ParameterDict()
        for cau_type in ['pos','neg']:
            self.cau_filter[cau_type] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        self.sparsemax = Sparsemax(dim=1)
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
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype] 
            # print(relation_msg.shape,'relation_msg')
            key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            cau_att = None
            cau_val = None
            if etype in ['tw','tt','td']:
                # if src is causal node
                # causal edges
                # src node 
                cau_types = edges.src['cau_type'] # 0,1,2,3  learn and mask out 0 type
                # print('cau_types',cau_types.shape)
                relation_att_cau = self.relation_att_cau[etype]
                relation_pri_cau = self.relation_pri_cau[etype]
                relation_msg_cau = self.relation_msg_cau[etype]
                # print(relation_att_cau.shape,'relation_att_cau')
                # print(relation_pri_cau.shape,'relation_pri_cau')
                # print(relation_msg_cau.shape,'relation_msg_cau')
                # src_k = edges.src['k'].transpose(1,0)
                # print(edges.src['k'].transpose(1,0).shape,'==')
                cau_key = torch.bmm(edges.src['k'].transpose(1,0), relation_att_cau).transpose(1,0)
                # print(cau_key.shape,'cau_key') # [339, 4, 8]
                
                pos_effect = self.cau_filter['pos']
                neg_effect = self.cau_filter['neg']
                rdm_effect = torch.zeros(neg_effect.size()).to(self.device) # TODO
                # print(pos_effect.shape,neg_effect.shape,'neg_effect',rdm_effect.shape)
                effect = torch.stack((rdm_effect,pos_effect,neg_effect),dim=0)
                # print(effect.shape,'====')
                effect_mask = effect[cau_types]
                # print(effect_mask.shape,'effect_mask') # [339, 4, 8, 8]
                # print(effect_mask.nonzero(as_tuple=False).sum())
                # exit()
                n, n_head, d_k, _ = effect_mask.size()
                mul1 = cau_key.reshape(-1,1,d_k)
                mul2 = effect_mask.reshape(-1,d_k,d_k)
                # print(mul1.shape,mul2.shape,'========')
                masked_effect = torch.bmm(mul1,mul2)
                masked_effect = masked_effect.reshape(n,n_head,-1)
                # print(masked_effect.shape,'masked_effect',masked_effect)
                # r = r.squeeze(2)
                # print(r.shape,'r222')

                cau_att   = (edges.dst['q'] * masked_effect).sum(dim=-1) * relation_pri_cau / self.sqrt_dk
                cau_val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg_cau).transpose(1,0)
                # print(cau_key.shape,'cau_key',cau_att.shape,'cau_att',cau_val.shape,'cau_val')
                return {'a': att, 'v': val, 'ca':cau_att,'cv':cau_val}
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        if 'ca' in edges.data:
            # print('=======')
            # print(edges.data['ca'])
            return {'v': edges.data['v'], 'a': edges.data['a'], 'ca':edges.data.pop('ca'),'cv':edges.data.pop('cv')}
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):

        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        if 'ca' in nodes.mailbox:
            cau_att = softmax_custom(nodes.mailbox['ca'], dim=1)
            cau_h   = torch.sum(cau_att.unsqueeze(dim = -1) * nodes.mailbox['cv'], dim=1)
            h += cau_h

        return {'t': h.view(-1, self.out_dim)}

    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            # if etype == 'ww':
            #     continue
            edge_dict.append(etype)
            # print(srctype, etype, dsttype)
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
            # alpha = torch.sigmoid(self.skip[ntype])
            # trans_out = self.a_linears[ntype](G.nodes[ntype].data.pop('t') + G.time_emb) # TODO h? or ht
            # trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            trans_out = G.nodes[ntype].data.pop('t') #+ G.time_emb
            trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

 
class SeqCauHGTLayer2(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False, device=torch.device("cpu")):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.device        = device
        self.use_norm    = use_norm
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        # self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        # self.skip = nn.ParameterDict() 
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            # self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            # self.skip[t] = nn.Parameter(torch.ones(1))
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        self.relation_pri = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))

        self.relation_pri_cau = nn.ParameterDict()
        self.relation_att_cau = nn.ParameterDict()
        self.relation_msg_cau = nn.ParameterDict()
        for etype in ['tw','tt','td']:
            self.relation_pri_cau[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att_cau[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg_cau[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.cau_filter = nn.ParameterDict()
        for cau_type in ['pos','neg']:
            self.cau_filter[cau_type] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        self.sparsemax = Sparsemax(dim=1)
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
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype] 
            # print(relation_msg.shape,'relation_msg')
            key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            cau_att = None
            cau_val = None
            if etype in ['tw','tt','td']:
                cau_types = edges.src['cau_type'] # 0,1,2,3  learn and mask out 0 type
                relation_att_cau = self.relation_att_cau[etype]
                relation_pri_cau = self.relation_pri_cau[etype]
                relation_msg_cau = self.relation_msg_cau[etype]
                cau_key = torch.bmm(edges.src['k'].transpose(1,0), relation_att_cau).transpose(1,0)
                
                pos_effect = self.cau_filter['pos']
                neg_effect = self.cau_filter['neg']
                rdm_effect = torch.zeros(neg_effect.size()).to(self.device) # TODO
                effect = torch.stack((rdm_effect,pos_effect,neg_effect),dim=0)
                effect_mask = effect[cau_types]
                n, n_head, d_k, _ = effect_mask.size()
                mul1 = cau_key.reshape(-1,1,d_k)
                mul2 = effect_mask.reshape(-1,d_k,d_k)
                masked_effect = torch.bmm(mul1,mul2)
                masked_effect = masked_effect.reshape(n,n_head,-1)
                cau_att   = (edges.dst['q'] * masked_effect).sum(dim=-1) * relation_pri_cau / self.sqrt_dk
                cau_val   = torch.bmm(edges.src['v'].transpose(1,0) + self.time_emb, relation_msg_cau).transpose(1,0)
                return {'a': att, 'v': val, 'ca':cau_att,'cv':cau_val}
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        if 'ca' in edges.data:
            # print('=======')
            # print(edges.data['ca'])
            return {'v': edges.data['v'], 'a': edges.data['a'], 'ca':edges.data.pop('ca'),'cv':edges.data.pop('cv')}
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):

        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        if 'ca' in nodes.mailbox:
            cau_att = softmax_custom(nodes.mailbox['ca'], dim=1)
            cau_h   = torch.sum(cau_att.unsqueeze(dim = -1) * nodes.mailbox['cv'], dim=1)
            h += cau_h

        return {'t': h.view(-1, self.out_dim)}

    def forward(self, G, inp_key, out_key):
        self.time_emb = G.time_emb
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype not in self.etypes:
                continue 
            edge_dict.append(etype)
            # print(srctype, etype, dsttype)
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
            # alpha = torch.sigmoid(self.skip[ntype])
            # trans_out = self.a_linears[ntype](G.nodes[ntype].data.pop('t') + G.time_emb) # TODO h? or ht
            # trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            trans_out = G.nodes[ntype].data.pop('t') #+ G.time_emb
            trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)



def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(axis=0)

# def softmax_custom(x,dim=0):
#     bi_x = torch.where(x > 0, 1, 0)
#     return torch.exp(x)* bi_x / (torch.exp(x)*bi_x).sum(axis=dim)

# ignore zero values
def softmax_custom(x,dim=0):
    bi_x = torch.where(x > 0, 1, 0)
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)
    x_exp_sum = torch.sum(x_exp*bi_x, dim, keepdim=True)
    output_custom = x_exp*bi_x / (x_exp_sum+1e-6)
    return output_custom

class SeqHGTLayerFlex(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.use_norm    = use_norm
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        # self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        # self.skip = nn.ParameterDict() 
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            # self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            # self.skip[t] = nn.Parameter(torch.ones(1))
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        self.relation_pri = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        # self.rnn = nn.RNNCell(out_dim, out_dim)
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
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype] 
            key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        # if 'timeh' in edges.data:
        #     # print(edges.data['v'].shape,edges.data['timeh'].shape,'==')
        #     edges.data['v'] += edges.data['timeh'].unsqueeze(1)
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}
        # return {'t': F.relu(h.view(-1, self.out_dim))}

    def forward(self, G, inp_key, out_key, etypes=None,ntypes=None):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        edge_dict = []
        # print(G.canonical_etypes,'====')
        # print(G.ntypes,'====G.ntypes=====')
        if etypes is not None:
            canonical_etypes = [v for v in G.canonical_etypes if v[1] in etypes]
        else:
            canonical_etypes = G.canonical_etypes
        if ntypes is not None:
            ntypes = [v for v in G.ntypes if v in ntypes]
        else:
            ntypes = G.ntypes
        for srctype, etype, dsttype in canonical_etypes:
            edge_dict.append(etype)
            # print(srctype, etype, dsttype)
            k_linear = self.k_linears[srctype]
            v_linear = self.v_linears[srctype] 
            q_linear = self.q_linears[dsttype]
            
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            
            G.apply_edges(func=self.edge_attention(etype), etype=etype)
           
        G.multi_update_all({etype : (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer = 'mean')
        

        # edge_dict = []
        # for srctype, etype, dsttype in G.canonical_etypes:
        #     if etype not in ['tw','td','tt']:
        #         continue
        #     edge_dict.append(etype)
        #     # print(srctype, etype, dsttype)
        #     k_linear = self.k_linears[srctype]
        #     v_linear = self.v_linears[srctype] 
        #     q_linear = self.q_linears[dsttype]
            
        #     G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
        #     G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
        #     G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            
        #     G.apply_edges(func=self.edge_attention(etype), etype=etype)
           
        # G.multi_update_all({etype : (self.message_func, self.reduce_func) \
        #                     for etype in edge_dict}, cross_reducer = 'mean')

        for ntype in ntypes:
            # alpha = torch.sigmoid(self.skip[ntype])
            # trans_out = self.a_linears[ntype](G.nodes[ntype].data.pop('t') + G.time_emb) # TODO h? or ht
            # trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            trans_out = G.nodes[ntype].data.pop('t') #+ G.time_emb
            trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_relations={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)

 

class GlobalAttentionPooling(nn.Module):
    r"""Apply Global Attention Pooling (`Gated Graph Sequence Neural Networks
    <https://arxiv.org/abs/1511.05493.pdf>`__) over the nodes in the graph.

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i}\mathrm{softmax}\left(f_{gate}
        \left(x^{(i)}_k\right)\right) f_{feat}\left(x^{(i)}_k\right)

    Parameters
    ----------
    gate_nn : gluon.nn.Block
        A neural network that computes attention scores for each feature.
    feat_nn : gluon.nn.Block, optional
        A neural network applied to each feature before combining them
        with attention scores.
    """
    def __init__(self, h_inp, h_hid):
        super(GlobalAttentionPooling, self).__init__()
        # with self.name_scope():
            # self.gate_nn = nn.Linear(h_inp,1)
            # self.feat_nn = nn.Linear(h_inp, h_inp)
            # todo for each meta edge
        self.feat_nns = nn.ModuleDict({
            'word':nn.Linear(h_inp, h_hid),
            'topic':nn.Linear(h_inp, h_hid),
            'doc':nn.Linear(h_inp, h_hid),
        })
        self.gate_nns = nn.ModuleDict({
            'word':nn.Linear(h_inp, 1),
            'topic':nn.Linear(h_inp, 1),
            'doc':nn.Linear(h_inp, 1),
        })

    def forward(self, graph, inp_key_dict):
        r"""Compute global attention pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input feature with shape :math:`(N, D)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        mxnet.NDArray
            The output feature with shape :math:`(B, D)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            # gate = self.gate_nn(feat)
            # assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
            # feat = self.feat_nn(feat) if self.feat_nn else feat
            readout = {}

            for ntype in inp_key_dict:
                inp_key = inp_key_dict[ntype]
                feat = graph.nodes[ntype].data[inp_key]
                # print('pool - ',feat.shape)
                gate = self.gate_nns[ntype](feat) # feat[ntype] 
                feat = self.feat_nns[ntype](feat)
                # print('gate - ',gate.shape,feat.shape)
                graph.nodes[ntype].data['gate'] = gate
                gate = dgl.softmax_nodes(graph, 'gate',ntype=ntype)
                graph.nodes[ntype].data['r'] = feat * gate
                readout[ntype] = dgl.sum_nodes(graph, 'r', ntype=ntype)
                # h
            # graph.ndata['gate'] = gate
            # gate = dgl.softmax_nodes(graph, 'gate')
            # graph.ndata['r'] = feat * gate
            # readout = dgl.sum_nodes(graph, 'r')
            return readout


class GlobalAttentionPooling2(nn.Module):
    def __init__(self, h_inp, h_hid):
        super().__init__()
        self.gate_nns = nn.ModuleDict({
            'word':nn.Linear(h_inp, 1),
            'topic':nn.Linear(h_inp, 1),
            'doc':nn.Linear(h_inp, 1),
        })

    def forward(self, graph, inp_key_dict):
        with graph.local_scope():
            readout = {}

            for ntype in inp_key_dict:
                inp_key = inp_key_dict[ntype]
                feat = graph.nodes[ntype].data[inp_key]
                gate = self.gate_nns[ntype](feat) # feat[ntype] 
                # print('gate - ',gate.shape,feat.shape)
                graph.nodes[ntype].data['gate'] = gate
                gate = dgl.softmax_nodes(graph, 'gate',ntype=ntype)
                graph.nodes[ntype].data['r'] = feat * gate
                readout[ntype] = dgl.sum_nodes(graph, 'r', ntype=ntype)
                # h
            # graph.ndata['gate'] = gate
            # gate = dgl.softmax_nodes(graph, 'gate')
            # graph.ndata['r'] = feat * gate
            # readout = dgl.sum_nodes(graph, 'r')
            return readout


# learn embedding by time order
class tempMP(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(tempMP, self).__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        self.time_emb = RelTemporalEncoding(n_hid, seq_len)
        self.attn_pool = GlobalAttentionPooling(n_hid*2, n_hid)
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        node_dict = {'doc': 0, 'topic': 1, 'word': 2}
        edge_dict = {'td': 0, 'tt': 1, 'wd': 2, 'wt': 3, 'ww': 4}
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(TempMessagePassingLayer(n_hid, n_hid, len(node_dict), len(edge_dict), n_heads, use_norm = use_norm))
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
        # print(len(g_list),'g_list ')
        bg = dgl.batch(g_list).to(self.device)  
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        bg.nodes['word'].data['h'] = self.adapt_ws(word_emb)
        bg.nodes['topic'].data['h'] = topic_emb
        bg.nodes['doc'].data['h'] = doc_emb 
        for ntype in ['word','topic','doc']:
            allone = torch.zeros(bg.num_nodes(ntype)).long().to(self.device)
            bg.nodes[ntype].data['timeh'] = self.time_emb(allone)
 
        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        for curr_time in range(self.seq_len):
            # print('curr_time',curr_time)
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, {('word', 'ww', 'word'): ww_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        }, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            for i in range(self.n_layers):
                self.gcs[i](sub_bg, 'h', 'h')
            for ntype in ['word','topic','doc']:
                bg.nodes[ntype].data['h'][orig_node_ids[ntype]] = sub_bg.nodes[ntype].data['h']
                bg.nodes[ntype].data['timeh'][orig_node_ids[ntype]] = time_emb
        for ntype in ['word','topic','doc']:
            bg.nodes[ntype].data['h'] = torch.cat((bg.nodes[ntype].data['h'],bg.nodes[ntype].data['timeh']),dim=-1)
        
        attn_pool_out = self.attn_pool(bg, 'h')
        global_info = []
        for ntype in attn_pool_out.keys():
            global_info.append(attn_pool_out[ntype])
        global_info = torch.cat(global_info,dim=-1)
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

 
'''
# no global attn pool
inp for each time is h0+t
use rnn 
'''
class tempMP3(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(tempMP3, self).__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        self.time_emb = RelTemporalEncoding(n_hid, seq_len)
        # self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        node_dict = {'doc': 0, 'topic': 1, 'word': 2}
        edge_dict = {'td': 0, 'tt': 1, 'wd': 2, 'wt': 3, 'ww': 4}
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(TempMessagePassingLayer2(n_hid, n_hid, len(node_dict), len(edge_dict), n_heads, use_norm = use_norm))
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
        # print(len(g_list),'g_list ')
        bg = dgl.batch(g_list).to(self.device)  
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        bg.nodes['word'].data['h0'] = self.adapt_ws(word_emb)
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb

        # bg.edges['ww'].data['timeh'] = self.time_emb(bg.edges['ww'].data['time'].long())
        # # print(bg.edges['ww'].data['timeh'].shape,'timeh',bg.edges['ww'].data['time'])
        # bg.edges['wd'].data['timeh'] = self.time_emb(bg.edges['wd'].data['time'].long())
        # bg.edges['wt'].data['timeh'] = self.time_emb(bg.edges['wt'].data['time'].long())
        # bg.edges['td'].data['timeh'] = self.time_emb(bg.edges['td'].data['time'].long())
         
        for ntype in ['word','topic','doc']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
 
        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        for curr_time in range(self.seq_len):
            # print('curr_time',curr_time)
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            # print(ww_edges_idx,'ww_edges_idx')
            sub_bg = dgl.edge_subgraph(bg_cpu, {('word', 'ww', 'word'): ww_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        }, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # for ntype in ['word','topic','doc']:
            #     # sub_bg.nodes[ntype].data['ht-1'] = sub_bg.nodes[ntype].data['ht']
            #     sub_bg.nodes[ntype].data['ht'] = sub_bg.nodes[ntype].data['h0'] + time_emb # add time into input, not good
            # print(sub_bg,'sub_bg')
            # print(orig_node_ids,'orig_node_ids',type(orig_node_ids))
            # graph conv
            sub_bg.time_emb = time_emb
            for i in range(self.n_layers):
                self.gcs[i](sub_bg, 'h0', 'ht-1')
            # update h to bg
            for ntype in ['word','topic','doc']:
                # print('==',ntype,sub_bg.nodes[ntype].data['h'].shape,bg.nodes[ntype].data['h'].shape)
                bg.nodes[ntype].data['ht-1'][orig_node_ids[ntype]] = sub_bg.nodes[ntype].data['ht-1']
                # bg.nodes[ntype].data['timeh'][orig_node_ids[ntype]] = time_emb
        
        # attn_pool_out = self.attn_pool(bg, 'ht-1')
        # # print(attn_pool_out.keys(),attn_pool_out)
        # global_info = []
        # for ntype in attn_pool_out.keys():
        #     # print(attn_pool_out[ntype].shape,ntype)
        #     global_info.append(attn_pool_out[ntype])
        # global_info = torch.cat(global_info,dim=-1)
        # print(global_info.shape,'global_info')
        #  
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='ht-1',ntype='doc')
            global_word_info = dgl.max_nodes(bg, feat='ht-1',ntype='word')
            global_topic_info = dgl.max_nodes(bg, feat='ht-1',ntype='topic')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='ht-1',ntype='doc')
            global_word_info = dgl.mean_nodes(bg, feat='ht-1',ntype='word')
            global_topic_info = dgl.mean_nodes(bg, feat='ht-1',ntype='topic')
        global_info = torch.cat((global_doc_info, global_word_info, global_topic_info),-1)
        # print(global_info.shape,'global_info')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred


class tempMP4(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(tempMP4, self).__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        self.time_emb = RelTemporalEncoding(n_hid, seq_len)
        # self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        node_dict = {'doc': 0, 'topic': 1, 'word': 2}
        edge_dict = {'td': 0, 'tt': 1, 'wd': 2, 'wt': 3, 'ww': 4}
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(TempMessagePassingLayer3(n_hid, n_hid, len(node_dict), len(edge_dict), n_heads, use_norm = use_norm))
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
        # print(len(g_list),'g_list ')
        bg = dgl.batch(g_list).to(self.device)  
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        bg.nodes['word'].data['h0'] = self.adapt_ws(word_emb)
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb
        # bg.edges['ww'].data['timeh'] = self.time_emb(bg.edges['ww'].data['time'].long())
        # # print(bg.edges['ww'].data['timeh'].shape,'timeh',bg.edges['ww'].data['time'])
        # bg.edges['wd'].data['timeh'] = self.time_emb(bg.edges['wd'].data['time'].long())
        # bg.edges['wt'].data['timeh'] = self.time_emb(bg.edges['wt'].data['time'].long())
        # bg.edges['td'].data['timeh'] = self.time_emb(bg.edges['td'].data['time'].long())
        # word and topic take info from last time step
        for ntype in ['word','topic','doc']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
 
        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        for curr_time in range(self.seq_len):
            # print('curr_time',curr_time)
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, {('word', 'ww', 'word'): ww_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        }, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            # sub_bg = dgl.remove_self_loop(sub_bg, etype='tt')
            # sub_bg = dgl.remove_self_loop(sub_bg, etype='ww')
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # for ntype in ['word','topic','doc']:
            #     # sub_bg.nodes[ntype].data['ht-1'] = sub_bg.nodes[ntype].data['ht']
            #     sub_bg.nodes[ntype].data['ht'] = sub_bg.nodes[ntype].data['h0'] + time_emb # add time into input, not good
            # print(sub_bg,'sub_bg')
            # print(orig_node_ids,'orig_node_ids',type(orig_node_ids))
            # graph conv
            sub_bg.time_emb = time_emb
            for i in range(self.n_layers):
                self.gcs[i](sub_bg, 'h0', 'ht-1')
            # update h to bg
            for ntype in ['word','topic','doc']:
                # print('==',ntype,sub_bg.nodes[ntype].data['h'].shape,bg.nodes[ntype].data['h'].shape)
                # print('ntype',ntype)
                bg.nodes[ntype].data['ht-1'][orig_node_ids[ntype]] = sub_bg.nodes[ntype].data['ht-1']
                # bg.nodes[ntype].data['timeh'][orig_node_ids[ntype]] = time_emb
        
        # attn_pool_out = self.attn_pool(bg, 'ht-1')
        # # print(attn_pool_out.keys(),attn_pool_out)
        # global_info = []
        # for ntype in attn_pool_out.keys():
        #     # print(attn_pool_out[ntype].shape,ntype)
        #     global_info.append(attn_pool_out[ntype])
        # global_info = torch.cat(global_info,dim=-1)
        # print(global_info.shape,'global_info')
        #  
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='ht-1',ntype='doc')
            global_word_info = dgl.max_nodes(bg, feat='ht-1',ntype='word')
            global_topic_info = dgl.max_nodes(bg, feat='ht-1',ntype='topic')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='ht-1',ntype='doc')
            global_word_info = dgl.mean_nodes(bg, feat='ht-1',ntype='word')
            global_topic_info = dgl.mean_nodes(bg, feat='ht-1',ntype='topic')
        global_info = torch.cat((global_doc_info, global_word_info, global_topic_info),-1)
        # print(global_info.shape,'global_info')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

# no causal main
class Temp11(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None

        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        # self.time_emb = RelTemporalEncoding(n_hid, seq_len)
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        # self.rnn = nn.RNNCell(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        })
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(SeqHGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
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
        # print(len(g_list),'g_list ')
        bg = dgl.batch(g_list).to(self.device)  
        # init_topic_emb = torch.mm(self.topic_weights,self.topic_gen_embeds)
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        for curr_time in range(self.seq_len):
            # print('curr_time',curr_time)
            # time1 = time.time()
            # time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # time2 = time.time()
            # print('find idx',time2-time1) 
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # time3 = time.time()
            # print('get subgraph',time3-time2)
            # sub_bg.time_emb = time_emb
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            # time4 = time.time()
            # print('graph conv info',time4-time3)
            for ntype in ['word','topic']:
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnn(sub_bg.nodes[ntype].data['ht'], sub_bg.nodes[ntype].data['ht-1'])
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            # time5 = time.time()
            # print('temporal info',time5-time4)
            # update h to bg
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            # time6 = time.time()
            # print('copy back to bg',time6-time5)

        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        # print(global_info.shape,'global_info')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

# time causal
class Temp2(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None

        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        self.cau_embeds = nn.Parameter(torch.Tensor(3,n_hid))
        self.cau_weight = nn.Parameter(torch.Tensor(seq_len,num_topic,3)) # TODO
        # self.time_emb = RelTemporalEncoding(n_hid, seq_len)
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        # self.rnn = nn.RNNCell(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        })
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(SeqHGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
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
        # print(len(g_list),'g_list ')
        bg = dgl.batch(g_list).to(self.device)  
        # init_topic_emb = torch.mm(self.topic_weights,self.topic_gen_embeds)
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        # topic_ids = bg.nodes['topic'].data['id'].long()
        # effect = bg.nodes['topic'].data['effect'].to_dense()
        # effect = (effect >0)*1. + (effect < 0)*(-1.)
        # print('effect',effect.shape)
        # print(bg.nodes['topic'].data['effect'].to_dense().shape,'======','topic_ids',topic_ids.shape)
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        # tt_edges_idx = [True for i in range(len(bg.edges(etype='tt')))]
        for curr_time in range(self.seq_len):
            # print('curr_time',curr_time)
            # time1 = time.time()
            # time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # time2 = time.time()
            # print('find idx',time2-time1) 
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # time3 = time.time()
            # print('get subgraph',time3-time2)
            # sub_bg.time_emb = time_emb
            topic_ids = sub_bg.nodes['topic'].data['id'].long()
            effect = sub_bg.nodes['topic'].data['effect']#.to_dense()
            effect = (effect >0)*1. + (effect < 0)*(-1.)
            causal_w = self.cau_weight[curr_time][topic_ids]
            # effect = sub_bg.nodes['topic'].data['effect'].to_dense()
            # print('causal_w',causal_w.shape,'cau_weight',self.cau_weight.shape,'topic_ids',topic_ids.shape)
            t = (effect * causal_w) @ self.cau_embeds 
            # print('t',t.shape)

            sub_bg.nodes['topic'].data['h0'] += t
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            # time4 = time.time()
            # print('graph conv info',time4-time3)
            for ntype in ['word','topic']:
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnn(sub_bg.nodes[ntype].data['ht'], sub_bg.nodes[ntype].data['ht-1'])
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            # time5 = time.time()
            # print('temporal info',time5-time4)
            # update h to bg
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            # time6 = time.time()
            # print('copy back to bg',time6-time5)

        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        # print(global_info.shape,'global_info')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred


# compare with Temp2 add a linear layer of the weighted causal embedding
# worse than Temp2
# 12/1 [-1,0,1] to original effect --> bad

# first get context; update topic emb; update doc and graph
class Temp3(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        self.cau_embeds = nn.Parameter(torch.Tensor(3,n_hid))
        # self.cau_time_weight = nn.Parameter(torch.Tensor(seq_len)) #TODO
        self.cau_weight = nn.Parameter(torch.Tensor(num_topic,3)) # TODO
        # self.time_emb = RelTemporalEncoding(n_hid, seq_len)
        self.cau_linear = nn.Linear(n_hid,n_hid)
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        # self.rnn = nn.RNNCell(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        })
        # self.causal_score_linear = nn.Linear(n_hid,n_hid)
        # self.rnns = nn.ModuleDict({
        #     'word': nn.RNNCell(n_hid, n_hid),
        #     'topic': nn.RNNCell(n_hid, n_hid)}
        # )
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(SeqHGTLayerFlex(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
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
        # print(len(g_list),'g_list ')
        bg = dgl.batch(g_list).to(self.device)  
        # init_topic_emb = torch.mm(self.topic_weights,self.topic_gen_embeds)
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        # topic_ids = bg.nodes['topic'].data['id'].long()
        # effect = bg.nodes['topic'].data['effect'].to_dense()
        # effect = (effect >0)*1. + (effect < 0)*(-1.)
        # print('effect',effect.shape)
        # print(bg.nodes['topic'].data['effect'].to_dense().shape,'======','topic_ids',topic_ids.shape)
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        # tt_edges_idx = [True for i in range(len(bg.edges(etype='tt')))]
        for curr_time in range(self.seq_len):
            # print('curr_time',curr_time)
            # time1 = time.time()
            # time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # time2 = time.time()
            # print('find idx',time2-time1) 
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # time3 = time.time()
            # print('get subgraph',time3-time2)
            # sub_bg.time_emb = time_emb
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            # time4 = time.time()
            # print('graph conv info',time4-time3)
            for ntype in ['word','topic']:
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnn(sub_bg.nodes[ntype].data['ht'], sub_bg.nodes[ntype].data['ht-1'])
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnns[ntype](sub_bg.nodes[ntype].data['ht'],sub_bg.nodes[ntype].data['ht-1'])
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            # time5 = time.time()
            # print('temporal info',time5-time4)
            # update h to bg
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            # time6 = time.time()
            # print('copy back to bg',time6-time5)
        learned_topic_emb = bg.nodes['topic'].data['ht-1'] # emb of topics with attention
        # print('learned_topic_emb++++++++++',learned_topic_emb.shape)
        topic_ids = bg.nodes['topic'].data['id'].long()
        effect = bg.nodes['topic'].data['effect']#.to_dense()
        effect = (effect >0)*1. + (effect < 0)*(-1.)
        causal_w = self.cau_weight[topic_ids]
        # effect = sub_bg.nodes['topic'].data['effect'].to_dense()
        # print('causal_w',causal_w.shape,'cau_weight',self.cau_weight.shape,'topic_ids',topic_ids.shape)
        causal_emb = (effect * causal_w) @ self.cau_embeds 
        # print('causal_emb=============',causal_emb.shape)
        score = torch.sum(learned_topic_emb * causal_emb, dim=-1).unsqueeze(-1)
        # print(score.shape,'vvvvvscorevv')
        bg.nodes['topic'].data['ht-1'] = score * causal_emb + (1-score) * learned_topic_emb
        # print(bg.nodes['topic'].data['ht-1'])
        # print(bg.edges(etype='ww'))
        # for etype in ['ww','wt']:
        #     s,t = bg.edges(etype=etype)
        #     print(etype,s,t,s.shape)
        bg.nodes['doc'].data['ht-1'] = bg.nodes['doc'].data['ht'] 
        # print(bg.nodes['doc'].data['ht-1'].shape)
        # print(bg.nodes['topic'].data['ht-1'].shape)
        # print(bg.nodes['word'].data['ht-1'].shape)
        for i in range(self.n_layers):
            if i == 0:
                self.gcs[i](bg, 'ht-1', 'ht',etypes=['tw','td','wd','dw','tt'],ntypes=['word','doc','topic'])
            else:
                self.gcs[i](bg, 'ht', 'ht',etypes=['tw','td','wd','dw','tt'],ntypes=['word','doc','topic'])

        # update causal emb
        out_key_dict = {'doc':'ht','topic':'ht','word':'ht'}
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        # print(global_info.shape,'global_info')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

# at t, use dot product to get the attention score and update causal emb
class Temp4(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        self.cau_embeds = nn.Parameter(torch.Tensor(3,n_hid))
        # self.cau_time_weight = nn.Parameter(torch.Tensor(seq_len)) #TODO
        self.cau_weight = nn.Parameter(torch.Tensor(num_topic,3)) # TODO
        # self.time_emb = RelTemporalEncoding(n_hid, seq_len)
        # self.cau_linear = nn.Linear(n_hid,n_hid)
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        # self.rnn = nn.RNNCell(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        })
        # self.causal_score_linear = nn.Linear(n_hid,n_hid)
        # self.rnns = nn.ModuleDict({
        #     'word': nn.RNNCell(n_hid, n_hid),
        #     'topic': nn.RNNCell(n_hid, n_hid)}
        # )
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(SeqHGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
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
        # print(len(g_list),'g_list ')
        bg = dgl.batch(g_list).to(self.device)  
        # init_topic_emb = torch.mm(self.topic_weights,self.topic_gen_embeds)
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb

        topic_ids = bg.nodes['topic'].data['id'].long()
        effect = bg.nodes['topic'].data['effect']#.to_dense()
        effect = (effect >0)*1. + (effect < 0)*(-1.)
        # effect = (effect != 0)*1
        # print('effect',effect.shape)
        causal_w = torch.sigmoid(self.cau_weight[topic_ids])
        cau_embeds = (effect * causal_w) @ self.cau_embeds 
        # print(bg.nodes['topic'].data['effect'].to_dense().shape,'======','topic_ids',topic_ids.shape)
        bg.nodes['topic'].data['c'] = cau_embeds
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        # tt_edges_idx = [True for i in range(len(bg.edges(etype='tt')))]
        for curr_time in range(self.seq_len):
            # print('curr_time',curr_time)
            # time1 = time.time()
            # time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # time2 = time.time()
            # print('find idx',time2-time1) 
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # time3 = time.time()
            # print('get subgraph',time3-time2)
            # sub_bg.time_emb = time_emb
            score = torch.sum(sub_bg.nodes['topic'].data['ht-1'] * sub_bg.nodes['topic'].data['c'], dim=-1).unsqueeze(-1)
            # print(score.shape,'vvvvvscorevv')
            sub_bg.nodes['topic'].data['h0'] = score * sub_bg.nodes['topic'].data['c'] + (1-score) * sub_bg.nodes['topic'].data['h0']
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            # time4 = time.time()
            # print('graph conv info',time4-time3)
            for ntype in ['word','topic']:
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnn(sub_bg.nodes[ntype].data['ht'], sub_bg.nodes[ntype].data['ht-1'])
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnns[ntype](sub_bg.nodes[ntype].data['ht'],sub_bg.nodes[ntype].data['ht-1'])
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            # time5 = time.time()
            # print('temporal info',time5-time4)
            # update h to bg
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        # update causal emb
        # out_key_dict = {'doc':'ht','topic':'ht','word':'ht'}
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        # print(global_info.shape,'global_info')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

# use additive attention not dot product
class Temp41(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        self.cau_embeds = nn.Parameter(torch.Tensor(3,n_hid))
        # self.cau_time_weight = nn.Parameter(torch.Tensor(seq_len)) #TODO
        self.cau_weight = nn.Parameter(torch.Tensor(num_topic,3)) # TODO
        # self.time_emb = RelTemporalEncoding(n_hid, seq_len)
        # self.cau_linear = nn.Linear(n_hid,n_hid)
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        # self.rnn = nn.RNNCell(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        })
        # self.causal_score_linear = nn.Linear(n_hid,n_hid)
        # self.rnns = nn.ModuleDict({
        #     'word': nn.RNNCell(n_hid, n_hid),
        #     'topic': nn.RNNCell(n_hid, n_hid)}
        # )
        self.add_attn = AddAttention(n_hid,n_hid, dropout)
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(SeqHGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
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
        # print(len(g_list),'g_list ')
        bg = dgl.batch(g_list).to(self.device)  
        # init_topic_emb = torch.mm(self.topic_weights,self.topic_gen_embeds)
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb

        topic_ids = bg.nodes['topic'].data['id'].long()
        effect = bg.nodes['topic'].data['effect']#.to_dense()
        effect = (effect >0)*1. + (effect < 0)*(-1.)
        # print('effect',effect.shape)
        causal_w = self.cau_weight[topic_ids]
        cau_embeds = (effect * causal_w) @ self.cau_embeds 
        # print(bg.nodes['topic'].data['effect'].to_dense().shape,'======','topic_ids',topic_ids.shape)
        bg.nodes['topic'].data['c'] = cau_embeds
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        # tt_edges_idx = [True for i in range(len(bg.edges(etype='tt')))]
        for curr_time in range(self.seq_len):
            # print('curr_time',curr_time)
            # time1 = time.time()
            # time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # time2 = time.time()
            # print('find idx',time2-time1) 
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # time3 = time.time()
            # print('get subgraph',time3-time2)
            # sub_bg.time_emb = time_emb
            attn_out = self.add_attn(sub_bg.nodes['topic'].data['c'],sub_bg.nodes['topic'].data['ht-1'])
            sub_bg.nodes['topic'].data['h0'] = attn_out
            # score = torch.sum(sub_bg.nodes['topic'].data['h0'] * sub_bg.nodes['topic'].data['c'], dim=-1).unsqueeze(-1)
            # print(score.shape,'vvvvvscorevv')
            # print(attn_out.shape,'attn_out')
            # sub_bg.nodes['topic'].data['h0'] = score * sub_bg.nodes['topic'].data['c'] + (1-score) * sub_bg.nodes['topic'].data['h0']
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            # time4 = time.time()
            # print('graph conv info',time4-time3)
            for ntype in ['word','topic']:
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnn(sub_bg.nodes[ntype].data['ht'], sub_bg.nodes[ntype].data['ht-1'])
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnns[ntype](sub_bg.nodes[ntype].data['ht'],sub_bg.nodes[ntype].data['ht-1'])
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            # time5 = time.time()
            # print('temporal info',time5-time4)
            # update h to bg
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        # update causal emb
        # out_key_dict = {'doc':'ht','topic':'ht','word':'ht'}
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        # print(global_info.shape,'global_info')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

 
# pos causal and neg causal emb
class Temp5(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))

        self.cau_embeds_pos = nn.Parameter(torch.zeros(3,n_hid))
        self.cau_embeds_neg = nn.Parameter(torch.zeros(3,n_hid))
        # self.cau_time_weight = nn.Parameter(torch.Tensor(seq_len)) #TODO
        self.cau_weight_pos = nn.Parameter(torch.Tensor(num_topic,3)) # TODO
        self.cau_weight_neg = nn.Parameter(torch.Tensor(num_topic,3)) # TODO

        # self.time_emb = RelTemporalEncoding(n_hid, seq_len)
        self.cau_w1 = nn.Parameter(torch.Tensor(n_hid,10))
        self.cau_w2 = nn.Parameter(torch.Tensor(10,n_hid))
        # self.cau_linear = nn.Linear(n_hid,n_hid)
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        # self.rnn = nn.RNNCell(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        })
        # self.causal_score_linear = nn.Linear(n_hid,n_hid)
        # self.rnns = nn.ModuleDict({
        #     'word': nn.RNNCell(n_hid, n_hid),
        #     'topic': nn.RNNCell(n_hid, n_hid)}
        # )
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(SeqHGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
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
        # print(len(g_list),'g_list ')
        bg = dgl.batch(g_list).to(self.device)  
        # init_topic_emb = torch.mm(self.topic_weights,self.topic_gen_embeds)
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb

        topic_ids = bg.nodes['topic'].data['id'].long()
        effect = bg.nodes['topic'].data['effect']#.to_dense()
        effect = (effect >0)*1. + (effect < 0)*(-1.)
        pos_effect = (effect >0)*1.
        neg_effect = (effect <0)*1.
        # effect = (effect != 0)*1
        # print('effect',effect.shape )
        causal_w_pos = self.cau_weight_pos[topic_ids]
        causal_w_neg = self.cau_weight_neg[topic_ids]
        cau_embeds = (pos_effect * causal_w_pos) @ self.cau_embeds_pos + (neg_effect * causal_w_neg) @ self.cau_embeds_neg
        # print(self.cau_embeds_pos,'++++++')
        # print(self.cau_embeds_neg,'------')
        bg.nodes['topic'].data['c'] = cau_embeds
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        # tt_edges_idx = [True for i in range(len(bg.edges(etype='tt')))]
        for curr_time in range(self.seq_len):
            # print('curr_time',curr_time)
            # time1 = time.time()
            # time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # time2 = time.time()
            # print('find idx',time2-time1) 
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # time3 = time.time()
            # print('get subgraph',time3-time2)
            # sub_bg.time_emb = time_emb
            tmp = sub_bg.nodes['topic'].data['ht-1'] @ (self.cau_w1 @ self.cau_w2) 
            score = torch.sum(tmp * sub_bg.nodes['topic'].data['c'], dim=-1).unsqueeze(-1)
            # print(score.shape,'score')
            sub_bg.nodes['topic'].data['h0'] = score * sub_bg.nodes['topic'].data['c'] + (1-score) * sub_bg.nodes['topic'].data['h0']
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            # time4 = time.time()
            # print('graph conv info',time4-time3)
            for ntype in ['word','topic']:
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnn(sub_bg.nodes[ntype].data['ht'], sub_bg.nodes[ntype].data['ht-1'])
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnns[ntype](sub_bg.nodes[ntype].data['ht'],sub_bg.nodes[ntype].data['ht-1'])
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            # time5 = time.time()
            # print('temporal info',time5-time4)
            # update h to bg
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        # update causal emb
        # out_key_dict = {'doc':'ht','topic':'ht','word':'ht'}
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        # print(global_info.shape,'global_info')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        emb_dis = torch.norm(self.cau_embeds_pos - self.cau_embeds_neg)
        # print(emb_dis*0.05,emb_dis.shape,'emb_dis')
        # v = F.hinge_embedding_loss(emb_dis,torch.tensor(-1),margin=5.)
        # print(v,'v')
        loss = self.criterion(y_pred.view(-1), y_data) + 0.01*emb_dis
        # print(loss,'loss')
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

# pos causal and neg causal emb
# element-wise product
class Temp6(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True, agg='mul'):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.agg = agg
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))

        self.cau_embeds_pos = nn.Parameter(torch.Tensor(3,n_hid))
        self.cau_embeds_neg = nn.Parameter(torch.Tensor(3,n_hid))
        self.cau_embeds_rdm = nn.Parameter(torch.Tensor(3,n_hid))
        # self.cau_time_weight = nn.Parameter(torch.Tensor(seq_len)) #TODO
        # self.cau_weight_pos = nn.Parameter(torch.Tensor(num_topic,3)) # TODO
        # self.cau_weight_neg = nn.Parameter(torch.Tensor(num_topic,3)) # TODO

        # self.time_emb = RelTemporalEncoding(n_hid, seq_len)
        self.cau_w1 = nn.Parameter(torch.Tensor(n_hid,10))
        self.cau_w2 = nn.Parameter(torch.Tensor(10,n_hid))
        # self.cau_linear = nn.Linear(n_hid,n_hid)
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        # self.rnn = nn.RNNCell(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        })
        # self.causal_score_linear = nn.Linear(n_hid,n_hid)
        # self.rnns = nn.ModuleDict({
        #     'word': nn.RNNCell(n_hid, n_hid),
        #     'topic': nn.RNNCell(n_hid, n_hid)}
        # )
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(SeqHGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
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
        # print(len(g_list),'g_list ')
        bg = dgl.batch(g_list).to(self.device)  
        # init_topic_emb = torch.mm(self.topic_weights,self.topic_gen_embeds)
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb

        # topic_ids = bg.nodes['topic'].data['id'].long()
        effect = bg.nodes['topic'].data['effect']#.to_dense()
        # effect = (effect >0)*1. + (effect < 0)*(-1.)
        pos_effect = (effect >0)*1.
        neg_effect = (effect <0)*1.
        rdm_effect = (effect == 0)*1.
        # effect = (effect != 0)*1
        # print('effect',effect.shape )
        # print('pos_effect',pos_effect.shape )
        pos_effect = pos_effect.unsqueeze(-1)
        neg_effect = neg_effect.unsqueeze(-1)
        rdm_effect = rdm_effect.unsqueeze(-1)
        # print('pos_effect 2',pos_effect.shape )
        # t = pos_effect * self.cau_embeds_pos
        # print(t.shape,'tttt',t)
        # causal_w_pos = self.cau_weight_pos[topic_ids]
        # causal_w_neg = self.cau_weight_neg[topic_ids]
        cau_embeds = pos_effect * self.cau_embeds_pos + neg_effect * self.cau_embeds_neg + rdm_effect * self.cau_embeds_rdm
        # print(cau_embeds.shape,'+++=======+++',cau_embeds)
        
        # print(self.cau_embeds_neg,'------')
        if self.agg == 'mul':
            e1 = cau_embeds[:,0]
            e2 = cau_embeds[:,1]
            e3 = cau_embeds[:,2]
            bg.nodes['topic'].data['c'] = e1 * e2 * e3
        # elif self.agg == 'mean':
        #     bg.nodes['topic'].data['c'] = cau_embeds.mean(1)
        elif self.agg == 'max':
            # print(cau_embeds.shape,cau_embeds.max(-1))
            bg.nodes['topic'].data['c'] = cau_embeds.max(1)[0]
        elif self.agg == 'sum':
            bg.nodes['topic'].data['c'] = cau_embeds.sum(1)
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        # tt_edges_idx = [True for i in range(len(bg.edges(etype='tt')))]
        for curr_time in range(self.seq_len):
            # print('curr_time',curr_time)
            # time1 = time.time()
            # time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # time2 = time.time()
            # print('find idx',time2-time1) 
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # time3 = time.time()
            # print('get subgraph',time3-time2)
            # sub_bg.time_emb = time_emb
            tmp = sub_bg.nodes['topic'].data['ht-1'] @ (self.cau_w1 @ self.cau_w2) 
            score = torch.sum(tmp * sub_bg.nodes['topic'].data['c'], dim=-1).unsqueeze(-1)
            # print(score.shape,'score')
            sub_bg.nodes['topic'].data['h0'] = score * sub_bg.nodes['topic'].data['c'] + (1-score) * sub_bg.nodes['topic'].data['h0']
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            # time4 = time.time()
            # print('graph conv info',time4-time3)
            for ntype in ['word','topic']:
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnn(sub_bg.nodes[ntype].data['ht'], sub_bg.nodes[ntype].data['ht-1'])
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnns[ntype](sub_bg.nodes[ntype].data['ht'],sub_bg.nodes[ntype].data['ht-1'])
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            # time5 = time.time()
            # print('temporal info',time5-time4)
            # update h to bg
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        # update causal emb
        # out_key_dict = {'doc':'ht','topic':'ht','word':'ht'}
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        # print(global_info.shape,'global_info')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        # emb_dis_pos_neg = torch.norm(self.cau_embeds_pos - self.cau_embeds_neg)
        # print(emb_dis*0.05,emb_dis.shape,'emb_dis')
        # v = F.hinge_embedding_loss(emb_dis,torch.tensor(-1),margin=5.)
        # print(v,'v')
        loss = self.criterion(y_pred.view(-1), y_data)  
        # print(loss,'loss')
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

# pos causal and neg causal emb and rdm emb
# element-wise product, sum
class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_inp, n_hid, max_len = 7, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_inp, 2) *
                             -(math.log(10000.0) / n_inp))
        emb = nn.Embedding(max_len, n_inp)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_inp)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_inp)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_inp, n_hid)
    def forward(self, t):
        # print(self.emb(t),'self.emb(t)')
        return self.lin(self.emb(t))
        # return x + self.lin(self.emb(t))

class TemporalEncoding(nn.Module):
    def __init__(self, n_inp, max_len = 7, dropout = 0.2):
        super(TemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_inp, 2) *
                             -(math.log(10000.0) / n_inp))
        emb = nn.Embedding(max_len, n_inp)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_inp)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_inp)
        emb.requires_grad = False
        self.emb = emb
    def forward(self, t):
        return self.emb(t)

class Temp7(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True, agg='mul'):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.agg = agg
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))

        self.cau_embeds_pos = nn.Parameter(torch.Tensor(3,n_hid))
        self.cau_embeds_neg = nn.Parameter(torch.Tensor(3,n_hid))
        # self.cau_embeds_rdm = nn.Parameter(torch.Tensor(3,n_hid))
        # self.cau_time_weight = nn.Parameter(torch.Tensor(seq_len)) #TODO
        # self.cau_weight_pos = nn.Parameter(torch.Tensor(num_topic,3)) # TODO
        # self.cau_weight_neg = nn.Parameter(torch.Tensor(num_topic,3)) # TODO

        self.time_emb = RelTemporalEncoding(n_hid//2, n_hid, seq_len)
        self.cau_time_attn = ScaledDotProductAttention(n_hid)
        self.cau_w1 = nn.Parameter(torch.Tensor(n_hid,10))
        self.cau_w2 = nn.Parameter(torch.Tensor(10,n_hid))
        # self.cau_linear = nn.Linear(n_hid,n_hid)
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        # self.rnn = nn.RNNCell(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        })
        # self.causal_score_linear = nn.Linear(n_hid,n_hid)
        # self.rnns = nn.ModuleDict({
        #     'word': nn.RNNCell(n_hid, n_hid),
        #     'topic': nn.RNNCell(n_hid, n_hid)}
        # )
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(SeqHGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
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
        # print(len(g_list),'g_list ')
        bg = dgl.batch(g_list).to(self.device)  
        # init_topic_emb = torch.mm(self.topic_weights,self.topic_gen_embeds)
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        # topic_ids = bg.nodes['topic'].data['id'].long()
        effect = bg.nodes['topic'].data['effect']#.to_dense()
        # effect = (effect >0)*1. + (effect < 0)*(-1.)
        pos_effect = (effect >0)*1.
        neg_effect = (effect <0)*1. 
        pos_effect = pos_effect.unsqueeze(-1)
        neg_effect = neg_effect.unsqueeze(-1) 
        cau_embeds = pos_effect * self.cau_embeds_pos + neg_effect * self.cau_embeds_neg #+ rdm_effect * self.cau_embeds_rdm
        bg.nodes['topic'].data['c'] = cau_embeds 
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        # tt_edges_idx = [True for i in range(len(bg.edges(etype='tt')))]
        for curr_time in range(self.seq_len):
            # print('curr_time',curr_time)
            # time1 = time.time()
            
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # time2 = time.time()
            # print('find idx',time2-time1) 
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # time3 = time.time()
            # print('get subgraph',time3-time2)
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            # sub_bg.time_emb = time_emb
            # time_emb = time_emb.view(1,1,time_emb.size(0)).repeat(len(sub_bg.nodes['topic'].data['c']))
            # print(time_emb.shape,'time_emb',time_emb)

            # print(sub_bg.nodes['topic'].data['c'].shape)
            causal = self.cau_time_attn(time_emb.unsqueeze(1),sub_bg.nodes['topic'].data['c'])
            # print(causal,'causal',curr_time)
            tmp = sub_bg.nodes['topic'].data['ht-1'] @ (self.cau_w1 @ self.cau_w2) 
            score = torch.sum(tmp * causal, dim=-1).unsqueeze(-1)
            # print(score.shape,'score')
            sub_bg.nodes['topic'].data['h0'] = score * causal + (1-score) * sub_bg.nodes['topic'].data['h0']
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            # time4 = time.time()
            # print('graph conv info',time4-time3)
            for ntype in ['word','topic']:
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnn(sub_bg.nodes[ntype].data['ht'], sub_bg.nodes[ntype].data['ht-1'])
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnns[ntype](sub_bg.nodes[ntype].data['ht'],sub_bg.nodes[ntype].data['ht-1'])
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            # time5 = time.time()
            # print('temporal info',time5-time4)
            # update h to bg
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        # update causal emb
        # out_key_dict = {'doc':'ht','topic':'ht','word':'ht'}
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        # print(global_info.shape,'global_info')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        emb_dis_pos_neg = torch.norm(self.cau_embeds_pos - self.cau_embeds_neg)
        # print(emb_dis_pos_neg*0.01,emb_dis_pos_neg.shape,'emb_dis_pos_neg')
        # v = F.hinge_embedding_loss(emb_dis,torch.tensor(-1),margin=5.)
        # print(v,'v')
        loss = self.criterion(y_pred.view(-1), y_data) + emb_dis_pos_neg*0.01
        # print(loss,'loss')
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred


class Temp71(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True, agg='mul'):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.agg = agg
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))

        self.cau_embeds_pos = nn.Parameter(torch.Tensor(3,n_hid))
        self.cau_embeds_neg = nn.Parameter(torch.Tensor(3,n_hid))
        self.cau_embeds_rdm = nn.Parameter(torch.Tensor(3,n_hid))

        # self.cau_embeds_rdm = nn.Parameter(torch.Tensor(3,n_hid))
        # self.cau_time_weight = nn.Parameter(torch.Tensor(seq_len)) #TODO
        self.cau_weight = nn.Parameter(torch.Tensor(num_topic,3)) # TODO
        # self.time_emb = RelTemporalEncoding(n_hid//2, n_hid, seq_len)
        self.cau_time_attn = ScaledDotProductAttention(n_hid)
        self.cau_w1 = nn.Parameter(torch.Tensor(n_hid,10))
        self.cau_w2 = nn.Parameter(torch.Tensor(10,n_hid))
        # self.cau_linear = nn.Linear(n_hid,n_hid)
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        # self.rnn = nn.RNNCell(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        })
        # self.causal_score_linear = nn.Linear(n_hid,n_hid)
        # self.rnns = nn.ModuleDict({
        #     'word': nn.RNNCell(n_hid, n_hid),
        #     'topic': nn.RNNCell(n_hid, n_hid)}
        # )
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(SeqHGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
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
        # print(len(g_list),'g_list ')
        bg = dgl.batch(g_list).to(self.device)  
        # init_topic_emb = torch.mm(self.topic_weights,self.topic_gen_embeds)
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        topic_ids = bg.nodes['topic'].data['id'].long()
        effect = bg.nodes['topic'].data['effect']#.to_dense()
        # effect = (effect >0)*1. + (effect < 0)*(-1.)
        pos_effect = (effect >0)*1.
        neg_effect = (effect <0)*1. 
        rdm_effect = (effect ==0)*1. 
        pos_effect = pos_effect.unsqueeze(-1)
        neg_effect = neg_effect.unsqueeze(-1) 
        rdm_effect = rdm_effect.unsqueeze(-1) 
        cau_embeds = pos_effect * self.cau_embeds_pos + neg_effect * self.cau_embeds_neg + rdm_effect * self.cau_embeds_rdm
        # print('cau_embeds',cau_embeds.shape)
        causal_w = self.cau_weight[topic_ids]
        # print('causal_w',causal_w.shape)
        cau_embeds = torch.bmm(causal_w.unsqueeze(1),cau_embeds).squeeze(1)
        # print('cau_embeds',cau_embeds.shape)

        bg.nodes['topic'].data['c'] = cau_embeds 
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        # tt_edges_idx = [True for i in range(len(bg.edges(etype='tt')))]
        for curr_time in range(self.seq_len):
            # print('curr_time',curr_time)
            # time1 = time.time()
            
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # time2 = time.time()
            # print('find idx',time2-time1) 
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # time3 = time.time()
            # print('get subgraph',time3-time2)

            # print(sub_bg.nodes['topic'].data['c'].shape)
            causal = sub_bg.nodes['topic'].data['c']
            # print(causal,'causal')
            tmp = sub_bg.nodes['topic'].data['ht-1'] @ (self.cau_w1 @ self.cau_w2) 
            score = torch.sum(tmp * causal, dim=-1).unsqueeze(-1)
            score = torch.sigmoid(score)
            # print(score.shape,'score' ,score)
            sub_bg.nodes['topic'].data['h0'] = score * causal + (1-score) * sub_bg.nodes['topic'].data['h0']
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            # time4 = time.time()
            # print('graph conv info',time4-time3)
            for ntype in ['word','topic']:
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnn(sub_bg.nodes[ntype].data['ht'], sub_bg.nodes[ntype].data['ht-1'])
                # sub_bg.nodes[ntype].data['ht-1'] = self.rnns[ntype](sub_bg.nodes[ntype].data['ht'],sub_bg.nodes[ntype].data['ht-1'])
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            # time5 = time.time()
            # print('temporal info',time5-time4)
            # update h to bg
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        # update causal emb
        # out_key_dict = {'doc':'ht','topic':'ht','word':'ht'}
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        # print(global_info.shape,'global_info')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        emb_dis_pos_neg = torch.norm(self.cau_embeds_pos - self.cau_embeds_neg)
        # print(emb_dis_pos_neg*0.01,emb_dis_pos_neg.shape,'emb_dis_pos_neg')
        # v = F.hinge_embedding_loss(emb_dis,torch.tensor(-1),margin=5.)
        # print(v,'v')
        loss = self.criterion(y_pred.view(-1), y_data) + emb_dis_pos_neg*0.01
        # print(loss,'loss')
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

# not good
class Temp72(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True, agg='mul',eta=1e-3):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.agg = agg
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        self.eta = eta
        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))

        self.cau_embeds_pos = nn.Parameter(torch.Tensor(3,n_hid))
        self.cau_embeds_neg = nn.Parameter(torch.Tensor(3,n_hid))
        # self.cau_embeds_rdm = nn.Parameter(torch.Tensor(3,n_hid))
        self.time_emb = TemporalEncoding(n_hid, seq_len)
        self.cau_time_attn = ScaledDotProductAttention(n_hid)
        # self.time_w = nn.Parameter(torch.Tensor(seq_len,3))
        self.cau_w = nn.Parameter(torch.Tensor(n_hid,n_hid))
        # self.cau_w_value = nn.Parameter(torch.Tensor(n_hid,n_hid))

        # self.W1 = nn.Parameter(torch.Tensor(n_hid,n_hid))
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        }) 
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(SeqHGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
        self.out_layer = nn.Sequential(
                # nn.Linear(n_hid*3, n_hid),
                # nn.BatchNorm1d(n_hid),
                nn.Linear(n_hid*3, 1) 
        )
        self.threshold = 0.5
        # self.out_func = torch.sigmoid
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        # topic_ids = bg.nodes['topic'].data['id'].long()
        effect = bg.nodes['topic'].data['effect']#.to_dense()
        # effect = (effect >0)*1. + (effect < 0)*(-1.)
        # t1 = time.time()
        pos_effect = ((effect >0)*1.).unsqueeze(-1)
        neg_effect = ((effect <0)*1.).unsqueeze(-1)
        # rdm_effect = ((effect == 0)*1.).unsqueeze(-1) 
        cau_embeds = pos_effect * self.cau_embeds_pos + neg_effect * self.cau_embeds_neg #+ rdm_effect * self.cau_embeds_rdm
        # print(time.time()-t1,'====t1')
        bg.nodes['topic'].data['c'] = cau_embeds 
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        for curr_time in range(self.seq_len):
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            causal = self.cau_time_attn(time_emb.unsqueeze(1),sub_bg.nodes['topic'].data['c'])
            # print(causal,'========')
            tmp = self.dropout(sub_bg.nodes['topic'].data['ht-1']) @ self.cau_w 
            score = torch.sum(tmp * causal, dim=-1).unsqueeze(-1)
            # print(score,'score')
            sub_bg.nodes['topic'].data['h0'] = score * causal + (1-score) * sub_bg.nodes['topic'].data['h0']
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            for ntype in ['word','topic']:
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        y_pred = self.out_layer(global_info)
        emb_dis_pos_neg = torch.norm(self.cau_embeds_pos - self.cau_embeds_neg)
        # emb_dis_pos_rdm = torch.norm(self.cau_embeds_pos - self.cau_embeds_rdm)
        # emb_dis_rdm_neg = torch.norm(self.cau_embeds_rdm - self.cau_embeds_neg)
        # print(emb_dis_pos_neg,emb_dis_pos_rdm,emb_dis_rdm_neg)
        dis = emb_dis_pos_neg #+ emb_dis_pos_rdm + emb_dis_rdm_neg
        loss = self.criterion(y_pred.view(-1), y_data) + self.eta * dis
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred


class Temp8(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True, with_rdm=False):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.with_rdm = with_rdm
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))

        # self.cau_embeds_pos = nn.Parameter(torch.Tensor(3,n_hid))
        # self.cau_embeds_neg = nn.Parameter(torch.Tensor(3,n_hid))
        # self.cau_embeds_rdm = nn.Parameter(torch.Tensor(3,n_hid))
        # self.time_emb = TemporalEncoding(n_hid, seq_len)
        # self.cau_time_attn = ScaledDotProductAttention(n_hid)
        # self.time_w = nn.Parameter(torch.Tensor(seq_len,3))
        # self.cau_w = nn.Parameter(torch.Tensor(n_hid,n_hid))
        # self.cau_w_value = nn.Parameter(torch.Tensor(n_hid,n_hid))

        # self.W1 = nn.Parameter(torch.Tensor(n_hid,n_hid))
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        }) 
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            if self.with_rdm:
                self.gcs.append(SeqCauHGTLayer_w_rdm(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm, device=self.device))
            else:
                self.gcs.append(SeqCauHGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm, device=self.device))
        self.out_layer = nn.Sequential(
                # nn.Linear(n_hid*3, n_hid),
                # nn.BatchNorm1d(n_hid),
                nn.Linear(n_hid*3, 1) 
        )
        self.threshold = 0.5
        # self.out_func = torch.sigmoid
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        # topic_ids = bg.nodes['topic'].data['id'].long()
        effect = bg.nodes['topic'].data['effect']#.to_dense()
        effect = (effect >0)*1. + (effect < 0)*(-1.)
        effect = effect.sum(-1)
        effect = ((effect > 0)*1.) + ((effect < 0)*2.)

        # print(effect,'2')

        # exit()
        # t1 = time.time()
        # pos_effect = ((effect >0)*1.).unsqueeze(-1)
        # neg_effect = ((effect <0)*1.).unsqueeze(-1)
        # # rdm_effect = ((effect == 0)*1.).unsqueeze(-1) 
        # cau_embeds = pos_effect * self.cau_embeds_pos + neg_effect * self.cau_embeds_neg #+ rdm_effect * self.cau_embeds_rdm
        # print(time.time()-t1,'====t1')
        bg.nodes['topic'].data['cau_type'] = effect.long() 
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        for curr_time in range(self.seq_len):
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            """
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            causal = self.cau_time_attn(time_emb.unsqueeze(1),sub_bg.nodes['topic'].data['c'])
            # print(causal,'========')
            tmp = self.dropout(sub_bg.nodes['topic'].data['ht-1']) @ self.cau_w 
            score = torch.sum(tmp * causal, dim=-1).unsqueeze(-1)
            # print(score,'score')
            sub_bg.nodes['topic'].data['h0'] = score * causal + (1-score) * sub_bg.nodes['topic'].data['h0']
            """
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            for ntype in ['word','topic']:
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        y_pred = self.out_layer(global_info)
        loss = self.criterion(y_pred.view(-1), y_data) 
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred



class Temp81(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True, with_rdm=False):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.with_rdm = with_rdm
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        self.time_emb = TemporalEncoding(n_hid // n_heads, seq_len) 
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling2(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        }) 
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        # etypes = ['wd','td','tt','ww','tw','dw']

        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            if self.with_rdm:
                self.gcs.append(SeqCauHGTLayer_w_rdm2(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm, device=self.device))
            else:
                self.gcs.append(SeqCauHGTLayer2(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm, device=self.device))
        self.out_layer = nn.Sequential(
                # nn.Linear(n_hid*3, n_hid),
                # nn.BatchNorm1d(n_hid),
                nn.Linear(n_hid*3, 1) 
        )
        self.threshold = 0.5
        # self.out_func = torch.sigmoid
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        # topic_ids = bg.nodes['topic'].data['id'].long()
        effect = bg.nodes['topic'].data['effect']#.to_dense()
        effect = (effect >0)*1. + (effect < 0)*(-1.)
        effect = effect.sum(-1)
        effect = ((effect > 0)*1.) + ((effect < 0)*2.)

        # print(effect,'2')

        # exit()
        # t1 = time.time()
        # pos_effect = ((effect >0)*1.).unsqueeze(-1)
        # neg_effect = ((effect <0)*1.).unsqueeze(-1)
        # # rdm_effect = ((effect == 0)*1.).unsqueeze(-1) 
        # cau_embeds = pos_effect * self.cau_embeds_pos + neg_effect * self.cau_embeds_neg #+ rdm_effect * self.cau_embeds_rdm
        # print(time.time()-t1,'====t1')
        bg.nodes['topic'].data['cau_type'] = effect.long() 
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        for curr_time in range(self.seq_len):
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            sub_bg.time_emb = time_emb
            # print(time_emb)
            """
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            causal = self.cau_time_attn(time_emb.unsqueeze(1),sub_bg.nodes['topic'].data['c'])
            # print(causal,'========')
            tmp = self.dropout(sub_bg.nodes['topic'].data['ht-1']) @ self.cau_w 
            score = torch.sum(tmp * causal, dim=-1).unsqueeze(-1)
            # print(score,'score')
            sub_bg.nodes['topic'].data['h0'] = score * causal + (1-score) * sub_bg.nodes['topic'].data['h0']
            """
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            for ntype in ['word','topic']:
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        y_pred = self.out_layer(global_info)
        loss = self.criterion(y_pred.view(-1), y_data) 
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred


class Temp82(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True, with_rdm=False, causal=0):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.with_rdm = with_rdm
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        self.time_emb = TemporalEncoding(n_hid // n_heads, seq_len) 
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling2(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        }) 
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        # etypes = ['wd','td','tt','ww','tw','dw']

        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            if self.with_rdm:
                self.gcs.append(SeqCauHGTLayer_w_rdm2(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm, device=self.device))
            else:
                self.gcs.append(SeqCauHGTLayer2(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm, device=self.device))
        self.out_layer = nn.Sequential(
                # nn.Linear(n_hid*3, n_hid),
                # nn.BatchNorm1d(n_hid),
                nn.Linear(n_hid*3, 1) 
        )
        self.threshold = 0.5
        # self.out_func = torch.sigmoid
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        # topic_ids = bg.nodes['topic'].data['id'].long()
        effect = bg.nodes['topic'].data['effect']#.to_dense()
        effect = effect[:,self.causal]
        effect = (effect >0)*1. + (effect < 0)*(-1.)
        # effect = effect.sum(-1)
        effect = ((effect > 0)*1.) + ((effect < 0)*2.)
        bg.nodes['topic'].data['cau_type'] = effect.long() 
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        for curr_time in range(self.seq_len):
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dt', 'topic'): td_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            sub_bg.time_emb = time_emb
            # print(time_emb)
            """
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            causal = self.cau_time_attn(time_emb.unsqueeze(1),sub_bg.nodes['topic'].data['c'])
            # print(causal,'========')
            tmp = self.dropout(sub_bg.nodes['topic'].data['ht-1']) @ self.cau_w 
            score = torch.sum(tmp * causal, dim=-1).unsqueeze(-1)
            # print(score,'score')
            sub_bg.nodes['topic'].data['h0'] = score * causal + (1-score) * sub_bg.nodes['topic'].data['h0']
            """
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            for ntype in ['word','topic']:
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        y_pred = self.out_layer(global_info)
        loss = self.criterion(y_pred.view(-1), y_data) 
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred



class DiTemp81(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True, with_rdm=False):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.with_rdm = with_rdm
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # self.topic_gen_embeds = nn.Parameter(torch.Tensor(10, n_hid))
        # self.topic_weights = nn.Parameter(torch.Tensor(num_topic, 10))
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        self.time_emb = TemporalEncoding(n_hid // n_heads, seq_len) 
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling2(n_hid, n_hid)
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        }) 
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wd','td','tt','ww','tw','dw']

        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            if self.with_rdm:
                self.gcs.append(SeqCauHGTLayer_w_rdm2(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm, device=self.device))
            else:
                self.gcs.append(SeqCauHGTLayer2(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm, device=self.device))
        self.out_layer = nn.Sequential(
                # nn.Linear(n_hid*3, n_hid),
                # nn.BatchNorm1d(n_hid),
                nn.Linear(n_hid*3, 1) 
        )
        self.threshold = 0.5
        # self.out_func = torch.sigmoid
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        # topic_ids = bg.nodes['topic'].data['id'].long()
        effect = bg.nodes['topic'].data['effect'].to_dense()
        effect = (effect >0)*1. + (effect < 0)*(-1.)
        effect = effect.sum(-1)
        effect = ((effect > 0)*1.) + ((effect < 0)*2.)
        bg.nodes['topic'].data['cau_type'] = effect.long() 
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        for curr_time in range(self.seq_len):
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['tw'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        ('doc', 'dw', 'word'):wd_edges_idx
                                        }, # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            sub_bg.time_emb = time_emb
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            for ntype in ['word','topic']:
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, out_key_dict)
            global_info = []
            for ntype in attn_pool_out.keys():
                global_info.append(attn_pool_out[ntype])
            global_info = torch.cat(global_info,dim=-1)

        y_pred = self.out_layer(global_info)
        loss = self.criterion(y_pred.view(-1), y_data) 
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred



class AddAttention(nn.Module):
    def __init__(self, dim1, dim2, dropout):
        super().__init__()
        self.linear_in = nn.Linear(dim1 + dim2, (dim1 + dim2)//2, bias=True)
        self.v = nn.Parameter(torch.Tensor((dim1 + dim2)//2, 1))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.drop  = nn.Dropout(dropout)
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, inp1, inp2):
        inp = torch.cat((inp1,inp2),dim=-1)
        # print(inp.shape,inp1.shape,inp2.shape)
        inp = self.drop(inp)
        h = torch.tanh(self.linear_in(inp))
        # print(h.shape,'h',self.v.shape,'v')
        attention_weights = torch.mm(h,self.v)
        # print(attention_weights,'attention_weights')
        output = attention_weights * inp1 + inp2
        return output
       

class ScaledDotProductAttention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions):
        super().__init__()

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=True)

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, context_len, dimensions = context.size()
        attention_scores = (context @ query) / (dimensions**0.5)
        attention_weights = torch.softmax(attention_scores,dim=1) 
        r = torch.bmm(context.transpose(1, 2).contiguous(),attention_weights)
        return r.squeeze(-1)
