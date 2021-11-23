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

    def forward(self, g, inp_key, out_key, ntype, etype):
        h = g.nodes[ntype].data[inp_key]
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        # h = h * g.nodes[ntype].data['norm'].unsqueeze(1)
        g.nodes[ntype].data['_h'] = h
        g.update_all(
                        # fn.copy_src(src='_h', out='m'),
                        fn.u_mul_e('_h', 'weight', 'm'),
                          fn.sum(msg='m', out='_h'),etype=etype)
        h = g.nodes[ntype].data.pop('_h')
        # normalization by square root of dst degree
        # h = h * g.nodes[ntype].data['norm'].unsqueeze(1)
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        g.nodes[ntype].data[out_key] = h
        # add temp information...
        # return h
    def __repr__(self):
        return '{}(in_dim={}, out_dim={})'.format(
            self.__class__.__name__, self.in_feats, self.out_feats)
 
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
class TempMessagePassingLayer6(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.5, use_norm = False):
        super(TempMessagePassingLayer6, self).__init__()
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
        # self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        for t in ['word','topic','doc']:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            # self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        
        # self.skip = nn.ParameterDict({
        #         'word': nn.Parameter(torch.ones(1)),
        #         'topic': nn.Parameter(torch.ones(1)),
        #         'doc': nn.Parameter(torch.ones(1)),
        # })
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
            trans_out = G.nodes[ntype].data.pop('t') + G.time_emb
            trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class HGTLayerHetero(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False):
        super(HGTLayerHetero, self).__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.num_types     = len(ntypes)
        self.num_relations = len(etypes)
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.use_norm    = use_norm
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        
        self.skip = nn.ParameterDict() 
        # self.rnns = nn.ModuleDict()
        for ntype in ntypes:
            self.skip[ntype] = nn.Parameter(torch.ones(1))
            # if ntype == 'topic':
            #     self.rnns[ntype] = nn.RNNCell(out_dim, out_dim)
        self.relation_pri = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
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
        
        for ntype in self.ntypes:
            if ntype == 'word':
                continue
            # print('ntype', ntype)
            
            alpha = torch.sigmoid(self.skip[ntype])
            trans_out = self.a_linears[ntype](G.nodes[ntype].data['t'] + G.time_emb) # TODO h? or ht
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            # if ntype in ['topic']:
            #     trans_out = self.rnns[ntype](trans_out, G.nodes[ntype].data['ht-1'])
            # if ntype in ['word']:
            #     trans_out = self.rnn(G.nodes[ntype].data[inp_key], G.nodes[ntype].data['ht-1'])
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_relations={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class HGTLayerHeteroAsyn(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False):
        super(HGTLayerHeteroAsyn, self).__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.num_types     = len(ntypes)
        self.num_relations = len(etypes)
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.use_norm    = use_norm
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        
        self.skip = nn.ParameterDict() 
        self.rnns = nn.ModuleDict()
        for ntype in ntypes:
            self.skip[ntype] = nn.Parameter(torch.ones(1))
            if ntype == 'topic':
                self.rnns[ntype] = nn.RNNCell(out_dim, out_dim)
        self.relation_pri = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
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
        # return {'t': h.view(-1, self.out_dim)}

    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        edge_dict = []
        # print(G.canonical_etypes)
        for srctype, etype, dsttype in [('topic', 'tt', 'topic'), ('word', 'wt', 'topic')]:
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
        
        # msg at last time
        ntype = 'topic'
        alpha = torch.sigmoid(self.skip[ntype])
        trans_out = self.a_linears[ntype](G.nodes[ntype].data['t'])# + G.time_emb) # TODO h? or ht
        trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
        trans_out = self.rnns[ntype](trans_out, G.nodes[ntype].data['ht-1'])
        if self.use_norm:
            G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
        else:
            G.nodes[ntype].data[out_key] = self.drop(trans_out)
        G.nodes[ntype].data[inp_key] = G.nodes[ntype].data[out_key]

        edge_dict = []
        for srctype, etype, dsttype in [('topic', 'td', 'doc'), ('word', 'wd', 'doc')]:
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
        
        ntype = 'doc'
        alpha = torch.sigmoid(self.skip[ntype])
        trans_out = self.a_linears[ntype](G.nodes[ntype].data['t'])#+ G.time_emb) # TODO h? or ht
        trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
        if self.use_norm:
            G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
        else:
            G.nodes[ntype].data[out_key] = self.drop(trans_out)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_relations={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class RGCNLayerHeteroAsyn(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False):
        super(RGCNLayerHeteroAsyn, self).__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.num_types     = len(ntypes)
        self.num_relations = len(etypes) 
        
        self.use_norm    = use_norm

        self.weight = nn.ModuleDict()
        for etype in etypes:
            self.weight[etype] = nn.Linear(in_dim, out_dim)
 
        self.a_linears   = nn.ModuleDict()
        self.skip = nn.ParameterDict() 
        self.rnns = nn.ModuleDict()
        for ntype in ntypes:
            self.skip[ntype] = nn.Parameter(torch.ones(1))
            self.a_linears[ntype] = nn.Linear(out_dim,  out_dim)
            if ntype == 'topic':
                self.rnns[ntype] = nn.RNNCell(out_dim, out_dim)
            if use_norm:
                self.norms[ntype] = nn.LayerNorm(out_dim) 
        self.drop           = nn.Dropout(dropout)
        
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
 
    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        # print(G.canonical_etypes)
        funcs = {}
        for srctype, etype, dsttype in [('topic', 'tt', 'topic'), ('word', 'wt', 'topic')]:

            Wh = self.weight[etype](G.nodes[srctype].data[inp_key])   #   feat_dict[srctype]
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 't'))

        G.multi_update_all(funcs, 'sum')

        # msg at last time
        ntype = 'topic'
        alpha = torch.sigmoid(self.skip[ntype])
        trans_out = self.a_linears[ntype](G.nodes[ntype].data['t'])# + G.time_emb) # TODO h? or ht
        trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
        trans_out = self.rnns[ntype](trans_out, G.nodes[ntype].data['ht-1'])
        if self.use_norm:
            G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
        else:
            G.nodes[ntype].data[out_key] = self.drop(trans_out)
        G.nodes[ntype].data[inp_key] = G.nodes[ntype].data[out_key]

        edge_dict = []

        funcs = {}
        for srctype, etype, dsttype in [('topic', 'td', 'doc'), ('word', 'wd', 'doc')]:

            Wh = self.weight[etype](G.nodes[srctype].data[inp_key])   #   feat_dict[srctype]
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 't'))

        G.multi_update_all(funcs, 'sum')

        ntype = 'doc'
        alpha = torch.sigmoid(self.skip[ntype])
        trans_out = self.a_linears[ntype](G.nodes[ntype].data['t'])#+ G.time_emb) # TODO h? or ht
        trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
        if self.use_norm:
            G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
        else:
            G.nodes[ntype].data[out_key] = self.drop(trans_out)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_relations={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)



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

class tempMP6(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(tempMP6, self).__init__()
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
        self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        self.rnns = nn.ModuleDict({
            'word':nn.RNNCell(n_hid, n_hid),
            'topic':nn.RNNCell(n_hid, n_hid),
        })
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        node_dict = {'doc': 0, 'topic': 1, 'word': 2}
        edge_dict = {'td': 0, 'tt': 1, 'wd': 2, 'wt': 3, 'ww': 4}
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(TempMessagePassingLayer6(n_hid, n_hid, len(node_dict), len(edge_dict), n_heads, use_norm = use_norm))
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
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx}, 
                                        # preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            sub_bg.time_emb = time_emb
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')

            for ntype in ['word','topic']:
                sub_bg.nodes[ntype].data['ht-1'] = self.rnns[ntype](sub_bg.nodes[ntype].data['ht'], sub_bg.nodes[ntype].data['ht-1'])
            # update h to bg
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype]] = sub_bg.nodes[ntype].data[key]

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



# bad results
class tempMP5(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(tempMP5, self).__init__()
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
        # self.time_emb = RelTemporalEncoding(n_hid, seq_len)
        if self.pool == 'attn':
            self.attn_pool = GlobalAttentionPooling(n_hid, n_hid)
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        self.rnns = nn.ModuleDict({
            'word':nn.RNNCell(n_hid, n_hid),
            # 'topic':nn.RNNCell(n_hid, n_hid),
        })
        self.gcn_word_layers = nn.ModuleList()
        # self.gcn_topic_layers = nn.ModuleList()

        # self.gcn_word_layers.append(GCNLayer(n_inp, n_hid, activation, dropout))
        for _ in range(n_layers):
            self.gcn_word_layers.append(GCNLayer(n_hid, n_hid, activation, dropout))
            # self.gcn_topic_layers.append(GCNLayer(n_hid, n_hid, activation, dropout))

        self.hetero_layers = nn.ModuleList()
        for _ in range(1):
            self.hetero_layers.append(HGTLayerHeteroAsyn(n_hid, n_hid, etypes=['wt','wd','td','tt'],ntypes=['word','topic','doc'], n_heads=4, dropout=dropout))
            # self.hetero_layers.append(RGCNLayerHeteroAsyn(n_hid, n_hid, etypes=['wt','wd','td','tt'],ntypes=['word','topic','doc'], n_heads=4, dropout=dropout))

        
        # self.gcs = nn.ModuleList()
        # for _ in range(n_layers):
        #     self.gcs.append(TempMessagePassingLayer3(n_hid, n_hid, len(node_dict), len(edge_dict), n_heads, use_norm = use_norm))
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
            # time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
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
            # sub_bg.time_emb = time_emb
            for layer in self.gcn_word_layers:
                layer(sub_bg, inp_key='h0', out_key='ht', ntype='word', etype='ww')
            word_t = sub_bg.nodes['word'].data['ht']
            word_t_1 = sub_bg.nodes['word'].data['ht-1']
            sub_bg.nodes['word'].data['h0'] = self.rnns['word'](word_t, word_t_1)

            for i in range(1):
                self.hetero_layers[i](sub_bg, 'h0', 'ht')
            # topic_t = sub_bg.nodes['topic'].data['ht']
            # topic_t_1 = sub_bg.nodes['topic'].data['ht-1']
            # sub_bg.nodes['topic'].data['ht'] = self.rnns['topic'](topic_t, topic_t_1)
            # update h to bg
            for ntype in ['word','topic','doc']:
                # print('==',ntype,sub_bg.nodes[ntype].data['h'].shape,bg.nodes[ntype].data['h'].shape)
                # print('ntype',ntype)
                bg.nodes[ntype].data['ht-1'][orig_node_ids[ntype]] = sub_bg.nodes[ntype].data['ht']
                # bg.nodes[ntype].data['timeh'][orig_node_ids[ntype]] = time_emb
        
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='ht-1',ntype='doc')
            global_word_info = dgl.max_nodes(bg, feat='ht-1',ntype='word')
            global_topic_info = dgl.max_nodes(bg, feat='ht-1',ntype='topic')
            global_info = torch.cat((global_doc_info, global_word_info, global_topic_info),-1)
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='ht-1',ntype='doc')
            global_word_info = dgl.mean_nodes(bg, feat='ht-1',ntype='word')
            global_topic_info = dgl.mean_nodes(bg, feat='ht-1',ntype='topic')
            global_info = torch.cat((global_doc_info, global_word_info, global_topic_info),-1)
        elif self.pool == 'attn':
            attn_pool_out = self.attn_pool(bg, 'ht-1')
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


