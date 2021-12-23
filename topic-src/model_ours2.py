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
 
 
 
# very similar to 3
class message_passing(nn.Module):
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
        # self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            # self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
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
            # relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype] 
            # key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            key   = edges.src['k']
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

 
# with time
class causal_message_passing_rdm(nn.Module):
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
        # self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            # self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))

        self.relation_pri_cau = nn.ParameterDict()
        # self.relation_att_cau = nn.ParameterDict()
        self.relation_msg_cau = nn.ParameterDict()
        self.comb_pri = nn.ParameterDict()
        for etype in ['tw','tt','td']:
            self.relation_pri_cau[etype] = nn.Parameter(torch.ones(self.n_heads))
            # self.relation_att_cau[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg_cau[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.comb_pri[etype] = nn.Parameter(torch.ones(n_heads, self.d_k))
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
            # relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype] 
            key   = edges.src['k']
            # key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            cau_att = None
            cau_val = None
            if etype in ['tw','tt','td']:
                # if src is causal node
                # causal edges
                # src node 
                cau_types = edges.src['cau_type'] # 0,1,2,3  learn and mask out 0 type
                # relation_att_cau = self.relation_att_cau[etype]
                relation_pri_cau = self.relation_pri_cau[etype]
                relation_msg_cau = self.relation_msg_cau[etype] 
                # cau_key = torch.bmm(edges.src['k'].transpose(1,0), relation_att_cau).transpose(1,0)
                cau_key = edges.src['k']
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
    
    # def reduce_func(self, nodes):
    #     att = F.softmax(nodes.mailbox['a'], dim=1)
    #     h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
    #     if 'ca' in nodes.mailbox:
    #         cau_att = F.softmax(nodes.mailbox['ca'], dim=1) # spasemax TODO
    #         cau_h   = torch.sum(cau_att.unsqueeze(dim = -1) * nodes.mailbox['cv'], dim=1)
    #         h = h + cau_h
    #     return {'t': h.view(-1, self.out_dim)}
    
    def reduce_func(self, etype):
        def reduce(nodes):
            att = F.softmax(nodes.mailbox['a'], dim=1)
            h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
            if 'ca' in nodes.mailbox:
                cau_att = F.softmax(nodes.mailbox['ca'], dim=1) # spasemax TODO
                cau_h   = torch.sum(cau_att.unsqueeze(dim = -1) * nodes.mailbox['cv'], dim=1)
                # print(self.comb_pri[etype].shape,'self.comb_pri[etype]',cau_h.shape,'cau_h')
                h = h + cau_h * self.comb_pri[etype]
            return {'t': h.view(-1, self.out_dim)}
        return reduce

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
           
        G.multi_update_all({etype : (self.message_func, self.reduce_func(etype)) \
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

 
class causal_message_passing(nn.Module):
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
        # self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            # self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))

        self.relation_pri_cau = nn.ParameterDict()
        # self.relation_att_cau = nn.ParameterDict()
        self.relation_msg_cau = nn.ParameterDict()
        for etype in ['tw','tt','td']:
            self.relation_pri_cau[etype] = nn.Parameter(torch.ones(self.n_heads))
            # self.relation_att_cau[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
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
            # relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype] 
            # print(relation_msg.shape,'relation_msg')
            key = edges.src['k']
            # key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            cau_att = None
            cau_val = None
            if etype in ['tw','tt','td']:
                cau_types = edges.src['cau_type'] # 0,1,2,3  learn and mask out 0 type
                # relation_att_cau = self.relation_att_cau[etype]
                relation_pri_cau = self.relation_pri_cau[etype]
                relation_msg_cau = self.relation_msg_cau[etype]
                # cau_key = torch.bmm(edges.src['k'].transpose(1,0), relation_att_cau).transpose(1,0)
                cau_key = edges.src['k']
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

 
 
# no causal main
class ours_temp(nn.Module):
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
            self.gcs.append(message_passing(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
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

 

class ours_causal(nn.Module):
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
                self.gcs.append(causal_message_passing_rdm(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm, device=self.device))
            else:
                self.gcs.append(causal_message_passing(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm, device=self.device))
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
