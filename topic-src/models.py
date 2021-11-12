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
 
 

############## do not update word emb first

class WordConvLayer2(nn.Module):
    def __init__(self, inpu_size, out_size, dropout):
        super().__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(inpu_size, out_size),
            }) 
        self.drop = nn.Dropout(dropout)
    def forward(self, G):
        # print(G,feat_dict,'G,feat_dict')
        funcs={}
        for cano_etype in [('word','ww','word')]:
            srctype, etype, dsttype = cano_etype 
            G.nodes[srctype].data['h'] = self.weight[etype](G.nodes[srctype].data['h'])  
            # dst_degs = G.in_degrees(G.nodes(dsttype), cano_etype).clamp(min=1).float()
            # G.nodes[dsttype].data['norm'] = (1. / dst_degs) 
            # print('norm   ',G.nodes[dsttype].data['norm'].shape)
            # print('weight   ',G.edges[etype].data['weight'].shape)
            # G.apply_edges(lambda edges: {'weight': edges.dst['norm'] * edges.data['weight']}, etype=cano_etype)
            funcs[etype] = (fn.u_mul_e('h', 'weight', 'm'), fn.mean('m', 'h')) 

        G.multi_update_all(funcs, 'sum')
        # return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
        for ntype in ['word']: 
            G.nodes[ntype].data['h'] = self.drop(F.relu(G.nodes[ntype].data['h']))
 
class HeteroConvLayer2(nn.Module):
    def __init__(self, inpu_size, out_size, dropout):
        super().__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(inpu_size, out_size),
                'wt': nn.Linear(inpu_size, out_size),
                'wd': nn.Linear(inpu_size, out_size),
                'td': nn.Linear(inpu_size, out_size),
                'tt': nn.Linear(inpu_size, out_size),
            }) 
        self.drop = nn.Dropout(dropout)
    def forward(self, G):
        # print(G,feat_dict,'G,feat_dict')
        funcs={}
        # G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        for cano_etype in G.canonical_etypes:
            srctype, etype, dsttype = cano_etype 
            # print('cano_etype',cano_etype)
            # print('srctype',srctype,etype,G.nodes[srctype].data['h'].shape)
            G.nodes[srctype].data['h'] = self.weight[etype](G.nodes[srctype].data['h'])  
            # dst_degs = G.in_degrees(G.nodes(dsttype), cano_etype).clamp(min=1).float()
            # G.nodes[dsttype].data['norm'] = (1. / dst_degs) 
            # # * G.nodes[dsttype].data['weight'] 
            # G.apply_edges(lambda edges: {'weight': edges.dst['norm']*edges.data['weight'] }, etype=cano_etype)

            funcs[etype] = (fn.u_mul_e('h', 'weight', 'm'), fn.mean('m', 'h')) 

        G.multi_update_all(funcs, 'sum')
        # return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
        for ntype in G.ntypes: 
            G.nodes[ntype].data['h'] = self.drop(F.relu(G.nodes[ntype].data['h']))
 

class HeteroConvLayer(nn.Module):
    def __init__(self, word_in_size, topic_in_size, out_size):
        super().__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(word_in_size, out_size),
                'wt': nn.Linear(word_in_size, out_size),
                'wd': nn.Linear(word_in_size, out_size),
                'td': nn.Linear(topic_in_size, out_size),
                'tt': nn.Linear(topic_in_size, out_size),
            }) 

    def forward(self, G, feat_dict):
        # print(G,feat_dict,'G,feat_dict')
        funcs={}
        # G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        G.edges['wd'].data['weight'] = G.edges['wd'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            # print('srctype, etype, dsttype',srctype, etype, dsttype,feat_dict[srctype].shape) 
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # print(etype,G.edges[etype].data['weight'].dtype,Wh.dtype)
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
 
class HeteroTopicLayer(nn.Module):
    def __init__(self, word_in_size, topic_in_size, out_size):
        super().__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(word_in_size, out_size),
                'wt': nn.Linear(word_in_size, out_size),
                # 'wd': nn.Linear(word_in_size, out_size),
                # 'td': nn.Linear(topic_in_size, out_size),
                'tt': nn.Linear(topic_in_size, out_size),
            }) 

    def forward(self, G, feat_dict):
        # print(G,feat_dict,'G,feat_dict')
        funcs={}
        G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype in ['wd','td']:
                continue
            # print('srctype, etype, dsttype',srctype, etype, dsttype,feat_dict[srctype].shape) 
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')
        return {ntype : G.nodes[ntype].data['h'] for ntype in ['word','topic']}
 
 
 
class HeteroConvCausalLayer0(nn.Module):
    def __init__(self, word_in_size, topic_in_size, out_size, device):
        super().__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(word_in_size, out_size),
                'wt': nn.Linear(word_in_size, out_size),
                'wd': nn.Linear(word_in_size, out_size),
                'td': nn.Linear(topic_in_size, out_size),
                'tt': nn.Linear(topic_in_size, out_size),
                'td_cau': nn.Linear(topic_in_size, out_size, bias=True),
                'td_noi': nn.Linear(topic_in_size, out_size, bias=True),
                'tt_cau': nn.Linear(topic_in_size, out_size, bias=True),
                'tt_noi': nn.Linear(topic_in_size, out_size, bias=True),
                'td_cau_trans': nn.Linear(3, 1,bias=False),
                'td_noi_trans': nn.Linear(3, 1,bias=False),
                'tt_cau_trans': nn.Linear(3, 1,bias=False),
                'tt_noi_trans': nn.Linear(3, 1,bias=False),
            }) 
        self.device = device

    def forward(self, G, feat_dict):
        funcs={}
        for srctype, etype, dsttype in G.canonical_etypes:
            node_emb = feat_dict[srctype]
            if srctype == 'topic':
                effect = G.nodes['topic'].data['effect'].to_dense().float()  # sparse
                effect = (effect!=0) * 1.
                random_mask = torch.bernoulli(0.1*torch.ones(effect.size()).to(self.device)) * (effect==0)#.view(-1, 1, -1)
                effect_w = self.weight['%s_cau_trans' % etype](effect)
                noise_w = self.weight['%s_noi_trans' % etype](random_mask)
                Wh = self.weight[etype](node_emb) + self.weight['%s_cau' % etype](node_emb * effect_w) - self.weight['%s_noi' % etype](node_emb * noise_w)
            else:
                Wh = self.weight[etype](node_emb)
            # Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

 
class HeteroConvCausalLayer1(nn.Module):
    def __init__(self, word_in_size, topic_in_size, out_size, device):
        super().__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(word_in_size, out_size),
                'wt': nn.Linear(word_in_size, out_size),
                'wd': nn.Linear(word_in_size, out_size),
                'td': nn.Linear(topic_in_size, out_size),
                'tt': nn.Linear(topic_in_size, out_size),
                # 'td_cau': nn.Linear(topic_in_size, out_size, bias=True),
                # 'td_noi': nn.Linear(topic_in_size, out_size, bias=True),
                # 'tt_cau': nn.Linear(topic_in_size, out_size, bias=True),
                # 'tt_noi': nn.Linear(topic_in_size, out_size, bias=True),
                # 'cau': nn.Linear(topic_in_size, out_size, bias=True),
                # 'noi': nn.Linear(topic_in_size, out_size, bias=True),
                # 'td_cau_trans': nn.Linear(3, 1,bias=True),
                # 'td_noi_trans': nn.Linear(3, topic_in_size,bias=False),
                # 'tt_cau_trans': nn.Linear(3, 1,bias=True),
                # 'tt_noi_trans': nn.Linear(3, topic_in_size,bias=False),
                # 'td_cau_weight':nn.Linear(3, 1)
            }) 
        # self.td_cau_weight = nn.Parameter(torch.Tensor(3, topic_in_size, out_size))
        # self.tt_cau_weight = nn.Parameter(torch.Tensor(3, topic_in_size, out_size))
        self.device = device
        self.weight_causal = nn.Linear(out_size, out_size,bias=False)
        self.weight_noise = nn.Linear(out_size, out_size,bias=False)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, G, feat_dict):
        # print(G,feat_dict,'G,feat_dict')
        funcs={}
        G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        for cano_etype in G.canonical_etypes:
            srctype, etype, dsttype = cano_etype
            node_emb = feat_dict[srctype]
            if srctype == 'topic':
                effect = G.nodes['topic'].data['effect'].view(-1,1)#.to_dense().float()  # sparse
                # print(effect.shape,effect.sum(),effect.mean(),effect.min(),effect.max())
                # num_time = effect.size(-1)
                # effect = (effect!=0) * 1.
                pos_mask = (effect > 0)*(1.)#.view(-1,1)
                neg_mask = (effect < 0)*(1.)#.view(-1,1) 
                Wh = self.weight[etype](node_emb) + self.weight_causal(node_emb * pos_mask) - self.weight_noise(node_emb * neg_mask)
                # effect = (effect > 0)+(effect < 0)*(-1.) 
                # print(effect.shape,'effect',node_emb.shape)
                # for i in range(num_time):
                # node_emb_repeated = node_emb.unsqueeze(0).repeat(num_time,1,1)
                # # print(node_emb_repeated.shape)
                # if etype == 'td':
                #     Wh = torch.bmm(node_emb_repeated,self.td_cau_weight)
                # else:
                #     Wh = torch.bmm(node_emb_repeated,self.tt_cau_weight)
                # Wh = Wh * torch.t(effect).unsqueeze(-1)
                # print(Wh,'Wh')
                # print(Wh.shape,'Wh2',Wh.nonzero().size())
                # print(Wh.sum(-1).nonzero().size(),'======')
                # random_mask = torch.bernoulli(0.1*torch.ones(effect.size()).to(self.device)) * (effect==0)#.view(-1, 1, -1)
                # effect_gate = torch.sigmoid(self.weight['%s_cau_trans' % etype](effect))
                # # print(effect_gate)
                # # causal_gate * node_emb 
                # Wh =  torch.tanh(self.weight['%s_cau' % etype](node_emb))*effect_gate + \
                #     torch.tanh(self.weight['%s_noi' % etype](node_emb))*(1-effect_gate)
                # ∂*f(x) + (1-∂)*g(x) 
                # print(Wh)
            else:
                # print('srctype, etype, dsttype',srctype, etype, dsttype) 
                Wh = self.weight[etype](node_emb)

            # dst_degs = G.in_degrees(G.nodes(dsttype), cano_etype).clamp(min=1.).float()
            # G.nodes[dsttype].data['norm'] = (1. / dst_degs) 
            # G.apply_edges(lambda edges: {'weight': edges.dst['norm']*edges.data['weight'].float() }, etype=cano_etype)
            # print('srctype, etype, dsttype',srctype, etype, dsttype,feat_dict[srctype].shape) 
            # Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}



class TopicConvCausalLayer0(nn.Module):
    def __init__(self, word_in_size, topic_in_size, out_size, device):
        super().__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(word_in_size, out_size),
                'wt': nn.Linear(word_in_size, out_size),
                'tt': nn.Linear(topic_in_size, out_size), 
            }) 
        # self.td_cau_weight = nn.Parameter(torch.Tensor(3, topic_in_size, out_size))
        self.tt_cau_weight = nn.Parameter(torch.Tensor(3, topic_in_size, out_size))
        self.device = device
        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, G, feat_dict):
        # print(G,feat_dict,'G,feat_dict')
        funcs={}
        # G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype in ['td','wd']:
                continue
            node_emb = feat_dict[srctype]
            if srctype == 'topic':
                effect = G.nodes['topic'].data['effect'].to_dense().float()  # sparse
                # print(effect.shape)
                num_time = effect.size(-1)
                effect = (effect > 0)+(effect < 0)*(-1.) 
                node_emb_repeated = node_emb.unsqueeze(0).repeat(num_time,1,1)
                # print(node_emb_repeated.shape)
                Wh = torch.bmm(node_emb_repeated,self.tt_cau_weight)
                # print(Wh.shape,'Wh')
                Wh = Wh * torch.t(effect).unsqueeze(-1) 
                Wh = self.weight[etype](node_emb) + Wh.mean(0)
                # W_etype x h + 1/|t| (sum (W_etype' x h) * I(causal? 1 or -1))
            else:
                # print('srctype, etype, dsttype',srctype, etype, dsttype) 
                Wh = self.weight[etype](node_emb)
            # print('srctype, etype, dsttype',srctype, etype, dsttype,feat_dict[srctype].shape) 
            # Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')
        return {ntype : G.nodes[ntype].data['h'] for ntype in ['word','topic']}


class HeteroCausalBeta(nn.Module):
    def __init__(self, word_in_size, topic_in_size, hidden_size, out_size, device, dropout, layer='word'):
        super().__init__()  
        if layer == 'word':
            self.layer1 = WordGraphLayer(word_in_size, hidden_size)
            self.layer2 = WordGraphLayer(hidden_size, out_size)
        elif layer == 'hetero':
            self.layer1 = HeteroConvLayer(word_in_size, topic_in_size, hidden_size)
            self.layer2 = HeteroConvLayer(hidden_size, hidden_size, out_size)
        elif layer == 'topic':
            self.layer1 = HeteroTopicLayer(word_in_size, topic_in_size, hidden_size)
            self.layer2 = HeteroTopicLayer(hidden_size, hidden_size, out_size)
        elif layer == 'cau0':
            self.layer1 = HeteroConvCausalLayer0(word_in_size, topic_in_size, hidden_size, device)
            self.layer2 = HeteroConvCausalLayer0(hidden_size, hidden_size, out_size, device)
        elif layer == 'cau1':
            self.layer1 = HeteroConvCausalLayer1(word_in_size, topic_in_size, hidden_size, device)
            self.layer2 = HeteroConvCausalLayer1(hidden_size, hidden_size, out_size, device)
        elif layer == 'tcau0':
            self.layer1 = TopicConvCausalLayer0(word_in_size, topic_in_size, hidden_size, device)
            self.layer2 = TopicConvCausalLayer0(hidden_size, hidden_size, out_size, device)
            
        self.drop = nn.Dropout(dropout)
        # self.layer = layer
        # if layer == 'cau1':
        #     self.norms = nn.ModuleDict({
        #         'word':nn.LayerNorm(hidden_size,elementwise_affine=True),
        #         'topic':nn.LayerNorm(hidden_size,elementwise_affine=True),
        #         'doc':nn.LayerNorm(hidden_size,elementwise_affine=True)
        #     })
        
    def forward(self, G, emb_dict):
        h_dict = self.layer1(G, emb_dict)
        # if self.layer == 'cau1':
        #     h_dict = {k : self.drop(self.norms[k](F.leaky_relu(h))) for k, h in h_dict.items()}
        # else:
        h_dict = {k : self.drop(F.relu(h)) for k, h in h_dict.items()}
        # h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # if self.layer == 'cau1':
        #     h_dict = {k : self.drop(self.norms[k](F.leaky_relu(h))) for k, h in h_dict.items()}
        # else:
        h_dict = {k : self.drop(F.relu(h)) for k, h in h_dict.items()}
        # h_dict = {k : self.drop(F.relu(h)) for k, h in h_dict.items()}
        return h_dict

class HeteroCausalBeta2(nn.Module):
    def __init__(self, inpu_size, hidden_size, out_size, device, dropout, layer='word'):
        super().__init__()  
        if layer == 'word':
            self.layers = nn.ModuleList([
                WordConvLayer2(inpu_size, hidden_size, dropout),
                WordConvLayer2(hidden_size, out_size, dropout)])
        elif layer == 'hetero':
            self.layers = nn.ModuleList([
                HeteroConvLayer2(inpu_size, hidden_size, dropout),
                HeteroConvLayer2(hidden_size, out_size, dropout)])
        # elif layer == 'cau0':
        #     self.layer1 = HeteroConvCausalLayer0(word_in_size, topic_in_size, hidden_size, device)
        #     self.layer2 = HeteroConvCausalLayer0(hidden_size, hidden_size, out_size, device)
        # elif layer == 'cau1':
        #     self.layer1 = HeteroConvCausalLayer1(word_in_size, topic_in_size, hidden_size, device)
        #     self.layer2 = HeteroConvCausalLayer1(hidden_size, hidden_size, out_size, device)
        # self.drop = nn.Dropout(dropout)
        # self.norms = nn.ModuleDict({
        #     'word':nn.LayerNorm(hidden_size,elementwise_affine=True),
        #     'topic':nn.LayerNorm(hidden_size,elementwise_affine=True),
        #     'doc':nn.LayerNorm(hidden_size,elementwise_affine=True)
        # })
        
    def forward(self, G):
        for i in range(len(self.layers)):
            # print(i)
            self.layers[i](G)


##############
 
class WordGraphLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(WordGraphLayer, self).__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(in_size, out_size),
            }) 

    def forward(self, G, feat_dict):
        funcs={}

        for srctype, etype, dsttype in [['word','ww','word']]: 
            # norm = dglnn.EdgeWeightNorm(norm='both')
            # norm_edge_weight = norm(g, edge_weight)
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        return {'word' : G.nodes['word'].data['h']}

# https://www.jianshu.com/p/767950b560c4

# a static graph model

class static_heto_cau0(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        self.hconv = HeteroCausalBeta(h_inp, h_dim, h_dim, h_dim, self.device, dropout,layer='cau0')
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = torch.zeros((bg.number_of_nodes('doc'), self.h_dim)).to(self.device)
        emb_dict = {
            'word':word_emb,
            'topic':topic_emb,
            'doc':doc_emb
        }
        emb_dict = self.hconv(bg,emb_dict)
        bg.nodes['doc'].data['emb'] = emb_dict['doc']
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='emb',ntype='doc')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='emb',ntype='doc')
        y_pred = self.out_layer(global_doc_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred
 

class static_heto_cau1(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        self.hconv = HeteroCausalBeta(h_inp, h_dim, h_dim, h_dim, self.device, dropout,layer='cau1')
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = torch.zeros((bg.number_of_nodes('doc'), self.h_dim)).to(self.device)
        emb_dict = {
            'word':word_emb,
            'topic':topic_emb,
            'doc':doc_emb
        }
        emb_dict = self.hconv(bg,emb_dict)
        bg.nodes['doc'].data['emb'] = emb_dict['doc']
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='emb',ntype='doc')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='emb',ntype='doc')
        y_pred = self.out_layer(global_doc_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred
 
class static_topic_cau0(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        self.hconv = HeteroCausalBeta(h_inp, h_dim, h_dim, h_dim, self.device, dropout,layer='tcau0')
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        emb_dict = {
            'word':word_emb,
            'topic':topic_emb,
        }
        emb_dict = self.hconv(bg,emb_dict)
        bg.nodes['topic'].data['emb'] = emb_dict['topic']
        if self.pool == 'max':
            global_info = dgl.max_nodes(bg, feat='emb',ntype='topic')
        elif self.pool == 'mean':
            global_info = dgl.mean_nodes(bg, feat='emb',ntype='topic')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred
 
 

class static_topic_graph(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        # self.hconv = HeteroConvNet(h_inp, h_dim, h_dim, h_dim)
        self.hconv = HeteroCausalBeta(h_inp, h_dim, h_dim, h_dim, self.device, dropout,layer='topic')
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        emb_dict = {
            'word':word_emb,
            'topic':topic_emb,
        }
        emb_dict = self.hconv(bg,emb_dict)
        bg.nodes['topic'].data['emb'] = emb_dict['topic']
        if self.pool == 'max':
            global_info = dgl.max_nodes(bg, feat='emb',ntype='topic')
        elif self.pool == 'mean':
            global_info = dgl.mean_nodes(bg, feat='emb',ntype='topic')
        y_pred = self.out_layer(global_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred
 
class HeteroBasic(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1,h_dim))
        # self.hconv = HeteroConvNet(h_inp, h_dim, h_dim, h_dim)
        self.hconv = HeteroCausalBeta(h_inp, h_dim, h_dim, h_dim, self.device, dropout,layer='hetero')

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
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        # doc_emb = torch.zeros((bg.number_of_nodes('doc'), self.h_dim)).to(self.device)
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        emb_dict = {
            'word':word_emb,
            'topic':topic_emb,
            'doc':doc_emb
        }
        emb_dict = self.hconv(bg,emb_dict)
        bg.nodes['doc'].data['emb'] = emb_dict['doc']
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='emb',ntype='doc')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='emb',ntype='doc')
        y_pred = self.out_layer(global_doc_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred
 
class static_heto_graph2(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        # self.hconv = HeteroConvNet(h_inp, h_dim, h_dim, h_dim)
        self.hconv = HeteroCausalBeta2(h_dim, h_dim, h_dim, self.device, dropout,layer='hetero')
        self.adapt_inp = nn.Linear(h_inp,h_dim)
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = torch.zeros((bg.number_of_nodes('doc'), self.h_dim)).to(self.device)
        
        bg.nodes['word'].data['h'] = self.adapt_inp(word_emb)
        bg.nodes['topic'].data['h'] = topic_emb
        bg.nodes['doc'].data['h'] = doc_emb

        # for ntype in G.ntypes:
        #     n_id = G.node_dict[ntype]
        #     G.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))

        self.hconv(bg)
        # bg.nodes['doc'].data['emb'] = emb_dict['doc']
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='h',ntype='doc')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='h',ntype='doc')
        y_pred = self.out_layer(global_doc_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred
 


class static_word_graph(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None 
        # self.hconv = WordGraphNet(h_inp, h_dim, h_dim) 
        self.hconv = HeteroCausalBeta(h_inp, h_dim, h_dim, h_dim,self.device, dropout,layer='word')

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
        # bg = dgl.add_self_loop(bg,etype='tt')
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        emb_dict = self.hconv(bg, {'word':word_emb})
        bg.nodes['word'].data['emb'] = emb_dict['word']
        if self.pool == 'max':
            global_word_info = dgl.max_nodes(bg, feat='emb',ntype='word')
        elif self.pool == 'mean':
            global_word_info = dgl.mean_nodes(bg, feat='emb',ntype='word')
        y_pred = self.out_layer(global_word_info)
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred) 
        return loss, y_pred


class static_word_graph2(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None 
        self.hconv = HeteroCausalBeta2(h_dim, h_dim, h_dim,self.device, dropout,layer='word')
        self.adapt_inp = nn.Linear(h_inp,h_dim)
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
        # bg = dgl.add_self_loop(bg,'ww')
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        bg.nodes['word'].data['h'] = self.adapt_inp(word_emb)
        self.hconv(bg)
        if self.pool == 'max':
            global_word_info = dgl.max_nodes(bg, feat='h',ntype='word')
        elif self.pool == 'mean':
            global_word_info = dgl.mean_nodes(bg, feat='h',ntype='word')
        y_pred = self.out_layer(global_word_info)
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred) 
        return loss, y_pred


class static_hgt(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        # self.hconv = HGT(h_inp, h_dim, h_dim, n_layers=2, n_heads=4, use_norm = True)
        self.hconv = HGT(h_dim, h_dim, h_dim, n_layers=2, n_heads=4, use_norm = False)

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

        bg.node_dict = {}
        bg.edge_dict = {}
        for ntype in bg.ntypes:
            bg.node_dict[ntype] = len(bg.node_dict)
        for etype in bg.etypes:
            bg.edge_dict[etype] = len(bg.edge_dict)
            bg.edges[etype].data['id'] = torch.ones(bg.number_of_edges(etype), dtype=torch.long).to(self.device) * bg.edge_dict[etype] 

        # print(bg.node_dict)
        # print(bg.edge_dict)
        # bg.node_dict = {'doc': 0, 'topic': 1, 'word': 2}
        # bg.edge_dict = {'td': 0, 'tt': 1, 'wd': 2, 'wt': 3, 'ww': 4}
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        # word_emb = torch.zeros((bg.number_of_nodes('word'), self.h_dim)).to(self.device)
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = torch.zeros((bg.number_of_nodes('doc'), self.h_dim)).to(self.device)
        # emb_dict = {
        #     'word':word_emb,
        #     'topic':topic_emb,
        #     'doc':doc_emb
        # }
        bg.nodes['word'].data['h'] = word_emb
        bg.nodes['topic'].data['h'] = topic_emb
        bg.nodes['doc'].data['h'] = doc_emb
        emb = self.hconv(bg,'doc')
        # print(emb.shape,'emb_dict')
        bg.nodes['doc'].data['emb'] = emb
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='emb',ntype='doc')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='emb',ntype='doc')
        y_pred = self.out_layer(global_doc_info)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred


class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(num_types):
            # if t == 'word':
            #     self.k_linears.append(nn.Linear(300,   out_dim))
            #     self.q_linears.append(nn.Linear(300,   out_dim))
            #     self.v_linears.append(nn.Linear(300,   out_dim))
            # else:
            #     self.k_linears.append(nn.Linear(in_dim,   out_dim))
            #     self.q_linears.append(nn.Linear(in_dim,   out_dim))
            #     self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        
        # self.k_linears   = nn.ModuleDict()
        # self.q_linears   = nn.ModuleDict()
        # self.v_linears   = nn.ModuleDict()
        # self.a_linears   = nn.ModuleDict()
        # self.norms       = nn.ModuleDict()
        # for t in ['word','topic','doc']:
        #     self.k_linears[t] = nn.Linear(in_dim,   out_dim)
        #     self.q_linears[t] = nn.Linear(in_dim,   out_dim)
        #     self.v_linears[t] = nn.Linear(in_dim,   out_dim)
        #     self.a_linears[t] = nn.Linear(out_dim,  out_dim)
        #     if use_norm:
        #         self.norms[t] = nn.LayerNorm(out_dim)
            
        # self.relation_pri = nn.ModuleDict()
        # self.relation_att = nn.ModuleDict()
        # self.relation_msg = nn.ModuleDict()
        # for r in ['td','tt','wd','wt','ww']:
        #     self.relation_pri[r] = nn.Parameter(torch.ones(self.n_heads))
        #     self.relation_att[r] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        #     self.relation_msg[r] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges):
        etype = edges.data['id'][0]
        relation_att = self.relation_att[etype]
        relation_pri = self.relation_pri[etype]
        relation_msg = self.relation_msg[etype]
        key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
        att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
        val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
        return {'a': att, 'v': val}
    
    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}
        
    def forward(self, G, inp_key, out_key):
        node_dict, edge_dict = G.node_dict, G.edge_dict
        for srctype, etype, dsttype in G.canonical_etypes:
            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]] 
            q_linear = self.q_linears[node_dict[dsttype]]
            
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            
            G.apply_edges(func=self.edge_attention, etype=etype)
        G.multi_update_all({etype : (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer = 'mean')
        for ntype in G.ntypes:
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            trans_out = self.a_linears[n_id](G.nodes[ntype].data['t'])
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[n_id](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)
    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class HGT(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(HGT, self).__init__()
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()

        node_dict = {'doc': 0, 'topic': 1, 'word': 2}
        edge_dict = {'td': 0, 'tt': 1, 'wd': 2, 'wt': 3, 'ww': 4}

        for t in range(len(node_dict)):
            if t == 2:
                self.adapt_ws.append(nn.Linear(300,  n_hid))
            else:
                self.adapt_ws.append(nn.Linear(n_inp,  n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, len(node_dict), len(edge_dict), n_heads, use_norm = use_norm))
        # self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, out_key):
        for ntype in G.ntypes:
            if ntype == 'word':
                n_id = G.node_dict[ntype]
                G.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['h']))
        for i in range(self.n_layers):
            self.gcs[i](G, 'h', 'h')
        return G.nodes[out_key].data['h']
        # return self.out(G.nodes[out_key].data['h'])

    def __repr__(self):
        return '{}(n_inp={}, n_hid={}, n_out={}, n_layers={})'.format(
            self.__class__.__name__, self.n_inp, self.n_hid,
            self.n_out, self.n_layers)

# python train.py --datafiles data_static_2014-01-01_2015-01-01_tt85_ww10_3 --horizon 5 --gpu 4 --seed 999 --train 0.6 --val 0.2 -m hgt --n-hidden 32 --special 14
# a temporal graph model

# batch all graph first, use hetero conv, then use a rnn
 
class temp_heto_graph(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        
        # self.hconv = HeteroConvNet(h_inp, h_dim, h_dim, h_dim)
        self.hconv = HeteroCausalBeta2(h_dim, h_dim, h_dim,self.device, dropout,layer='hetero')
        self.adapt_inp = nn.Linear(h_inp,h_dim)
        self.rnn = nn.RNN(h_dim, h_dim, num_layers=2, batch_first=True, dropout=dropout)
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
        # g_list: [[g,g,g],[g,g,g,g,g]]
        # sort by len, 
        # g_list = [[0,1,2,3],[0,1,2,3,4,5],[0,1,2,3,4,5,6,7]]
        # g_list_len = torch.LongTensor(list(map(len, g_list)))
        g_list_len = torch.IntTensor(list(map(len, g_list)))#.to(self.device)
        # print('g_list_len',g_list_len)
        # g_list_len = g_list_len.to(self.device)
        g_len, idx = g_list_len.sort(0, descending=True)
        num_non_zero = len(torch.nonzero(g_len)) # on zero, this step can be removed
        g_len = g_len.int()
        g_len_non_zero = g_len[:num_non_zero]
        if torch.max(g_list_len) == 0:
            print('all are empty list in g_list')
            exit()  

        y_data_sorted = y_data[idx]
        g_list_sorted_flat = []
        num_doc = []
        for id in idx:
            g_list_sorted_flat += g_list[id]
            for g_t in g_list[id]:
                num_doc.append(g_t.num_nodes('doc'))
        
        bg = dgl.batch(g_list_sorted_flat).to(self.device) 
 
        # bg = dgl.batch(g_list).to(self.device) 
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = torch.zeros((bg.number_of_nodes('doc'), self.h_dim)).to(self.device)
        
        bg.nodes['word'].data['h'] = self.adapt_inp(word_emb)
        bg.nodes['topic'].data['h'] = topic_emb
        bg.nodes['doc'].data['h'] = doc_emb
 
        self.hconv(bg)

        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='h',ntype='doc')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='h',ntype='doc')
        # print('global_doc_info',global_doc_info.shape)
        # doc_len = [g.num_nodes('doc') for g in g_list]
        doc_emb_split = torch.split(global_doc_info, g_len.tolist())
        # print(len(doc_emb_split),'len doc_emb_split',doc_emb_split[0].shape)
        embed_seq_tensor = torch.zeros(num_non_zero, self.seq_len, self.h_dim).to(self.device)
        for i, embeds in enumerate(doc_emb_split): 
            embed_seq_tensor[i, torch.arange(len(embeds)), :] = embeds
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               g_len,
                                                               batch_first=True)
        # print(packed_input,'packed_input')
        output, hn = self.rnn(packed_input)
        # print(hn.shape,'hn','output')
        hn = hn[-1] 
        y_pred = self.out_layer(hn)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data_sorted)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

class temp_word_graph(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        
        # self.hconv = WordGraphNet(h_inp, h_dim, h_dim)
        self.hconv = HeteroCausalBeta2(h_dim, h_dim, h_dim,self.device, dropout,layer='word')
        self.adapt_inp = nn.Linear(h_inp,h_dim)
        self.rnn = nn.RNN(h_dim, h_dim, num_layers=2, batch_first=True, dropout=dropout)
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
        # g_list: [[g,g,g],[g,g,g,g,g]]
        # sort by len, 
        # g_list = [[0,1,2,3],[0,1,2,3,4,5],[0,1,2,3,4,5,6,7]]
        # g_list_len = torch.LongTensor(list(map(len, g_list)))
        g_list_len = torch.IntTensor(list(map(len, g_list)))#.to(self.device)
        # print('g_list_len',g_list_len)
        # g_list_len = g_list_len.to(self.device)
        g_len, idx = g_list_len.sort(0, descending=True)
        num_non_zero = len(torch.nonzero(g_len)) # on zero, this step can be removed
        g_len = g_len.int()
        g_len_non_zero = g_len[:num_non_zero]
        if torch.max(g_list_len) == 0:
            print('all are empty list in g_list')
            exit()  

        y_data_sorted = y_data[idx]
        g_list_sorted_flat = []
        num_doc = []
        for id in idx:
            g_list_sorted_flat += g_list[id]
            for g_t in g_list[id]:
                num_doc.append(g_t.num_nodes('doc'))
        
        bg = dgl.batch(g_list_sorted_flat).to(self.device) 

        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        bg.nodes['word'].data['h'] = self.adapt_inp(word_emb)
        self.hconv(bg)
        # bg.nodes['doc'].data['emb'] = emb_dict['doc']
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='h',ntype='word')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='h',ntype='word')
        
        # print('global_doc_info',global_doc_info.shape)
        # doc_len = [g.num_nodes('doc') for g in g_list]
        doc_emb_split = torch.split(global_doc_info, g_len.tolist())
        # print(len(doc_emb_split),'len doc_emb_split',doc_emb_split[0].shape)
        embed_seq_tensor = torch.zeros(num_non_zero, self.seq_len, self.h_dim).to(self.device)
        for i, embeds in enumerate(doc_emb_split): 
            embed_seq_tensor[i, torch.arange(len(embeds)), :] = embeds
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               g_len,
                                                               batch_first=True)
        # print(packed_input,'packed_input')
        output, hn = self.rnn(packed_input)
        # print(hn.shape,'hn','output')
        hn = hn[-1] 
        y_pred = self.out_layer(hn)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data_sorted)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred


class temp_word_graph2(nn.Module):
    def __init__(self, h_inp, vocab_size, h_dim, device, seq_len=7, num_topic=50, num_word=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.h_inp = h_inp
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.num_topic = num_topic
        # self.num_rels = num_rels
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        # initialize rel and ent embedding
        # self.word_embeds = nn.Parameter(torch.Tensor(num_word, h_dim)) # change it to blocks
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, h_dim))
        
        # self.hconv = WordGraphNet(h_inp, h_dim, h_dim)
        self.hconv = HeteroCausalBeta2(h_dim, h_dim, 1, self.device, dropout,layer='word')
        self.adapt_inp = nn.Linear(h_inp,h_dim)
        self.rnn = nn.RNN(vocab_size, h_dim, num_layers=1, batch_first=True, dropout=dropout)
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
        # g_list: [[g,g,g],[g,g,g,g,g]]
        # sort by len, 
        # g_list = [[0,1,2,3],[0,1,2,3,4,5],[0,1,2,3,4,5,6,7]]
        # g_list_len = torch.LongTensor(list(map(len, g_list)))
        g_list_len = torch.IntTensor(list(map(len, g_list)))#.to(self.device)
        # print('g_list_len',g_list_len)
        # g_list_len = g_list_len.to(self.device)
        g_len, idx = g_list_len.sort(0, descending=True)
        num_non_zero = len(torch.nonzero(g_len)) # on zero, this step can be removed
        g_len = g_len.int()
        g_len_non_zero = g_len[:num_non_zero]
        if torch.max(g_list_len) == 0:
            print('all are empty list in g_list')
            exit()  

        y_data_sorted = y_data[idx]
        g_list_sorted_flat = []
        num_doc = []
        for id in idx:
            g_list_sorted_flat += g_list[id]
            for g_t in g_list[id]:
                num_doc.append(g_t.num_nodes('doc'))
        
        bg = dgl.batch(g_list_sorted_flat).to(self.device) 

        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        bg.nodes['word'].data['h'] = self.adapt_inp(word_emb)
        self.hconv(bg)
        embed_tensor = torch.zeros(len(g_list_sorted_flat), 15000).to(self.device)
        ids = bg.nodes['word'].data['id']
        learned_emb = bg.nodes['word'].data['h'] 
        word_len = [g.num_nodes('word') for g in g_list_sorted_flat] # word len for each sperate graph
        learned_emb_split = torch.split(learned_emb, word_len)
        ids_split = torch.split(ids, word_len)
        # for i in range(len(word_len)):
        for i, embeds in enumerate(learned_emb_split): 
            # print(embeds.shape,i,ids_split[i].shape)
            id = ids_split[i]
            embed_tensor[i, id] = embeds.view(-1)
        # print(ids,'ids',embed_tensor,embed_tensor.shape)
        # bg.nodes['doc'].data['emb'] = emb_dict['doc']
        # if self.pool == 'max':
        #     global_doc_info = dgl.max_nodes(bg, feat='h',ntype='word')
        # elif self.pool == 'mean':
        #     global_doc_info = dgl.mean_nodes(bg, feat='h',ntype='word')
        
        # print('global_doc_info',global_doc_info.shape)
        # doc_len = [g.num_nodes('doc') for g in g_list]
        doc_emb_split = torch.split(embed_tensor, g_len.tolist())
        # print(len(doc_emb_split),'len doc_emb_split',doc_emb_split[0].shape)
        embed_seq_tensor = torch.zeros(num_non_zero, self.seq_len, 15000).to(self.device)
        for i, embeds in enumerate(doc_emb_split): 
            embed_seq_tensor[i, torch.arange(len(embeds)), :] = embeds
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               g_len,
                                                               batch_first=True)
        # print(packed_input,'packed_input')
        output, hn = self.rnn(packed_input)
        # print(hn.shape,'hn','output')
        hn = hn[-1] 
        y_pred = self.out_layer(hn)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data_sorted)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred





##################################################

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

class HeteroNetG(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(HeteroNetG, self).__init__() 
        self.layer1 = HeteroLayerG(in_size, hidden_size)
        self.layer2 = HeteroLayerG(hidden_size, out_size)

    def forward(self, G):
        # h_dict = {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
        G = self.layer1(G)
        h_dict = {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
        for ntype in h_dict:
            G.nodes[ntype].data['h'] = F.leaky_relu(h_dict[ntype])
        # h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G)
        return G

# use HeteroNetG
class static_heto_graph0(nn.Module):
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
        
        self.hconv = HeteroNetG(h_inp, h_dim, h_dim)
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = torch.zeros((bg.number_of_nodes('doc'), self.h_dim)).to(self.device)
        bg.nodes['word'].data['h'] = word_emb
        bg.nodes['topic'].data['h'] = topic_emb
        bg.nodes['doc'].data['h'] = doc_emb
         
        bg = self.hconv(bg)
        doc_emb = bg.nodes['doc'].data['h']
        doc_len = [g.num_nodes('doc') for g in g_list]
        doc_emb_split = torch.split(doc_emb, doc_len)
        # print(len(doc_emb_split),'doc_emb_split',doc_emb_split[0].shape)
        # padding to same size  
        # print(max(doc_len),'max(doc_len)')
        # embed_pad_tensor = torch.zeros(len(doc_len), max(doc_len), self.h_dim).to(self.device)
        mean_embed = torch.zeros(len(doc_len), self.h_dim).to(self.device)
        for i, embeds in enumerate(doc_emb_split): 
            mean_embed[i, :] = embeds.mean(0)
        # for i, embeds in enumerate(doc_emb_split): 
        #         embed_pad_tensor[i, torch.arange(0,len(embeds)), :] = embeds

        # doc_pool = embed_pad_tensor.mean(1)
        # doc_pool = self.maxpooling(embed_pad_tensor)
        # print(doc_pool.shape,'doc_pool')
        # doc_emb_mean = doc_emb.mean(0)
        y_pred = self.out_layer(mean_embed)
        # print(y_pred.shape,'y_pred',y_pred,y_data.shape,'y_data')
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

