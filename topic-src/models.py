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

class HeteroLayerCausalUni(nn.Module):
    def __init__(self, in_size, out_size, device):
        super().__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(in_size, out_size),
                'wt': nn.Linear(out_size, out_size),
                'wd': nn.Linear(out_size, out_size),
                'td': nn.Linear(out_size, out_size),
                'tt': nn.Linear(out_size, out_size),
            }) 
        self.device = device
        self.weight_causal = nn.Linear(out_size, out_size,bias=False)
        self.weight_noise = nn.Linear(out_size, out_size,bias=False)

        self.pos_cause =  nn.Parameter(torch.Tensor(1))
        self.neg_cause =  nn.Parameter(torch.Tensor(1))
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
        for srctype, etype, dsttype in [['word','ww','word']]: 
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        funcs = {}
        feat_dict['word'] = G.nodes['word'].data['h']
        # print(G.canonical_etypes)
        G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype == 'ww':
                continue
            # TODO if it is tt, wt, td consider causal and random de noise ... 
            # TODO update message propogation
            node_emb = feat_dict[srctype]
            if srctype == 'topic':
                v = G.nodes['topic'].data['effect']#.double()
                # causal_mask = torch.where(v > 0., 1., v)
                # causal_mask = torch.where(causal_mask < 0., -1., causal_mask)
                causal_mask = (v!=0)*1.0 #(v>0)+(v<0)*1
                # causal_mask = (v>0)*1.0*self.pos_cause + (v<0)*1.0*self.neg_cause 
                # causal_mask = v
                # t = (v == 0).nonzero().view(-1)
                random_mask = torch.bernoulli(torch.tensor([0.1]*len(causal_mask)).to(self.device)) * (causal_mask==0)#.view(-1, 1, -1)
                causal_mask = causal_mask.view(-1, 1)
                random_mask = random_mask.view(-1, 1)
                # print(causal_mask.shape,random_mask.shape,node_emb.shape)
                # v = torch.nonzero(v,as_tuple=True)
                # print('=====',node_emb * causal_mask)
                Wh = self.weight[etype](node_emb) + self.weight_causal(node_emb * causal_mask) - self.weight_noise(node_emb * random_mask)
                # print(self.weight_causal(node_emb * causal_mask)[:,0],'+++')
                # print(self.weight_noise(node_emb * random_mask)[:,0],'===')
            else:
                # print('srctype, etype, dsttype',srctype, etype, dsttype) 
                Wh = self.weight[etype](node_emb)
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroLayerCausalCus(nn.Module):
    def __init__(self, in_size, out_size, device):
        super().__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(in_size, out_size),
                'wt': nn.Linear(out_size, out_size),
                'wd': nn.Linear(out_size, out_size),
                'td': nn.Linear(out_size, out_size),
                'tt': nn.Linear(out_size, out_size),
                'td_cau': nn.Linear(out_size, out_size,bias=False),
                'td_noi': nn.Linear(out_size, out_size,bias=False),
                'tt_cau': nn.Linear(out_size, out_size,bias=False),
                'tt_noi': nn.Linear(out_size, out_size,bias=False),
            }) 
        self.device = device
        # self.weight_causal = nn.Linear(out_size, out_size,bias=False)
        # self.weight_noise = nn.Linear(out_size, out_size,bias=False)
        self.pos_cause =  nn.Parameter(torch.Tensor(1))
        self.neg_cause =  nn.Parameter(torch.Tensor(1))
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
        for srctype, etype, dsttype in [['word','ww','word']]: 
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        funcs = {}
        feat_dict['word'] = G.nodes['word'].data['h']
        # print(G.canonical_etypes)
        G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype == 'ww':
                continue
            # TODO if it is tt, wt, td consider causal and random de noise ... 
            # TODO update message propogation
            node_emb = feat_dict[srctype]
            if srctype == 'topic':
                v = G.nodes['topic'].data['effect'] 
                causal_mask = (v!=0)*1.0 #(v>0)+(v<0)*1
                # causal_mask = (v>0)*1.0*self.pos_cause + (v<0)*1.0*self.neg_cause 
                random_mask = torch.bernoulli(torch.tensor([0.1]*len(causal_mask)).to(self.device)) * (causal_mask==0)#.view(-1, 1, -1)
                causal_mask = causal_mask.view(-1, 1)
                random_mask = random_mask.view(-1, 1)
                Wh = self.weight[etype](node_emb) + self.weight['%s_cau' % etype](node_emb * causal_mask) - self.weight['%s_noi' % etype](node_emb * random_mask)
            else:
                # print('srctype, etype, dsttype',srctype, etype, dsttype) 
                Wh = self.weight[etype](node_emb)
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}



class __HeteroLayer2(nn.Module):
    def __init__(self, in_size, out_size):
        super(self).__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(in_size, out_size),
                'wt': nn.Linear(out_size, out_size),
                'wd': nn.Linear(out_size, out_size),
                'td': nn.Linear(out_size, out_size),
                'tt': nn.Linear(out_size, out_size),
            }) 

    def forward(self, G, feat_dict):
        funcs={}
        G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            # print('srctype, etype, dsttype',srctype, etype, dsttype) 
            # if srctype == 'word':
            #     feat_dict[srctype] = G.nodes[srctype].data['h']
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class HeteroNetCausalUni(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, device):
        super().__init__() 
        self.layer1 = HeteroLayerCausalUni(in_size, hidden_size, device)
        self.layer2 = HeteroLayerCausalUni(hidden_size, out_size, device)

    def forward(self, G, emb_dict):
        h_dict = self.layer1(G, emb_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        return h_dict

class HeteroNetCausalCus(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, device):
        super().__init__() 
        self.layer1 = HeteroLayerCausalCus(in_size, hidden_size, device)
        self.layer2 = HeteroLayerCausalCus(hidden_size, out_size, device)

    def forward(self, G, emb_dict):
        h_dict = self.layer1(G, emb_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        return h_dict

class HeteroLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(HeteroLayer, self).__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(in_size, out_size),
                'wt': nn.Linear(out_size, out_size),
                'wd': nn.Linear(out_size, out_size),
                'td': nn.Linear(out_size, out_size),
                'tt': nn.Linear(out_size, out_size),
            }) 

    def forward(self, G, feat_dict):
        # print(G,feat_dict,'G,feat_dict')
        funcs={}
        for srctype, etype, dsttype in [['word','ww','word']]: 
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        funcs = {}
        feat_dict['word'] = G.nodes['word'].data['h']
        # print(G.canonical_etypes)
        G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype == 'ww':
                continue
            # print('srctype, etype, dsttype',srctype, etype, dsttype) 
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(HeteroNet, self).__init__() 
        self.layer1 = HeteroLayer(in_size, hidden_size)
        self.layer2 = HeteroLayer(hidden_size, out_size)

    def forward(self, G, emb_dict):
        h_dict = self.layer1(G, emb_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        return h_dict

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
        G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            # print('srctype, etype, dsttype',srctype, etype, dsttype,feat_dict[srctype].shape) 
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class HeteroConvNet(nn.Module):
    def __init__(self, word_in_size, topic_in_size, hidden_size, out_size):
        super().__init__() 
        self.layer1 = HeteroConvLayer(word_in_size, topic_in_size, hidden_size)
        self.layer2 = HeteroConvLayer(hidden_size, hidden_size, out_size)

    def forward(self, G, emb_dict):
        # print('1')
        h_dict = self.layer1(G, emb_dict)
        # print('relu')
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        # print('2')
        h_dict = self.layer2(G, h_dict)
        # print('done')
        return h_dict

 
class HeteroConvLayerCausalCus(nn.Module):
    def __init__(self, word_in_size, topic_in_size, out_size):
        super().__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(word_in_size, out_size),
                'wt': nn.Linear(word_in_size, out_size),
                'wd': nn.Linear(word_in_size, out_size),
                'td': nn.Linear(topic_in_size, out_size),
                'tt': nn.Linear(topic_in_size, out_size),
                'td_cau': nn.Linear(topic_in_size, out_size,bias=True),
                'td_noi': nn.Linear(topic_in_size, out_size,bias=True),
                'tt_cau': nn.Linear(topic_in_size, out_size,bias=True),
                'tt_noi': nn.Linear(topic_in_size, out_size,bias=True),
            }) 

    def forward(self, G, feat_dict):
        # print(G,feat_dict,'G,feat_dict')
        funcs={}
        G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            node_emb = feat_dict[srctype]
            if srctype == 'topic':
                v = G.nodes['topic'].data['effect'] 
                causal_mask = (v!=0)*1.0 #(v>0)+(v<0)*1
                # causal_mask = (v>0)*1.0*self.pos_cause + (v<0)*1.0*self.neg_cause 
                random_mask = torch.bernoulli(torch.tensor([0.1]*len(causal_mask)).to(self.device)) * (causal_mask==0)#.view(-1, 1, -1)
                causal_mask = causal_mask.view(-1, 1)
                random_mask = random_mask.view(-1, 1)
                Wh = self.weight[etype](node_emb) + self.weight['%s_cau' % etype](node_emb * causal_mask) - self.weight['%s_noi' % etype](node_emb * random_mask)
            else:
                # print('srctype, etype, dsttype',srctype, etype, dsttype) 
                Wh = self.weight[etype](node_emb)

            # print('srctype, etype, dsttype',srctype, etype, dsttype,feat_dict[srctype].shape) 
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroConvNetCausalCus(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, device):
        super().__init__() 
        self.layer1 = HeteroConvLayerCausalCus(in_size, hidden_size, device)
        self.layer2 = HeteroConvLayerCausalCus(hidden_size, out_size, device)

    def forward(self, G, emb_dict):
        h_dict = self.layer1(G, emb_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        return h_dict

class WordGraphNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(WordGraphNet, self).__init__() 
        self.layer1 = WordGraphLayer(in_size, hidden_size)
        self.layer2 = WordGraphLayer(hidden_size, out_size)

    def forward(self, G, emb_dict):
        h_dict = self.layer1(G, emb_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        return h_dict


class WordGraphLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(WordGraphLayer, self).__init__()
        self.weight = nn.ModuleDict({
                'ww': nn.Linear(in_size, out_size),
            }) 
    def forward(self, G, feat_dict):
        funcs={}
        for srctype, etype, dsttype in [['word','ww','word']]: 
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        return {'word' : G.nodes['word'].data['h']}

# https://www.jianshu.com/p/767950b560c4

# a static graph model
 
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
        
        self.hconv = HeteroNet(h_inp, h_dim, h_dim)
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
        emb_dict = {
            'word':word_emb,
            'topic':topic_emb,
            'doc':doc_emb
        }
        emb_dict = self.hconv(bg,emb_dict)
        doc_emb = emb_dict['doc'] 
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


class static_heto_graph2(nn.Module):
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
        self.hconv = HeteroConvNet(h_inp, h_dim, h_dim, h_dim)
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
        emb_dict = {
            'word':word_emb,
            'topic':topic_emb,
            'doc':doc_emb
        }
        emb_dict = self.hconv(bg,emb_dict)
        doc_emb = emb_dict['doc'] 
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

class static_heto_graph_causal_uni(nn.Module):
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
        
        self.hconv = HeteroNetCausalUni(h_inp, h_dim, h_dim, self.device)
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
        # dgl.add_self_loop(bg, etype='ww')
        # dgl.add_self_loop(bg, etype='wt')
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = torch.zeros((bg.number_of_nodes('doc'), self.h_dim)).to(self.device)
        emb_dict = {
            'word':word_emb,
            'topic':topic_emb,
            'doc':doc_emb
        }
        # print(topic_emb,'++++++++')
        emb_dict = self.hconv(bg,emb_dict)
        doc_emb = emb_dict['doc'] 
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


class static_heto_graph_causal_cus(nn.Module):
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
        
        self.hconv = HeteroNetCausalCus(h_inp, h_dim, h_dim, self.device)
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
        # dgl.add_self_loop(bg, etype='ww')
        # dgl.add_self_loop(bg, etype='wt')
        word_emb = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id']].view(-1, self.topic_embeds.shape[1])
        doc_emb = torch.zeros((bg.number_of_nodes('doc'), self.h_dim)).to(self.device)
        emb_dict = {
            'word':word_emb,
            'topic':topic_emb,
            'doc':doc_emb
        }
        # print(topic_emb,'++++++++')
        emb_dict = self.hconv(bg,emb_dict)
        doc_emb = emb_dict['doc'] 
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

class static_word_graph(nn.Module):
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
        self.hconv = WordGraphNet(h_inp, h_dim, h_dim) 
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
        emb_dict = self.hconv(bg, {'word':word_emb})
        word_emb = emb_dict['word']
        word_len = [g.num_nodes('word') for g in g_list]
        word_emb_split = torch.split(word_emb, word_len)
        # print(len(doc_emb_split),'doc_emb_split',doc_emb_split[0].shape)
        # padding to same size  
        mean_embed = torch.zeros(len(word_len), self.h_dim).to(self.device)
        for i, embeds in enumerate(word_emb_split): 
            mean_embed[i, :] = embeds.mean(0)
        # embed_pad_tensor = torch.zeros(len(word_len), max(word_len), self.h_dim).to(self.device)
        # for i, embeds in enumerate(word_emb_split): 
        #         embed_pad_tensor[i, torch.arange(0,len(embeds)), :] = embeds
        # print(embed_pad_tensor.shape,'embed_pad_tensor') # batch,max # doc, f 
        # word_pool = embed_pad_tensor.mean(1)
        y_pred = self.out_layer(mean_embed)
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred) 
        return loss, y_pred

 

# a temporal graph model

