import  time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
# from dgl import DGLGraph

# https://github.com/dmlc/dgl/blob/ddc2faa547da03e0b791648677ed06ce1daf3e0d/examples/pytorch/gcn/gcn_spmv.py

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


class LevelHeteroConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout):
        super().__init__()
        
        self.weight = nn.ModuleDict({
                # 'ww': nn.Linear(word_in_size, out_size),
                'wt': nn.Linear(in_feats, out_feats),
                'wd': nn.Linear(in_feats, out_feats),
                'td': nn.Linear(in_feats, out_feats),
                # 'tt': nn.Linear(topic_in_size, out_size),
            }) 
        self.activation = activation
        self.drop = nn.Dropout(dropout)

    def forward(self, G, feat_dict):
        for ntype in feat_dict:
            G.nodes[ntype].data['h'] = feat_dict[ntype]
        # print(G,feat_dict,'G,feat_dict')
        funcs={}
        # G.edges['tt'].data['weight'] = G.edges['tt'].data['weight'].float()
        G.edges['wd'].data['weight'] = G.edges['wd'].data['weight'].float()
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype in ['ww','tt']:
                continue
            # print('srctype, etype, dsttype',srctype, etype, dsttype,feat_dict[srctype].shape) 
            Wh = self.weight[etype](G.nodes[srctype].data['h'])   #   feat_dict[srctype]
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # print(etype,G.edges[etype].data['weight'].dtype,Wh.dtype)
            funcs[etype] = (fn.u_mul_e('Wh_%s' % etype, 'weight', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')

        for ntype in G.ntypes:
            if ntype != 'doc':
                continue
            # n_id = node_dict[ntype]
            # print(self.skip[ntype],'self.skip[ntype]')
            feat = G.nodes[ntype].data['h']
            G.nodes[ntype].data['h'] = self.drop(self.activation(feat))

        # return G.nodes['doc'].data['h']
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
 
class GCNHet(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, activation, device, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(GCNHet, self).__init__()
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

        # self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        # node_dict = {'doc': 0, 'topic': 1, 'word': 2}
        # edge_dict = {'td': 0, 'tt': 1, 'wd': 2, 'wt': 3, 'ww': 4}
        self.gcn_word_layers = nn.ModuleList()
        self.gcn_word_layers.append(GCNLayer(n_inp, n_hid, activation, dropout))
        for _ in range(n_layers-1):
            self.gcn_word_layers.append(GCNLayer(n_hid, n_hid, activation, dropout))
        self.gcn_topic_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gcn_topic_layers.append(GCNLayer(n_hid, n_hid, activation, dropout))
        self.hetero_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.hetero_layers.append(LevelHeteroConvLayer(n_hid, n_hid, activation, dropout))
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
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        # torch.zeros((bg.number_of_nodes('doc'), self.n_hid)).to(self.device)
        # for i in range(len(self.gcn_topic_layers)):
        # print('word_emb',word_emb.shape,topic_emb.shape,'doc_emb',doc_emb.shape)

        for layer in self.gcn_word_layers:
            word_emb = layer(bg, word_emb, 'word','ww')

        for layer in self.gcn_topic_layers:
            topic_emb = layer(bg, topic_emb, 'topic','tt')
        # print('word_emb',word_emb.shape,topic_emb.shape)
        feat_dict = {'word':word_emb, 'topic':topic_emb, 'doc':doc_emb}
        for layer in self.hetero_layers:
            feat_dict = layer(bg, feat_dict)

        # bg.nodes['word'].data['h'] = torch.tanh(self.adapt_ws(word_emb))
        # bg.nodes['word'].data['h'] = self.adapt_ws(word_emb)
        # bg.nodes['topic'].data['h'] = topic_emb
        # bg.nodes['doc'].data['h'] = doc_emb
        # for i in range(self.n_layers):
        #     self.gcs[i](bg, 'h', 'h')
 
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
