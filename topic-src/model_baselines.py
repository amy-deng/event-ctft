import  time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
# from dgl import DGLGraph
from dgl.nn import GATConv

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
        g.update_all(
            # fn.copy_src(src='h', out='m'),
                        fn.u_mul_e('h', 'weight', 'm'),
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

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, device, seq_len=7, vocab_size=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.in_feats = in_feats
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.device = device
        self.pool = pool
        # self.dropout = nn.Dropout(dropout)
        self.word_embeds = None 
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_feats, n_hidden, activation, dropout))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
        # output layer
        self.out_layer = nn.Linear(n_hidden, 1) 
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
        h = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])

        for layer in self.layers:
            h = layer(bg, h, ntype='word',etype='ww')

        bg.nodes['word'].data['h'] = h
        if self.pool == 'max':
            global_word_info = dgl.max_nodes(bg, feat='h',ntype='word')
        elif self.pool == 'mean':
            global_word_info = dgl.mean_nodes(bg, feat='h',ntype='word')
        y_pred = self.out_layer(global_word_info)
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred) 
        return loss, y_pred


 
class GATNet(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GATNet, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation,allow_zero_in_degree=True))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation,allow_zero_in_degree=True))
        # output projection
        # self.gat_layers.append(GATConv(
        #     num_hidden * heads[-2], num_classes, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        return h
        # output projection
        # logits = self.gat_layers[-1](self.g, h).mean(1)
        # return logits

class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, heads, activation, device, seq_len=7, vocab_size=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.in_feats = in_feats
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.device = device
        self.pool = pool
        self.word_embeds = None 
        self.layers = nn.ModuleList()
        heads_list = [heads for i in range(n_layers)]
        self.conv = GATNet(n_layers,in_feats, n_hidden, heads_list,activation,dropout,dropout,0.2,False)
        self.out_layer = nn.Linear(n_hidden*heads, 1) 
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
        h = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        # h_sub_g = dgl.metapath_reachable_graph(bg, ('ww',))
        # print(time.time()-time1,'1')
        sub_g = dgl.edge_type_subgraph(bg, [('word', 'ww', 'word')])
        h_sub_g = dgl.to_homogeneous(sub_g)
        h = self.conv(h_sub_g, h)
        bg.nodes['word'].data['h'] = h
        if self.pool == 'max':
            global_word_info = dgl.max_nodes(bg, feat='h',ntype='word')
        elif self.pool == 'mean':
            global_word_info = dgl.mean_nodes(bg, feat='h',ntype='word')
        y_pred = self.out_layer(global_word_info)
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred) 
        return loss, y_pred



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
        # G.edges['wd'].data['weight'] = G.edges['wd'].data['weight'].float()
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


class RGCN(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, activation, device, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(RGCN, self).__init__()
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
            self.hetero_layers.append(HeteroRGCNLayer(n_hid, n_hid, activation, dropout, ['ww','wt','tt','wd','td','tw','dw','dt'],['word','topic','doc']))
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
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        feat_dict = {'word':word_emb, 'topic':topic_emb, 'doc':doc_emb}
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
        global_info = torch.cat((global_doc_info, global_word_info, global_topic_info),-1)
        y_pred = self.out_layer(global_info)
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

 