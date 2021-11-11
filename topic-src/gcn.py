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

    def forward(self, g, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * g.nodes['word'].data['norm'].unsqueeze(1)
        g.nodes['word'].data['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'),etype='ww')
        h = g.nodes['word'].data.pop('h')
        # normalization by square root of dst degree
        h = h * g.nodes['word'].data['norm'].unsqueeze(1)
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

class GCN_0(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNLayer(n_hidden, n_classes, None, dropout))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h


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
        self.layers.append(GCNLayer(in_feats, n_hidden, activation, 0.))
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
            h = layer(bg, h)

        bg.nodes['word'].data['h'] = h
        if self.pool == 'max':
            global_word_info = dgl.max_nodes(bg, feat='h',ntype='word')
        elif self.pool == 'mean':
            global_word_info = dgl.mean_nodes(bg, feat='h',ntype='word')
        y_pred = self.out_layer(global_word_info)
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred) 
        return loss, y_pred