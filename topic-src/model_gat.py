import  time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
import time
# from dgl import DGLGraph 
from dgl.nn import GATConv
# https://github.com/dmlc/dgl/blob/ddc2faa547da03e0b791648677ed06ce1daf3e0d/examples/pytorch/gcn/gcn_spmv.py


 
 
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