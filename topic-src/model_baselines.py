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
        # h = h * g.nodes[ntype].data['norm'].unsqueeze(1)
        g.nodes[ntype].data['h'] = h
        g.update_all(
            # fn.copy_src(src='h', out='m'),
                        fn.u_mul_e('h', 'weight', 'm'),
                        fn.sum(msg='m', out='h'),etype=etype)
        h = g.nodes[ntype].data.pop('h')
        # normalization by square root of dst degree
        # h = h * g.nodes[ntype].data['norm'].unsqueeze(1)
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



class GCNLayerM(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayerM, self).__init__()
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
        # h = h * g.nodes[ntype].data['norm'].unsqueeze(1)
        g.nodes[ntype].data['h'] = h
        g.update_all(
            # fn.copy_src(src='h', out='m'),
                        fn.u_mul_e('h', 'weight', 'm'),
                        fn.sum(msg='m', out='h'),etype=etype)
        h = g.nodes[ntype].data.pop('h')
        # normalization by square root of dst degree
        # h = h * g.nodes[ntype].data['norm'].unsqueeze(1)
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        # print(h)
        return h

    def __repr__(self):
        return '{}(in_dim={}, out_dim={})'.format(
            self.__class__.__name__, self.in_feats, self.out_feats)


class EvolveGCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(EvolveGCNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        # self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        # if bias:
        #     self.bias = nn.Parameter(torch.Tensor(out_feats))
        # else:
        #     self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h, ntype, etype, weight):
        h = torch.mm(h, weight)
        # normalization by square root of src degree
        # h = h * g.nodes[ntype].data['norm'].unsqueeze(1)
        g.nodes[ntype].data['h'] = h
        g.update_all(
            # fn.copy_src(src='h', out='m'),
                        fn.u_mul_e('h', 'weight', 'm'),
                        fn.sum(msg='m', out='h'),etype=etype)
        h = g.nodes[ntype].data.pop('h')
        # normalization by square root of dst degree
        # h = h * g.nodes[ntype].data['norm'].unsqueeze(1)
        # bias
        # if self.bias is not None:
        #     h = h + self.bias
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        # print(h)
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


 
class HGTLayer(nn.Module):
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
        self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        self.skip = nn.ParameterDict() 
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            self.skip[t] = nn.Parameter(torch.ones(1))
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
            alpha = torch.sigmoid(self.skip[ntype])
            trans_out = self.a_linears[ntype](G.nodes[ntype].data.pop('t'))
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

class TempHGTLayer(nn.Module):
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
        self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        self.skip = nn.ParameterDict() 
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            self.skip[t] = nn.Parameter(torch.ones(1))
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
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        if 'timeh' in edges.data:
            edges.data['v'] += edges.data['timeh'].unsqueeze(1)
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': F.relu(h.view(-1, self.out_dim))}
        
    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
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
            alpha = torch.sigmoid(self.skip[ntype])
            trans_out = self.a_linears[ntype](G.nodes[ntype].data.pop('t'))
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

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

class HGT(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, device, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super().__init__()
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
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
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
        # torch.zeros((bg.number_of_nodes('doc'), self.n_hid)).to(self.device)
        bg.nodes['word'].data['h'] = self.adapt_ws(word_emb)
        bg.nodes['topic'].data['h'] = topic_emb
        bg.nodes['doc'].data['h'] = doc_emb
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

class TempHGT(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1,n_hid))
        self.time_emb = RelTemporalEncoding(n_hid//n_heads,seq_len)
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(TempHGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
        self.out_layer =  nn.Linear(n_hid*3, 1) 
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
        # torch.zeros((bg.number_of_nodes('doc'), self.n_hid)).to(self.device)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h'] = word_emb
        bg.nodes['topic'].data['h'] = topic_emb
        bg.nodes['doc'].data['h'] = doc_emb
        # bg.edges['ww'].data['time'] = torch.zeros(bg.edges['ww'].data['weight'].shape).int()
        # print('time',bg.edges['ww'].data['time'].shape)
        ww_time = self.time_emb(bg.edges['ww'].data['time'].long())
        bg.edges['ww'].data['timeh'] = ww_time
        wd_time = self.time_emb(bg.edges['wd'].data['time'].long())
        # print(bg.edges['ww'].data['timeh'].shape,'timeh',bg.edges['ww'].data['time'])
        bg.edges['wd'].data['timeh'] = wd_time
        bg.edges['dw'].data['timeh'] = wd_time
        wt_time = self.time_emb(bg.edges['wt'].data['time'].long())
        bg.edges['wt'].data['timeh'] = wt_time
        bg.edges['tw'].data['timeh'] = wt_time
        td_time = self.time_emb(bg.edges['td'].data['time'].long())
        bg.edges['td'].data['timeh'] = self.time_emb(bg.edges['td'].data['time'].long())
        bg.edges['dt'].data['timeh'] = td_time
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
        y_pred = self.out_layer(global_info)
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

# dynamic gcn
# only use words, keep nodes
class dyngcn(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max'):
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
         
        self.adapt_ws  = nn.Linear(n_inp,  n_hid) 
        self.temp_encoding = nn.Linear(n_hid*2,  n_hid)
        self.bn = nn.BatchNorm1d(2*n_hid)
        # input layer
        self.layers = nn.ModuleList()
        # self.temp_encoding = nn.ModuleList()
        # self.bn = nn.ModuleList()
        # for i in range(seq_len):
        #     self.layers.append(GCNLayerM(n_hid, n_hid, activation, dropout))
        #     self.temp_encoding.append(nn.Linear(n_hid*2,  n_hid))
        #     self.bn.append(nn.BatchNorm1d(2*n_hid))
        self.layers.append(GCNLayerM(n_hid, n_hid, activation, dropout))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayerM(n_hid, n_hid, activation, dropout))
        
        self.out_layer = nn.Linear(n_hid, 1) 
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
        # topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        # doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        bg.nodes['word'].data['h'] = word_emb
        # for curr_time in range(self.seq_len):
        for curr_time in range(0,self.seq_len):
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        # ('topic', 'tt', 'topic'): tt_edges_idx,
                                        # ('word', 'wt', 'topic'): wt_edges_idx,
                                        # ('topic', 'td', 'doc'): td_edges_idx,
                                        # ('word', 'wd', 'doc'):wd_edges_idx,
                                        # ('topic', 'tw', 'word'): wt_edges_idx,
                                        # ('doc', 'dt', 'topic'): td_edges_idx,
                                        # ('doc', 'dw', 'word'):wd_edges_idx
                                        }#,preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            # sub_bg.time_emb = time_emb
            # print('curr_time=',curr_time,sub_bg.nodes['word'].data['h0'].shape)
            # if curr_time == 0:
            #     h = sub_bg.nodes['word'].data['h0']
            # else:
            h = sub_bg.nodes['word'].data['h']
            h0 = sub_bg.nodes['word'].data['h0']
            cat_h = torch.cat((h,h0),dim=-1)
            
            cat_h = self.dropout(self.bn(cat_h))
            # cat_h = self.bn(cat_h)
            h = torch.tanh(self.temp_encoding(self.bn(cat_h)))
            # h = torch.relu(self.temp_encoding(cat_h))+h
            # h = self.layers[curr_time](sub_bg, h, ntype='word',etype='ww') 
            for layer in self.layers:
                h = layer(sub_bg, h, ntype='word',etype='ww') 
            bg.nodes['word'].data['h'][orig_node_ids['word'].long()] = h
            
        if self.pool == 'max':
            global_word_info = dgl.max_nodes(bg, feat='h',ntype='word')
        elif self.pool == 'mean':
            global_word_info = dgl.mean_nodes(bg, feat='h',ntype='word')
        y_pred = self.out_layer(global_word_info)
        loss = self.criterion(y_pred.view(-1), y_data) 
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

# EvolveGCN
# only use words, keep nodes
class EvolveGCN(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max'):
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
         
        self.adapt_ws  = nn.Linear(n_inp,  n_hid) 
        # self.temp_encoding = nn.Linear(n_hid*2,  n_hid)
        # self.bn = nn.BatchNorm1d(2*n_hid)
        # input layer
        self.layers = nn.ModuleList()
        # self.weights = nn.ModuleList()
        self.lstmCell = nn.ModuleList()
        self.weights = nn.Parameter(torch.Tensor(self.n_layers,n_hid, n_hid))
        for i in range(self.n_layers):
            self.layers.append(EvolveGCNLayer(n_hid, n_hid, activation, dropout))
            # self.weights.append(nn.Parameter(torch.Tensor(n_hid, n_hid)))
            self.lstmCell.append(nn.LSTMCell(n_hid, n_hid))
        # self.layers = EvolveGCNLayer(n_hid, n_hid, activation, dropout)
        # self.weights = nn.Parameter(torch.Tensor(n_hid, n_hid)) 
        # self.lstmCell = nn.LSTMCell(n_hid, n_hid)
        self.out_layer = nn.Linear(n_hid, 1) 
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
        # topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        # doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        bg.nodes['word'].data['h'] = word_emb
        hx_list = self.weights
        # self.weights
        # for curr_time in range(self.seq_len):
        for curr_time in range(0,self.seq_len):
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            # td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        # ('topic', 'tt', 'topic'): tt_edges_idx,
                                        # ('word', 'wt', 'topic'): wt_edges_idx,
                                        # ('topic', 'td', 'doc'): td_edges_idx,
                                        # ('word', 'wd', 'doc'):wd_edges_idx,
                                        # ('topic', 'tw', 'word'): wt_edges_idx,
                                        # ('doc', 'dt', 'topic'): td_edges_idx,
                                        # ('doc', 'dw', 'word'):wd_edges_idx
                                        }#,preserve_nodes=True
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] # {'word':,'topic':,'doc':}
            # time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            # sub_bg.time_emb = time_emb
            # print('curr_time=',curr_time,sub_bg.nodes['word'].data['h0'].shape)
            # if curr_time == 0:
            #     h = sub_bg.nodes['word'].data['h0']
            # else:
            h = sub_bg.nodes['word'].data['h']
            # h0 = sub_bg.nodes['word'].data['h0']
            # cat_h = torch.cat((h,h0),dim=-1)
            
            # cat_h = self.dropout(self.bn(cat_h))
            # cat_h = self.bn(cat_h)
            # h = torch.tanh(self.temp_encoding(self.bn(cat_h)))
            # h = torch.relu(self.temp_encoding(cat_h))+h
            # h = self.layers[curr_time](sub_bg, h, ntype='word',etype='ww') 
            new_hx_list = []
            cx_list = []
            for i in range(len(self.layers)):
                hx = hx_list[i]
                if curr_time == 0:
                    hx, cx = self.lstmCell[i](hx)
                else:
                    cx = cx_list[i]
                    hx, cx = self.lstmCell[i](hx, (hx, cx))
                new_hx_list.append(hx)
                cx_list.append(cx)
                h = self.layers[i](sub_bg, h, 'word','ww', hx)
            # obtain weights
            # if curr_time == 0:
            #     hx, cx = self.lstmCell(hx)
            # else:
            #     hx, cx = self.lstmCell(hx, (hx, cx))
            # h = self.layers(sub_bg, h, 'word','ww', hx)
            # for layer in self.layers:
            #     h = layer(sub_bg, h, ntype='word',etype='ww') 
            bg.nodes['word'].data['h'][orig_node_ids['word'].long()] = h
            
        if self.pool == 'max':
            global_word_info = dgl.max_nodes(bg, feat='h',ntype='word')
        elif self.pool == 'mean':
            global_word_info = dgl.mean_nodes(bg, feat='h',ntype='word')
        y_pred = self.out_layer(global_word_info)
        loss = self.criterion(y_pred.view(-1), y_data) 
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred
