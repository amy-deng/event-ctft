import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import torch
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
import time
from layers import *
from utils import *
# from sparsemax import Sparsemax
# from tcn import *
try:
    import dgl
except:
    print("<<< dgl are not imported >>>")
    pass


class GCN(nn.Module):
    def __init__(self, in_feat, hid_feat, dropout=0.2, n_layer=2, n_C=None):
        super().__init__()
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.n_layer = n_layer
        d_bn = hid_feat
        if n_C:
            d_bn = n_C
        self.conv_layers.append(GraphConvLayer(in_feat, hid_feat))
        self.bn_layers.append(nn.BatchNorm1d(d_bn))
        for i in range(1, n_layer):
            self.conv_layers.append(GraphConvLayer(hid_feat, hid_feat))
            if i < n_layer - 1:
                self.bn_layers.append(nn.BatchNorm1d(d_bn))

    def forward(self, x, adj):
        for i in range(self.n_layer):
            x = self.conv_layers[i](x, adj)
            if i < self.n_layer -1:
                x = F.relu(self.bn_layers[i](x))
                x = F.dropout(x, self.dropout, training=self.training)
        return x
 
class GCN_dndc(nn.Module):
    def __init__(self, m, in_feat, hid_feat, dropout=0.2, n_layer=2):
        super().__init__()
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.n_layer = n_layer
        d_bn = m
        self.conv_layers.append(GraphConvLayer(in_feat, hid_feat))
        self.bn_layers.append(nn.BatchNorm1d(d_bn))
        for i in range(1, n_layer):
            self.conv_layers.append(GraphConvLayer(hid_feat, hid_feat))
            if i < n_layer - 1:
                self.bn_layers.append(nn.BatchNorm1d(d_bn))

    def forward(self, x, adj):
        for i in range(self.n_layer):
            # print(x.shape,adj.shape,' in gcn ====')
            x = self.conv_layers[i](x, adj)
            if i < self.n_layer -1:
                x = F.relu(self.bn_layers[i](x))
                x = F.dropout(x, self.dropout, training=self.training)
        return x
 
class SpGCN(nn.Module):
    def __init__(self, in_feat, hid_feat, n_layer=2, n_C=None, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.n_layer = n_layer
        d_bn = hid_feat
        if n_C:
            d_bn = n_C
        self.conv_layers.append(SpGraphConvLayer(in_feat, hid_feat))
        self.bn_layers.append(nn.BatchNorm1d(d_bn))
        for i in range(1, n_layer):
            self.conv_layers.append(SpGraphConvLayer(hid_feat, hid_feat))
            if i < n_layer - 1:
                self.bn_layers.append(nn.BatchNorm1d(d_bn))
        # self.conv1 = SpGraphConvLayer(in_feat, in_feat)
        # self.conv1_bn = nn.BatchNorm1d(in_feat)
        # self.conv2 = SpGraphConvLayer(in_feat, out_feat)
        # self.conv2_bn = nn.BatchNorm1d(out_feat)

    def forward(self, x, adj):
        # print(x.shape)
        for i in range(self.n_layer):
            x = self.conv_layers[i](x, adj)
            if i < self.n_layer -1:
                x = F.relu(self.bn_layers[i](x))
                x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.conv1_bn(self.conv1(x, adj)))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.conv2_bn(self.conv2(x, adj)))
        # x = F.relu(self.conv1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.conv2(x, adj)
        return x
    
class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, nheads, alpha=0.2):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads[0])]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_atts = [SpGraphAttentionLayer(nhid * nheads[0], 
                                             nhid, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)  for _ in range(nheads[1])]

        for i, out_att in enumerate(self.out_atts):
            self.add_module('out_att_{}'.format(i), out_att)
        self.conv1_bn = nn.BatchNorm1d(nhid)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.stack([out_att(x, adj) for out_att in self.out_atts], dim=2)
        x = torch.mean(x, dim=2)
        return F.elu(self.conv1_bn(x))
        # return F.elu(x)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, nheads, alpha=0.2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads[0])]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.out_atts = [GraphAttentionLayer(nhid * nheads[0], 
                                             nhid, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)  for _ in range(nheads[1])]

        for i, out_att in enumerate(self.out_atts):
            self.add_module('out_att_{}'.format(i), out_att)

        self.conv1_bn = nn.BatchNorm1d(nhid)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        print(x.shape,'x')
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.stack([out_att(x, adj) for out_att in self.out_atts], dim=2)
        x = torch.mean(x, dim=2)
        x = self.conv1_bn(x)
        return x

        
 
 
class LR(nn.Module): 
    def __init__(self, args, data):
        super().__init__() 
        self.linear = nn.Linear(data.f*args.window, 1) # output potential outcome
        # self.linear1 = nn.Linear(in_feat, 1)
        self.binary = (not args.realy)
        if self.binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
  
    def forward(self, X, A, C, Y, Cc, G, E, idx_list=None,epoch=0):
        # print(X.shape,Y.shape)
        b,l,m,_ = X.size()
        X = X.permute(0,2,1,3).contiguous().view(b*m,-1)
        Y = Y[:,-1].contiguous().view(-1)
        y_pred = self.linear(X).view(-1)

        # print(X.shape,Y.shape,y_pred.shape)
        loss = self.criterion(y_pred,Y)

        # exit()
        # X = X.view(-1, X.size(-1))
        # Y = Y.view(-1)
        # X = torch.cat((X,C.view(-1,1)),-1)
        # # y0 = self.linear0(X).view(-1)
        # # y1 = self.linear1(X).view(-1)
        # yy = self.linear(X).view(-1,2)
        # y0 = yy[:,0]
        # y1 = yy[:,1]
        # # print(X.shape,'X',y0.shape, Y.shape)
        # y = torch.where(C.view(-1) > 0, y1, y0)
        # loss = self.criterion(y, Y)
        if self.binary:
            y = torch.sigmoid(y_pred)
        return loss, y, y, y 

class RNN(nn.Module): 
    def __init__(self, args, data):
        super().__init__() 
        self.rnn = nn.GRU(data.f,data.f,batch_first=True)
        self.linear = nn.Linear(data.f, 1)
        self.binary = (not args.realy)
        if self.binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
  
    def forward(self, X, A, C, Y, Cc, G, E, idx_list=None,epoch=0):
        # print(X.shape,Y.shape)
        b,l,m,f = X.size()
        X = X.permute(0,2,1,3).contiguous().view(b*m,l,f)
        output, h_n = self.rnn(X)
        # output, (h_n, cn) = self.rnn(X)
        Y = Y[:,-1].contiguous().view(-1)
        y_pred = self.linear(h_n.squeeze(0)).view(-1)
        loss = self.criterion(y_pred,Y)
        if self.binary:
            y = torch.sigmoid(y_pred)
        return loss, y, y, y 



class OLS1(nn.Module): 
    def __init__(self,in_feat,binary=True,device=torch.device('cpu')): 
        super().__init__() 
        self.linear = nn.Linear(in_feat+1, 1) # output potential outcome
        self.binary = binary
        self.device = device
        if binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
  
    def forward(self, X, C, Y): 
        X = X.view(-1, X.size(-1))
        Y = Y.view(-1)
        C = C.view(-1,1) 
        C0 = torch.zeros(C.shape).to(self.device)
        X0 = torch.cat((X,C0),-1)
        y0 = self.linear(X0).view(-1)
        C1 = torch.zeros(C.shape).to(self.device)
        X1 = torch.cat((X,C1),-1)
        y1 = self.linear(X1).view(-1)
        y = torch.where(C.view(-1) > 0, y1, y0)
        loss = self.criterion(y, Y)
        if self.binary:
            y = torch.sigmoid(y)
            y0 = torch.sigmoid(y0)
            y1 = torch.sigmoid(y1)
        return loss, y, y0, y1 

class OLS2(nn.Module): 
    def __init__(self,in_feat,binary=True,device=torch.device('cpu')): 
        super().__init__() 
        self.linear_t0 = nn.Linear(in_feat, 1) # output potential outcome
        self.linear_t1 = nn.Linear(in_feat, 1)
        self.device = device
        self.binary = binary
        if binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
  
    def forward(self, X, C, Y): 
        X = X.view(-1, X.size(-1))
        Y = Y.view(-1)
        y0 = self.linear_t0(X).view(-1) # update 20210224
        y1 = self.linear_t1(X).view(-1)
        y = torch.where(C.view(-1) > 0, y1, y0)
        loss = self.criterion(y, Y)
        if self.binary:
            y = torch.sigmoid(y)
            y0 = torch.sigmoid(y0)
            y1 = torch.sigmoid(y1)
        return loss, y, y0, y1#, Y

class TARNet(nn.Module): 
    def __init__(self, in_feat, rep_hid, hyp_hid, rep_layer=2, hyp_layer=2, binary=True, p=0.5, dropout=0.2, device=torch.device('cpu')): 
        super().__init__() 
        self.p = p
        self.device = device
        self.hyp_layer = hyp_layer
        self.rep_layer_fst = nn.Linear(in_feat, rep_hid)
        self.rep_bn_fst = nn.BatchNorm1d(rep_hid)
    
        self.rep_layers = nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(rep_layer-1)])
        self.rep_bns = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(rep_layer-1)])

        self.hyp_layer_fst0 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst0 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-1)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-1)])
        self.hyp_out0 = nn.Linear(rep_hid, 1) 

        self.hyp_layer_fst1 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst1 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers1= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-1)])
            self.hyp_bns1 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-1)])
        self.hyp_out1 = nn.Linear(rep_hid, 1) 
        self.dropout = nn.Dropout(p=dropout)
        self.binary = binary
        if binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
  
    def forward(self, X, C, Y): 
        X = X.view(-1, X.size(-1))
        Y = Y.view(-1)
        h = self.dropout(F.relu(self.rep_bn_fst(self.rep_layer_fst(X))))
        for fc, bn in zip(self.rep_layers, self.rep_bns):
            h = self.dropout(F.relu(bn(fc(h))))

        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))

        h1 = self.dropout(F.relu(self.hyp_bn_fst1(self.hyp_layer_fst1(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers1, self.hyp_bns1):
                h1 = self.dropout(F.relu(bn(fc(h1))))

        y0 = self.hyp_out0(h0).view(-1)
        y1 = self.hyp_out1(h1).view(-1)
        C_1d = C.view(-1)
        y = torch.where(C_1d > 0, y1, y0)
        loss = self.criterion(y, Y, reduction='none')
        weight = C_1d/(2*self.p) + (1-C_1d)/(2*(1-self.p))
        loss = torch.mean(loss * weight)
        if self.binary:
            y = torch.sigmoid(y)
            y0 = torch.sigmoid(y0)
            y1 = torch.sigmoid(y1)
        return loss, y, y0, y1 

class CFR_MMD(nn.Module): 
    def __init__(self, in_feat, rep_hid, hyp_hid, rep_layer=2, hyp_layer=2, binary=True, p=0.5, dropout=0.2, device=torch.device('cpu')): 
        super().__init__() 
        self.p = p
        self.device = device
        self.hyp_layer = hyp_layer
        self.rep_layer_fst = nn.Linear(in_feat, rep_hid)
        self.rep_bn_fst = nn.BatchNorm1d(rep_hid)
    
        self.rep_layers = nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(rep_layer-1)])
        self.rep_bns = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(rep_layer-1)])

        self.hyp_layer_fst0 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst0 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-1)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-1)])
        self.hyp_out0 = nn.Linear(rep_hid, 1) 

        self.hyp_layer_fst1 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst1 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers1= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-1)])
            self.hyp_bns1 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-1)])
        self.hyp_out1 = nn.Linear(rep_hid, 1) 
        self.dropout = nn.Dropout(p=dropout)
        self.binary = binary
        if binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
  
    def forward(self, X, C, Y): 
        X = X.view(-1, X.size(-1))
        Y = Y.view(-1)

        h = self.dropout(F.relu(self.rep_bn_fst(self.rep_layer_fst(X))))
        for fc, bn in zip(self.rep_layers, self.rep_bns):
            h = self.dropout(F.relu(bn(fc(h))))

        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))

        h1 = self.dropout(F.relu(self.hyp_bn_fst1(self.hyp_layer_fst1(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers1, self.hyp_bns1):
                h1 = self.dropout(F.relu(bn(fc(h1))))

        y0 = self.hyp_out0(h0).view(-1)
        y1 = self.hyp_out1(h1).view(-1)
 
        try:
            imb = mmd2_rbf(h,C,self.p)
        except:
            imb = 0.

        C_1d = C.view(-1)
        weight = C_1d/(2*self.p) + (1-C_1d)/(2*(1-self.p))
        # weight = weight.to(self.device)
        y = torch.where(C_1d > 0, y1, y0)
        loss = self.criterion(y, Y, reduction='none')
        loss = torch.mean(loss * weight) + 1e-4*imb
        if self.binary:
            y = torch.sigmoid(y)
            y0 = torch.sigmoid(y0)
            y1 = torch.sigmoid(y1)
        return loss, y, y0, y1 

class CFR_WASS(nn.Module): 
    def __init__(self, in_feat, rep_hid, hyp_hid, rep_layer=2, hyp_layer=2, binary=True, p=0.5, dropout=0.2, device=torch.device('cpu')): 
        super().__init__() 
        self.p = p
        self.device = device
        self.hyp_layer = hyp_layer
        self.rep_layer_fst = nn.Linear(in_feat, rep_hid)
        self.rep_bn_fst = nn.BatchNorm1d(rep_hid)
    
        self.rep_layers = nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(rep_layer-1)])
        self.rep_bns = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(rep_layer-1)])

        self.hyp_layer_fst0 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst0 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-1)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-1)])
        self.hyp_out0 = nn.Linear(rep_hid, 1) 

        self.hyp_layer_fst1 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst1 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers1= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-1)])
            self.hyp_bns1 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-1)])
        self.hyp_out1 = nn.Linear(rep_hid, 1) 
        self.dropout = nn.Dropout(p=dropout)
        self.binary = binary
        if binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
  
    def forward(self, X, C, Y): 
        X = X.view(-1, X.size(-1))
        Y = Y.view(-1)
        h = self.dropout(F.relu(self.rep_bn_fst(self.rep_layer_fst(X))))
        for fc, bn in zip(self.rep_layers, self.rep_bns):
            h = self.dropout(F.relu(bn(fc(h))))

        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))

        h1 = self.dropout(F.relu(self.hyp_bn_fst1(self.hyp_layer_fst1(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers1, self.hyp_bns1):
                h1 = self.dropout(F.relu(bn(fc(h1))))

        y0 = self.hyp_out0(h0).view(-1)
        y1 = self.hyp_out1(h1).view(-1)
        try:
            imb, _ = wasserstein_ht(h,C,self.p,self.device)
        except:
            imb = 0.

        C_1d = C.view(-1)
        y = torch.where(C_1d > 0, y1, y0)
        weight = C_1d/(2*self.p) + (1-C_1d)/(2*(1-self.p))
        # weight = weight.to(self.device)
        loss = self.criterion(y, Y, reduction='none')
        loss = torch.mean(loss * weight) + 1e-4*imb
        if self.binary:
            y = torch.sigmoid(y)
            y0 = torch.sigmoid(y0)
            y1 = torch.sigmoid(y1)
        return loss, y, y0, y1 


class PDDM(nn.Module):
    def __init__(self, h_dim, dropout=0.2): 
        super().__init__() 
        self.nn_u = nn.Linear(h_dim,h_dim,bias=True) 
        self.nn_v = nn.Linear(h_dim,h_dim,bias=True) 
        self.nn_c = nn.Linear(h_dim*2,h_dim,bias=True) 
        self.bn = nn.LayerNorm(h_dim*2)
        self.nn_s = nn.Linear(h_dim,1,bias=False) 
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,z1,z2):
        u = torch.abs(z1-z2)
        v = torch.abs(z1+z2)/2.0
        u = u / (torch.norm(u,p=2)+1e-7)
        v = v / (torch.norm(v,p=2)+1e-7)
        # print(u.shape,'u',v.shape,'v')
        u1 = torch.relu(self.nn_u(u))
        v1 = torch.relu(self.nn_v(v))
        uv = torch.cat((u1,v1),dim=-1) # size h_dim*2
        uv = self.bn(uv)
        uv = self.dropout(uv)
        h = torch.relu(self.nn_c(uv))
        hat_S = self.nn_s(h)
        # print(hat_S.shape,'hat s')
        return hat_S.double()

 
def func_PDDM_dis(a,b):
    r = 0.75*torch.abs((a+b)/2.0-0.5) - torch.abs((a-b)/2.0) + 0.5
    return r.double()

def func_MPDM(zi,zm,zj,zk):
    tmp = ((zi+zm)/2.0-(zj+zk)/2.0)
    r = torch.pow(2, tmp).sum()
    return r.double()

class SITE(nn.Module):
    def __init__(self, in_feat, rep_hid, hyp_hid, rep_layer=2, hyp_layer=2, binary=True, dropout=0.2): 
        super().__init__() 
        self.hyp_layer = hyp_layer
        self.rep_layer_fst = nn.Linear(in_feat, rep_hid)
        self.rep_bn_fst = nn.BatchNorm1d(rep_hid)
    
        self.rep_layers = nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(rep_layer-1)])
        self.rep_bns = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(rep_layer-1)])

        self.hyp_layer_fst0 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst0 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-1)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-1)])
        self.hyp_out0 = nn.Linear(rep_hid, 1) 

        self.hyp_layer_fst1 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst1 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers1= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-1)])
            self.hyp_bns1 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-1)])
        self.hyp_out1 = nn.Linear(rep_hid, 1) 
        self.dropout = nn.Dropout(p=dropout)
  
        self.PDDM = PDDM(rep_hid,dropout) 
        self.bn = nn.BatchNorm1d(rep_hid)
        self.dropout = nn.Dropout(p=dropout)
        self.binary = binary
        if binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
  
    def forward(self, X, C, P, Y): 
        X = X.view(-1, X.size(-1))
        Y = Y.view(-1)
        C = C.view(-1)
        P = P.view(-1)
        # print(X.shape,C.shape,P.shape,Y.shape)
        I_t = (C>0).nonzero().view(-1)#torch.where(C>0)[0]
        I_c = (C<1).nonzero().view(-1)#torch.where(C<1)[0]
        t_idx_map = dict(zip(range(len(I_t)),I_t.data.cpu().numpy()))
        c_idx_map = dict(zip(range(len(I_c)),I_c.data.cpu().numpy()))
        # print('c_idx_map',c_idx_map)
        # print(I_t.shape,I_c.shape,'I_c')
        prop_t = P[I_t].data.cpu()
        prop_c = P[I_c].data.cpu()
        # find x_i, x_j 
        index_i, index_j = find_middle_pair(prop_t, prop_c)
        # print('index_i, index_j',index_i, index_j)
        # find x_k, x_l
        index_k = torch.argmax(torch.abs(prop_c - prop_t[index_i])).item()
        index_l = find_nearest_point(prop_c, prop_c[index_k])
        # print('index_k, index_l',index_k, index_l)
        # find x_n, x_m
        index_m = torch.argmax(np.abs(prop_t - prop_c[index_j])).item()
        index_n = find_nearest_point(prop_t, prop_t[index_m,])
        # print('index_m, index_n',index_m, index_n)
        index_i = t_idx_map[index_i]
        index_j = c_idx_map[index_j]
        index_k = c_idx_map[index_k]
        index_l = c_idx_map[index_l]
        index_m = t_idx_map[index_m]
        index_n = t_idx_map[index_n]

        h = self.dropout(F.relu(self.rep_bn_fst(self.rep_layer_fst(X))))
        for fc, bn in zip(self.rep_layers, self.rep_bns):
            h = self.dropout(F.relu(bn(fc(h))))
        z = h
        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))

        h1 = self.dropout(F.relu(self.hyp_bn_fst1(self.hyp_layer_fst1(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers1, self.hyp_bns1):
                h1 = self.dropout(F.relu(bn(fc(h1))))

        y0 = self.hyp_out0(h0).view(-1)
        y1 = self.hyp_out1(h1).view(-1)
 
        y = torch.where(C > 0, y1, y0)
        loss_factual = self.criterion(y, Y)
        func_dis_los = nn.MSELoss(reduction='none')
        hat_S_kl = self.PDDM(z[index_k],z[index_l])
        S_kl = func_PDDM_dis(P[index_k],P[index_l]) 
        hat_S_mn = self.PDDM(z[index_m],z[index_n])
        S_mn = func_PDDM_dis(P[index_m],P[index_n]) 
        hat_S_km = self.PDDM(z[index_k],z[index_m])
        S_km = func_PDDM_dis(P[index_k],P[index_m]) 
        hat_S_im = self.PDDM(z[index_i],z[index_m])
        S_im = func_PDDM_dis(P[index_i],P[index_m]) 
        hat_S_jk = self.PDDM(z[index_j],z[index_k])
        S_jk = func_PDDM_dis(P[index_j],P[index_k]) 
        loss_PDDM = 0.2 * torch.sum(func_dis_los(hat_S_kl,S_kl) + func_dis_los(hat_S_mn,S_mn) + func_dis_los(hat_S_km,S_km) + func_dis_los(hat_S_im,S_im) + func_dis_los(hat_S_jk, S_jk))
        # print('loss_PDDM',loss_PDDM,type(loss_PDDM))
        loss_MPDM = func_MPDM(z[index_i],z[index_m],z[index_j],z[index_k])
        # print('loss_MPDM',loss_MPDM,type(loss_MPDM))
        beta = 1e-1
        gamma = 1e-2
        loss = loss_factual + beta*loss_PDDM + gamma*loss_MPDM
        if self.binary:
            y = torch.sigmoid(y)
            y0 = torch.sigmoid(y0)
            y1 = torch.sigmoid(y1)
        return loss, y, y0, y1 

    