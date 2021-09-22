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


class DNN_F(nn.Module): 
    def __init__(self, args, data_loader): 
        super().__init__() 
        pass

    def forward(self, X, Y):
        pass 


class DnnEncoder(nn.Module):
    def __init__(self, in_feat, rep_hid, rep_layer=2, dropout=0.2, device=torch.device('cpu')):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.rep_layer_fst = nn.Linear(in_feat, rep_hid)
        self.rep_bn_fst = nn.BatchNorm1d(rep_hid)
    
        self.rep_layers = nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(rep_layer-1)])
        self.rep_bns = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(rep_layer-1)])

    def forward(self, X): 
        X = X.view(X.size(0), -1)
        # print(X.shape)
        h = self.dropout(F.relu(self.rep_bn_fst(self.rep_layer_fst(X))))
        for fc, bn in zip(self.rep_layers, self.rep_bns):
            h = self.dropout(F.relu(bn(fc(h))))
        return h

class RnnEncoder(nn.Module):
    def __init__(self, in_feat, rep_hid, rep_layer=2, dropout=0.2, device=torch.device('cpu')):
        super().__init__()
        self.rep_layer = rep_layer
        self.rep_gru = nn.GRU(in_feat,rep_hid,rep_layer,batch_first=True,dropout=dropout)
       
    def forward(self, X, h_0=None):
        output, hn = self.rep_gru(X,h_0)
        # print(output.shape,'output')
        # print(hn.shape,'hn')
        h = output[:,-1]
        return h

# TODO
class message_weight(nn.Module):
    def __init__(self, in_feat, rep_hid, m, rep_layer=2, dropout=0.2, device=torch.device('cpu')):
        super().__init__()
        self.rep_layer = rep_layer
        self.linear = nn.Linear(in_feat, rep_hid)
        self.rep_gru = nn.GRU(rep_hid,rep_hid,rep_layer,batch_first=True,dropout=dropout)
        self.w = nn.Parameter(torch.Tensor(m-1))
        torch.nn.init.ones_(self.w)
    def forward(self, X):
        h = self.linear(X)
        h = h.permute(0,2,3,1).contiguous()
        h = h @ self.w
        output, hn = self.rep_gru(h)
        return output

class message_mean(nn.Module):
    def __init__(self, in_feat, rep_hid, m, rep_layer=2, dropout=0.2, device=torch.device('cpu')):
        super().__init__()
        self.rep_layer = rep_layer
        self.linear = nn.Linear(in_feat, rep_hid)
        # self.w = 
        self.rep_gru = nn.GRU(rep_hid,rep_hid,rep_layer,batch_first=True,dropout=dropout)
    def forward(self, X):
        h = self.linear(X)
        h = h.mean(1)
        output, hn = self.rep_gru(h)
        return output

class message_p(nn.Module):
    def __init__(self, in_feat, rep_hid, m, rep_layer=2, dropout=0.2, device=torch.device('cpu')):
        super().__init__()
        self.rep_layer = rep_layer
        self.linear = nn.Linear(in_feat, rep_hid)
        self.rep_gru = nn.GRU(rep_hid,rep_hid,rep_layer,batch_first=True,dropout=dropout)
        # self.w = nn.Parameter(torch.Tensor(m-1))
        # torch.nn.init.ones_(self.w)
    def forward(self, X, C):
        h = self.linear(X)
        # print(h.shape,'h ',C.shape,'x c')
        # print(C)
        c_repeat = C.unsqueeze(-1).repeat(1,1,1,h.size(-1))
        # h = h.permute(0,2,3,1).contiguous()
        # print(h.shape,'h ',C.shape,'x c')
        h = (h * c_repeat).sum(1)
        # print(h)
        output, hn = self.rep_gru(h)
        return output

class message_weight_p(nn.Module):
    def __init__(self, in_feat, rep_hid, m, rep_layer=2, dropout=0.2, device=torch.device('cpu')):
        super().__init__()
        self.rep_layer = rep_layer
        self.linear = nn.Linear(in_feat, rep_hid)
        self.rep_gru = nn.GRU(rep_hid,rep_hid,rep_layer,batch_first=True,dropout=dropout)
        self.w = nn.Parameter(torch.Tensor(m-1))
        torch.nn.init.ones_(self.w)
    def forward(self, X, C):
        h = self.linear(X)
        # print(h.shape,'h ',C.shape,'x c',self.w.shape,'w')
        # print(C)
        w_repeat = self.w.view(1,-1,1)
        w_repeat = w_repeat.repeat(C.size(0),1,C.size(2))
        c = C*w_repeat
        c_repeat = c.unsqueeze(-1).repeat(1,1,1,h.size(-1))
        # h = h.permute(0,2,3,1).contiguous()
        # print(h.shape,'h ',C.shape,'x c')
        h = (h * c_repeat).sum(1)
        # print(h)
        output, hn = self.rep_gru(h)
        return output


class message_mlp(nn.Module):
    def __init__(self, in_feat, rep_hid, m, rep_layer=2, dropout=0.2, device=torch.device('cpu')):
        super().__init__()
        self.rep_layer = rep_layer
        self.linear = nn.Linear(in_feat, rep_hid)
        self.rep_gru = nn.GRU(rep_hid,rep_hid,rep_layer,batch_first=True,dropout=dropout)
        # self.w = nn.Parameter(torch.Tensor(m-1))
        # torch.nn.init.ones_(self.w)
        self.mlp = nn.Sequential(
            nn.Linear(rep_hid*2, rep_hid),
            nn.ReLU(),
             nn.Linear(rep_hid, 1) # x_target + 
        )
        self.linear2 =  nn.Linear(3, 1)
    def forward(self, X, C, h_target,distance):
        h = self.linear(X)
        # print(h.shape,'h ',C.shape,'x c',self.w.shape,'w')
        h_target_repeat = h_target.unsqueeze(1).repeat(1,h.size(1),1,1)
        # print(C.shape,'C',h_target.shape,'h_target',h_target_repeat.shape,'h_target_repeat',h.shape,'h')
        w = self.mlp(torch.cat((h_target_repeat,h),dim=-1))
        # print(w.shape,'w',distance.shape,'distance')
        distance_repeat = distance.view(1,-1,1,1).repeat(w.size(0),1,w.size(2),1)
        c = C.unsqueeze(-1)
        # print(c.shape,distance_repeat.shape,w.shape)
        coef = torch.cat((c,distance_repeat,w),dim=-1)
        learned_w = self.linear2(coef)
        # print(learned_w.shape,'learned_w')
        learned_w = learned_w.repeat(1,1,1,h.size(-1))
        # exit()
        # w_repeat = self.w.view(1,-1,1)
        # w_repeat = w_repeat.repeat(C.size(0),1,C.size(2))
        # c = C*w_repeat
        # c_repeat = c.unsqueeze(-1).repeat(1,1,1,h.size(-1))
        # h = h.permute(0,2,3,1).contiguous()
        # print(h.shape,'h ',C.shape,'x c')
        h = (h * learned_w).sum(1)
        # print(h)
        output, hn = self.rep_gru(h)
        return output

class DNN_F(nn.Module): 
    def __init__(self, args, data_loader): 
        super().__init__() 
        # self.p = p
        in_feat = data_loader.f
        self.device = args.device
        # self.dropout = args.dropout
        # self.p_alpha = args.p_alpha
        self.hyp_layer = args.hyp_layer
        self.rep_layer = args.rep_layer
        self.rep_dim = args.rep_dim
        self.hyp_dim = args.hyp_dim
        # encoder
        # self.rep_gru = nn.GRU(in_feat,rep_hid,1,batch_first=True,dropout=dropout)
        if args.enc == 'gru':
            self.encoder = RnnEncoder(in_feat, self.rep_dim, self.rep_layer, args.dropout, self.device)
        elif args.enc == 'dnn':
            self.encoder = DnnEncoder(in_feat, self.rep_dim, self.rep_layer, args.dropout, self.device)
        # decoder
        self.hyp_layer_fst0 = nn.Linear(self.rep_dim, self.hyp_dim)
        self.hyp_bn_fst0 = nn.BatchNorm1d(self.hyp_dim)
        if self.hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(self.hyp_dim, self.hyp_dim) for i in range(self.hyp_layer-2)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(self.hyp_dim) for i in range(self.hyp_layer-2)])
        self.hyp_out0 = nn.Linear(self.hyp_dim, 1) 
 
        self.dropout = nn.Dropout(p=args.dropout)
        # self.decoder = Decoder(self.rep_dim, self.hyp_dim, self.hyp_layer, self.dropout, self.device)
        self.binary = (not args.realy)
        if self.binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss

    def forward(self, X, Y): 
        # X = X.view(X.size(0), -1)
        # print('X',X.shape, 'Y',Y.shape)
        n, m, w, f = X.shape
        X = X[:,0]
        Y = Y[:,0]
        # X = X.view(n*m,w,f) 
        # Y = Y.view(-1) 
        # print('X',X.shape, 'Y',Y.shape)
        # print(X,Y)
        # output, hn = self.rep_gru(X)
        # h = output[:,-1]
        h = self.encoder(X)
        # h = self.dropout(F.relu(self.rep_bn_fst(self.rep_layer_fst(X))))
        # for fc, bn in zip(self.rep_layers, self.rep_bns):
        #     h = self.dropout(F.relu(bn(fc(h))))
        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))
 
        y = self.hyp_out0(h0).view(-1)
        loss = self.criterion(y, Y, reduction='mean')
        if self.binary:
            y = torch.sigmoid(y)
        return loss, y

class DNN_F_NEI(nn.Module): 
    def __init__(self, args, data_loader): 
        super().__init__() 
        # self.p = p
        in_feat = data_loader.f
        self.device = args.device
        # self.dropout = args.dropout
        # self.p_alpha = args.p_alpha
        self.hyp_layer = args.hyp_layer
        self.rep_layer = args.rep_layer
        self.rep_dim = args.rep_dim
        self.hyp_dim = args.hyp_dim
        # encoder
        # self.rep_gru = nn.GRU(in_feat,rep_hid,1,batch_first=True,dropout=dropout)
        self.message = message(in_feat, self.rep_dim, data_loader.m, self.rep_layer, args.dropout, self.device)
        self.linear = nn.Linear(in_feat, self.rep_dim)
        self.linear2 = nn.Linear(self.rep_dim*2, self.rep_dim)

        self.rnncell = nn.GRUCell(self.rep_dim, self.rep_dim)
        if args.enc == 'gru':
            self.encoder = RnnEncoder(self.rep_dim*2, self.rep_dim, self.rep_layer, args.dropout, self.device)
        elif args.enc == 'dnn':
            self.encoder = DnnEncoder(in_feat, self.rep_dim, self.rep_layer, args.dropout, self.device)
        # decoder
        self.hyp_layer_fst0 = nn.Linear(self.rep_dim, self.hyp_dim)
        self.hyp_bn_fst0 = nn.BatchNorm1d(self.hyp_dim)
        if self.hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(self.hyp_dim, self.hyp_dim) for i in range(self.hyp_layer-2)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(self.hyp_dim) for i in range(self.hyp_layer-2)])
        self.hyp_out0 = nn.Linear(self.hyp_dim, 1) 
 
        self.dropout = nn.Dropout(p=args.dropout)
        # self.decoder = Decoder(self.rep_dim, self.hyp_dim, self.hyp_layer, self.dropout, self.device)
        self.binary = (not args.realy)
        if self.binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss

    def forward(self, X, Y): 
        # X = X.view(X.size(0), -1)
        # print('X',X.shape, 'Y',Y.shape)
        n, m, w, f = X.shape
        # X = X[:,0]
        # Y = Y[:,0]
        # X = X.view(n*m,w,f) 
        # Y = Y.view(-1) 
        target_X = X[:,0]
        target_Y = Y[:,0]
        # print('target_X',target_X.shape, 'target_Y',target_Y.shape)
        neighbor_X = X[:,1:] # n, m, w, f 
        neighbor_Y = Y[:,1:]
        # print('neighbor_X',neighbor_X.shape, 'neighbor_Y',neighbor_Y.shape)
        # for each time step 
        # get message 
        h_nei = self.message(neighbor_X)
        # print(h_nei.shape,'h_nei')
        h_self = self.linear(target_X)
        
        # '''
        # rnn = nn.GRUCell(10, 20)
        # input = torch.randn(6, 3, 10)
        # hx = torch.randn(3, 20)
        output = []
        hx = torch.cat((torch.zeros(h_nei[:,0].shape),h_nei[:,0]),dim=-1)
        # print(hx.shape,'hx - - - -')
        hx = self.linear2(hx)
        # print(hx.shape,'hx = = == = ')
        hx = self.rnncell(h_self[:,0], hx)
        for i in range(1,7):
            hx = self.linear2(torch.cat((hx,h_nei[:,i]),dim=-1))
            hx = self.rnncell(h_self[:,i], hx)
        h = hx
        # '''
        # exit()
        # print('X',X.shape, 'Y',Y.shape)
        # print(X,Y)
        # output, hn = self.rep_gru(X)
        # h = output[:,-1]
        '''
        h = torch.cat((h_nei,h_self),dim=-1)
        h = self.encoder(h)
        '''
        # h = self.dropout(F.relu(self.rep_bn_fst(self.rep_layer_fst(X))))
        # for fc, bn in zip(self.rep_layers, self.rep_bns):
        #     h = self.dropout(F.relu(bn(fc(h))))
        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))
 
        y = self.hyp_out0(h0).view(-1)
        loss = self.criterion(y, target_Y, reduction='mean')
        if self.binary:
            y = torch.sigmoid(y)
        return loss, y
 
class Nei_mean(nn.Module): 
    def __init__(self, args, data_loader): 
        super().__init__() 
        # self.p = p
        in_feat = data_loader.f
        self.device = args.device
        # self.dropout = args.dropout
        # self.p_alpha = args.p_alpha
        self.hyp_layer = args.hyp_layer
        self.rep_layer = args.rep_layer
        self.rep_dim = args.rep_dim
        self.hyp_dim = args.hyp_dim
        self.device = args.device
        # encoder
        # self.rep_gru = nn.GRU(in_feat,rep_hid,1,batch_first=True,dropout=dropout)
        self.message = message_mean(in_feat, self.rep_dim, data_loader.m, self.rep_layer, args.dropout, self.device)
        self.linear = nn.Linear(in_feat, self.rep_dim)
        self.linear2 = nn.Linear(self.rep_dim*2, self.rep_dim)

        self.rnncell = nn.GRUCell(self.rep_dim, self.rep_dim) 
        # decoder
        self.hyp_layer_fst0 = nn.Linear(self.rep_dim, self.hyp_dim)
        self.hyp_bn_fst0 = nn.BatchNorm1d(self.hyp_dim)
        if self.hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(self.hyp_dim, self.hyp_dim) for i in range(self.hyp_layer-2)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(self.hyp_dim) for i in range(self.hyp_layer-2)])
        self.hyp_out0 = nn.Linear(self.hyp_dim, 1) 
 
        self.dropout = nn.Dropout(p=args.dropout)
        # self.decoder = Decoder(self.rep_dim, self.hyp_dim, self.hyp_layer, self.dropout, self.device)
        self.binary = (not args.realy)
        if self.binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss

    def forward(self, X, Y): 
        n, m, w, f = X.shape
        target_X = X[:,0]
        target_Y = Y[:,0]
        neighbor_X = X[:,1:] # n, m, w, f 
        neighbor_Y = Y[:,1:]
        h_nei = self.message(neighbor_X)
        # print(h_nei.shape,'h_nei')
        h_self = self.linear(target_X)
         
        hx = torch.cat((torch.zeros(h_nei[:,0].shape).to(self.device),h_nei[:,0]),dim=-1)
        hx = torch.tanh(self.linear2(hx))
        hx = self.rnncell(h_self[:,0], hx)
        for i in range(1,7):
            hx = torch.tanh(self.linear2(torch.cat((hx,h_nei[:,i]),dim=-1)))
            hx = self.rnncell(h_self[:,i], hx)
        h = hx
        
        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))
 
        y = self.hyp_out0(h0).view(-1)
        loss = self.criterion(y, target_Y, reduction='mean')
        if self.binary:
            y = torch.sigmoid(y)
        return loss, y

class Nei_weight(nn.Module): 
    def __init__(self, args, data_loader): 
        super().__init__() 
        # self.p = p
        in_feat = data_loader.f
        self.device = args.device
        # self.dropout = args.dropout
        # self.p_alpha = args.p_alpha
        self.hyp_layer = args.hyp_layer
        self.rep_layer = args.rep_layer
        self.rep_dim = args.rep_dim
        self.hyp_dim = args.hyp_dim
        self.device = args.device 
        # encoder
        # self.rep_gru = nn.GRU(in_feat,rep_hid,1,batch_first=True,dropout=dropout)
        self.message = message_weight(in_feat, self.rep_dim, data_loader.m, self.rep_layer, args.dropout, self.device)
        self.linear = nn.Linear(in_feat, self.rep_dim)
        self.linear2 = nn.Linear(self.rep_dim*2, self.rep_dim)

        self.rnncell = nn.GRUCell(self.rep_dim, self.rep_dim) 
        # decoder
        self.hyp_layer_fst0 = nn.Linear(self.rep_dim, self.hyp_dim)
        self.hyp_bn_fst0 = nn.BatchNorm1d(self.hyp_dim)
        if self.hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(self.hyp_dim, self.hyp_dim) for i in range(self.hyp_layer-2)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(self.hyp_dim) for i in range(self.hyp_layer-2)])
        self.hyp_out0 = nn.Linear(self.hyp_dim, 1) 
 
        self.dropout = nn.Dropout(p=args.dropout)
        # self.decoder = Decoder(self.rep_dim, self.hyp_dim, self.hyp_layer, self.dropout, self.device)
        self.binary = (not args.realy)
        if self.binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss

    def getConfounder(self, X):
        neighbor_X = X[:,1:] # n, m, w, f 
        h_nei = self.message(neighbor_X)
        # print(h_nei.shape,'==')
        return h_nei[:,-1]
        # return h_nei.reshape(h_nei.size(0),-1)

    def forward(self, X, Y): 
        n, m, w, f = X.shape
        target_X = X[:,0]
        target_Y = Y[:,0]
        neighbor_X = X[:,1:] # n, m, w, f 
        neighbor_Y = Y[:,1:]
        h_nei = self.message(neighbor_X)
        # print(h_nei.shape,'h_nei') 
        # print(h_nei[:,-1])
        # -1 is the external influence/ get this and estimate 
        h_self = self.linear(target_X)
        
        hx = torch.cat((torch.zeros(h_nei[:,0].shape).to(self.device),h_nei[:,0]),dim=-1)
        hx = torch.tanh(self.linear2(hx))
        hx = self.rnncell(h_self[:,0], hx)
        for i in range(1,7):
            hx = torch.tanh(self.linear2(torch.cat((hx,h_nei[:,i]),dim=-1)))
            hx = self.rnncell(h_self[:,i], hx)
        h = hx
        
        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))
 
        y = self.hyp_out0(h0).view(-1)
        loss = self.criterion(y, target_Y, reduction='mean')
        if self.binary:
            y = torch.sigmoid(y)
        return loss, y
 
class Nei_p(nn.Module): 
    def __init__(self, args, data_loader): 
        super().__init__() 
        # self.p = p
        in_feat = data_loader.f
        self.device = args.device
        # self.dropout = args.dropout
        # self.p_alpha = args.p_alpha
        self.hyp_layer = args.hyp_layer
        self.rep_layer = args.rep_layer
        self.rep_dim = args.rep_dim
        self.hyp_dim = args.hyp_dim
        self.device = args.device 
        # encoder
        # self.rep_gru = nn.GRU(in_feat,rep_hid,1,batch_first=True,dropout=dropout)
        self.message = message_p(in_feat, self.rep_dim, data_loader.m, self.rep_layer, args.dropout, self.device)
        self.linear = nn.Linear(in_feat, self.rep_dim)
        self.linear2 = nn.Linear(self.rep_dim*2, self.rep_dim)

        self.rnncell = nn.GRUCell(self.rep_dim, self.rep_dim) 
        # decoder
        self.hyp_layer_fst0 = nn.Linear(self.rep_dim, self.hyp_dim)
        self.hyp_bn_fst0 = nn.BatchNorm1d(self.hyp_dim)
        if self.hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(self.hyp_dim, self.hyp_dim) for i in range(self.hyp_layer-2)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(self.hyp_dim) for i in range(self.hyp_layer-2)])
        self.hyp_out0 = nn.Linear(self.hyp_dim, 1) 
 
        self.dropout = nn.Dropout(p=args.dropout)
        # self.decoder = Decoder(self.rep_dim, self.hyp_dim, self.hyp_layer, self.dropout, self.device)
        self.binary = (not args.realy)
        if self.binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss

    def forward(self, X, Y, C): 
        n, m, w, f = X.shape
        target_X = X[:,0]
        target_Y = Y[:,0]
        neighbor_X = X[:,1:] # n, m, w, f 
        neighbor_Y = Y[:,1:]
        neighbor_P = C[:,1:]
        h_nei = self.message(neighbor_X, neighbor_P)
        # print(h_nei.shape,'h_nei')
        h_self = self.linear(target_X)
         
        hx = torch.cat((torch.zeros(h_nei[:,0].shape).to(self.device),h_nei[:,0]),dim=-1)
        hx = torch.tanh(self.linear2(hx))
        hx = self.rnncell(h_self[:,0], hx)
        for i in range(1,7):
            hx = torch.tanh(self.linear2(torch.cat((hx,h_nei[:,i]),dim=-1)))
            hx = self.rnncell(h_self[:,i], hx)
        h = hx
        
        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))
 
        y = self.hyp_out0(h0).view(-1)
        loss = self.criterion(y, target_Y, reduction='mean')
        if self.binary:
            y = torch.sigmoid(y)
        return loss, y
 
class Nei_wp(nn.Module): 
    def __init__(self, args, data_loader): 
        super().__init__() 
        # self.p = p
        in_feat = data_loader.f
        self.device = args.device
        # self.dropout = args.dropout
        # self.p_alpha = args.p_alpha
        self.hyp_layer = args.hyp_layer
        self.rep_layer = args.rep_layer
        self.rep_dim = args.rep_dim
        self.hyp_dim = args.hyp_dim
        self.device = args.device 
        # encoder
        # self.rep_gru = nn.GRU(in_feat,rep_hid,1,batch_first=True,dropout=dropout)
        self.message = message_weight_p(in_feat, self.rep_dim, data_loader.m, self.rep_layer, args.dropout, self.device)
        self.linear = nn.Linear(in_feat, self.rep_dim)
        self.linear2 = nn.Linear(self.rep_dim*2, self.rep_dim)

        self.rnncell = nn.GRUCell(self.rep_dim, self.rep_dim) 
        # decoder
        self.hyp_layer_fst0 = nn.Linear(self.rep_dim, self.hyp_dim)
        self.hyp_bn_fst0 = nn.BatchNorm1d(self.hyp_dim)
        if self.hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(self.hyp_dim, self.hyp_dim) for i in range(self.hyp_layer-2)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(self.hyp_dim) for i in range(self.hyp_layer-2)])
        self.hyp_out0 = nn.Linear(self.hyp_dim, 1) 
 
        self.dropout = nn.Dropout(p=args.dropout)
        # self.decoder = Decoder(self.rep_dim, self.hyp_dim, self.hyp_layer, self.dropout, self.device)
        self.binary = (not args.realy)
        if self.binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss

    def forward(self, X, Y, C): 
        n, m, w, f = X.shape
        target_X = X[:,0]
        target_Y = Y[:,0]
        neighbor_X = X[:,1:] # n, m, w, f 
        neighbor_Y = Y[:,1:]
        neighbor_P = C[:,1:]
        h_self = self.linear(target_X)
        h_nei = self.message(neighbor_X, neighbor_P)
        # print(h_nei.shape,'h_nei')
         
        hx = torch.cat((torch.zeros(h_nei[:,0].shape).to(self.device),h_nei[:,0]),dim=-1)
        hx = torch.tanh(self.linear2(hx))
        hx = self.rnncell(h_self[:,0], hx)
        for i in range(1,7):
            hx = torch.tanh(self.linear2(torch.cat((hx,h_nei[:,i]),dim=-1)))
            hx = self.rnncell(h_self[:,i], hx)
        h = hx
        
        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))
 
        y = self.hyp_out0(h0).view(-1)
        loss = self.criterion(y, target_Y, reduction='mean')
        if self.binary:
            y = torch.sigmoid(y)
        return loss, y
 
class Nei_mlp(nn.Module): 
    def __init__(self, args, data_loader): 
        super().__init__() 
        # self.p = p
        in_feat = data_loader.f
        self.device = args.device
        # self.dropout = args.dropout
        # self.p_alpha = args.p_alpha
        self.hyp_layer = args.hyp_layer
        self.rep_layer = args.rep_layer
        self.rep_dim = args.rep_dim
        self.hyp_dim = args.hyp_dim
        self.device = args.device 
        self.distance = data_loader.distance
        # encoder
        # self.rep_gru = nn.GRU(in_feat,rep_hid,1,batch_first=True,dropout=dropout)
        self.message = message_mlp(in_feat, self.rep_dim, data_loader.m, self.rep_layer, args.dropout, self.device)
        self.linear = nn.Linear(in_feat, self.rep_dim)
        self.linear2 = nn.Linear(self.rep_dim*2, self.rep_dim)

        self.rnncell = nn.GRUCell(self.rep_dim, self.rep_dim) 
        # decoder
        self.hyp_layer_fst0 = nn.Linear(self.rep_dim, self.hyp_dim)
        self.hyp_bn_fst0 = nn.BatchNorm1d(self.hyp_dim)
        if self.hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(self.hyp_dim, self.hyp_dim) for i in range(self.hyp_layer-2)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(self.hyp_dim) for i in range(self.hyp_layer-2)])
        self.hyp_out0 = nn.Linear(self.hyp_dim, 1) 
 
        self.dropout = nn.Dropout(p=args.dropout)
        # self.decoder = Decoder(self.rep_dim, self.hyp_dim, self.hyp_layer, self.dropout, self.device)
        self.binary = (not args.realy)
        if self.binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss

    def forward(self, X, Y, C): 
        n, m, w, f = X.shape
        target_X = X[:,0]
        target_Y = Y[:,0]
        neighbor_X = X[:,1:] # n, m, w, f 
        neighbor_Y = Y[:,1:]
        neighbor_P = C[:,1:]
        h_self = self.linear(target_X)
        h_nei = self.message(neighbor_X, neighbor_P, h_self, self.distance[1:])
        # print(h_nei.shape,'h_nei')
         
        hx = torch.cat((torch.zeros(h_nei[:,0].shape).to(self.device),h_nei[:,0]),dim=-1)
        hx = torch.tanh(self.linear2(hx))
        hx = self.rnncell(h_self[:,0], hx)
        for i in range(1,7):
            hx = torch.tanh(self.linear2(torch.cat((hx,h_nei[:,i]),dim=-1)))
            hx = self.rnncell(h_self[:,i], hx)
        h = hx
        
        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))
 
        y = self.hyp_out0(h0).view(-1)
        loss = self.criterion(y, target_Y, reduction='mean')
        if self.binary:
            y = torch.sigmoid(y)
        return loss, y
 

class LinearCausalLearned(nn.Module): 
    def __init__(self, args, data_loader): 
        super().__init__() 
        # self.p = p
        in_feat = data_loader.f
        self.device = args.device 
        self.device = args.device 
        self.linear = nn.Linear(in_feat+(args.window)*64, 1, bias=False)
        self.criterion = nn.L1Loss()# F.mse_loss
        self.noise = GaussianNoise()

    def forward(self, X, Y, C):
        batch_size, n_loc, n_feat = X.size()
        # print(X.shape,'X',Y.shape,'Y',C.shape,'C')
        # sum up all time steps
        inp = torch.cat((X[:,0],C),dim=-1)
        # inp = torch.cat((X.view(batch_size,-1),C),dim=-1)

        
        # print(inp.shape,'inp')
        outputs = self.noise(self.linear(inp)).squeeze(1)
        # print(outputs.shape,'output')
        loss = self.criterion(outputs,Y[:,0])
        return loss, outputs
# ['0.148 0.042', '0.0 0.0', '0.141 0.02', '0.088 0.015']  raw and learned

# ['0.161 0.035', '0.0 0.0', '0.16 0.029', '0.102 0.025'] learned

class LinearCausalRaw(nn.Module): 
    def __init__(self, args, data_loader): 
        super().__init__() 
        # self.p = p
        in_feat = data_loader.f
        self.device = args.device 
        self.device = args.device 
        # self.linear = nn.Linear(in_feat, 1) 
        self.linear = nn.Linear(in_feat*data_loader.m, 1, bias=False) 
        self.criterion = nn.L1Loss()# F.mse_loss
        self.noise = GaussianNoise()
    def forward(self, X, Y, C):
        batch_size, n_loc, n_feat = X.size()
        # sum up all time steps
        
        # inp = torch.cat((X[:,0],C),dim=-1)
        inp = X.view(batch_size,-1)
        # inp = X[:,0]
        # print(inp.shape,'inp')
        outputs = self.noise(self.linear(inp)).squeeze(1)
        # print(outputs.shape,'output')
        loss = self.criterion(outputs,Y[:,0])
        return loss, outputs
# ['0.146 0.046', '0.0 0.0', '0.142 0.028', '0.094 0.024'] 
# ['0.151 0.047', '0.0 0.0', '0.141 0.022', '0.091 0.016']

# only use X[:,0]  
# ['0.169 0.048', '0.0 0.0', '0.159 0.03', '0.112 0.027']


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 

# nei ['0.331 0.022', '0.0 0.0', '0.901 0.002', '0.789 0.013', '0.653 0.026', '0.736 0.041', '0.691 0.016']
# ['0.326 0.022', '0.0 0.0', '0.903 0.007', '0.82 0.026', '0.628 0.037', '0.84 0.046', '0.718 0.04']
# ['0.334 0.021', '0.0 0.0', '0.904 0.004', '0.808 0.017', '0.643 0.021', '0.793 0.043', '0.709 0.013']
# ['0.321 0.025', '0.0 0.0', '0.9 0.012', '0.801 0.022', '0.642 0.04', '0.793 0.057', '0.707 0.031']
# ['0.324 0.028', '0.0 0.0', '0.902 0.014', '0.8 0.029', '0.656 0.045', '0.779 0.08', '0.709 0.028']
# ['0.354 0.035', '0.0 0.0', '0.888 0.017', '0.802 0.026', '0.635 0.041', '0.805 0.064', '0.707 0.023']


# ['0.38 0.038', '0.0 0.0', '0.877 0.015', '0.829 0.016', '0.645 0.029', '0.884 0.036', '0.746 0.027'] pw = 2
['0.364 0.027', '0.0 0.0', '0.885 0.017', '0.807 0.025', '0.649 0.035', '0.82 0.06', '0.722 0.025']


['0.306 0.037', '0.0 0.0', '0.921 0.016', '0.857 0.023', '0.722 0.04', '0.896 0.041', '0.798 0.023']
# ['0.293 0.048', '0.0 0.0', '0.926 0.013', '0.864 0.02', '0.717 0.036', '0.923 0.043', '0.806 0.023'] 7-2-2
['0.295 0.029', '0.0 0.0', '0.932 0.01', '0.84 0.027', '0.737 0.039', '0.845 0.082', '0.783 0.031']
# ['0.292 0.028', '0.0 0.0', '0.928 0.013', '0.846 0.024', '0.733 0.042', '0.864 0.062', '0.79 0.021'] 7

# 7 3 2 nei
['0.314 0.041', '0.0 0.0', '0.931 0.015', '0.862 0.024', '0.732 0.042', '0.903 0.054', '0.806 0.023']
# gruf
['0.307 0.047', '0.0 0.0', '0.929 0.012', '0.864 0.022', '0.736 0.027', '0.904 0.051', '0.81 0.023']


['0.313 0.045', '0.0 0.0', '0.931 0.013', '0.859 0.026', '0.735 0.023', '0.892 0.056', '0.805 0.02']
# 7-2-2 nei
['0.292 0.035', '0.0 0.0', '0.907 0.02', '0.813 0.032', '0.682 0.055', '0.818 0.072', '0.741 0.041']
['0.288 0.04', '0.0 0.0', '0.912 0.013', '0.819 0.038', '0.712 0.056', '0.804 0.109', '0.748 0.049']
['0.308 0.044', '0.0 0.0', '0.908 0.013', '0.801 0.017', '0.685 0.057', '0.775 0.052', '0.724 0.029']
['0.284 0.053', '0.0 0.0', '0.917 0.013', '0.815 0.037', '0.814 0.021', '0.707 0.054', '0.791 0.038', '0.745 0.031']
['0.291 0.045', '0.0 0.0', '0.914 0.014', '0.807 0.043', '0.801 0.027', '0.712 0.054', '0.752 0.053', '0.73 0.043']
['0.287 0.046', '0.0 0.0', '0.919 0.015', '0.825 0.031', '0.813 0.025', '0.709 0.058', '0.788 0.057', '0.744 0.039']

# 7-5-2
# gruf
['0.323 0.055', '0.0 0.0', '0.925 0.02', '0.826 0.03', '0.718 0.058', '0.813 0.103', '0.756 0.034']
# ['0.347 0.075', '0.0 0.0', '0.919 0.019', '0.825 0.048', '0.684 0.05', '0.839 0.12', '0.746 0.038'] weight
# ['0.355 0.07', '0.0 0.0', '0.921 0.016', '0.84 0.037', '0.694 0.062', '0.866 0.076', '0.767 0.043'] mean
['0.355 0.07', '0.0 0.0', '0.921 0.016', '0.819 0.038', '0.84 0.037', '0.694 0.062', '0.866 0.076', '0.767 0.043']

#711
['0.625 0.011', '0.0 0.0', '0.701 0.018', '0.726 0.024', '0.654 0.016', '0.668 0.023', '0.704 0.027', '0.685 0.016']
