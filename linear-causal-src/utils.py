# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, os, re, time, string, random, math
import torch
#import nltk
# import bitarray as bt
import torch
import torch.nn.functional as F
from sklearn import preprocessing
import scipy.sparse as sp
from torch.autograd import Variable
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class DataLoaderFreq(object):
    def __init__(self, args):
        self.w = args.window # window 10 / 7
        self.h = args.horizon # 1
        self.hw = args.pred_window
        self.cuda = args.cuda
        self.dataset = args.dataset
        self.datafile = args.datafile
        self.train = args.train
        self.val = args.val
        self.test = args.test

        with open('../data/{}/{}'.format(self.dataset,self.datafile),'rb') as f:
            self.data = pkl.load(f)
        # self.data = self.data[:6]
        self.m, self.t, self.f = self.data.shape
        print("m = {} \t t = {} \t f = {}".format(self.m, self.t, self.f))
        if len(self.dataset) > 3:
            print(' ------    9,18,19,14,24  --------')
            # self.y = self.data[:,:,18]
            self.y = self.data[:,:,[9,18,19,14,24]].sum(-1) 
            # print(self.y)
            print(self.y.shape,'===========')
            '''
            ['Abduction/forced disappearance', 'Agreement', 'Air/drone strike', 'Armed clash', 'Arrests', 'Attack', 
            'Change to group/activity', 'Chemical weapon', 'Disrupted weapons use', 'Excessive force against protesters', 
            'Government regains territory', 'Grenade', 'Headquarters or base established', 'Looting/property destruction', 
            'Mob violence', 'Non-state actor overtakes territory', 'Non-violent transfer of territory', 'Other', 
            'Peaceful protest', 'Protest with intervention', 'Remote explosive/landmine/IED', 'Sexual violence', 
            'Shelling/artillery/missile attack', 'Suicide bomb', 'Violent demonstration']
            '''
        else:
            self.y = self.data[:,:,13]
        self.y = np.where(self.y > 0., 1., 0.)
        print("data {} \t y {}".format(self.data.shape, self.y.shape))
        print(self.y.mean(1))
        self._split(int(self.train * self.t), int((self.train + self.val) * self.t), self.t)

    def _split(self, train, val, test):
        self.train_set = range(self.w+self.h-1, train)
        self.val_set = range(train, val)
        self.test_set = range(val, self.t)

        self.train = self._batchify(self.train_set) # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.val_set)
        self.test = self._batchify(self.test_set)

        ## standardize
        x_train_2d = self.train[0].view(-1,self.f)
        scaler = preprocessing.StandardScaler().fit(x_train_2d)
        # print(scaler.mean_.shape,scaler.scale_.shape,'===')
        x_train_2d = torch.from_numpy(scaler.transform(x_train_2d))
        print(x_train_2d.shape,'x_train_2d',type(x_train_2d))
        self.train[0] = x_train_2d.view(self.train[0].shape) 

        x_val_2d = self.val[0].view(-1,self.f)
        x_val_2d = torch.from_numpy(scaler.transform(x_val_2d))
        self.val[0] = x_val_2d.view(self.val[0].shape) 

        x_test_2d = self.test[0].view(-1,self.f)
        x_test_2d = torch.from_numpy(scaler.transform(x_test_2d))
        self.test[0] = x_test_2d.view(self.test[0].shape) 


    def _batchify(self, idx_set):  
        n = len(idx_set)
        X = torch.zeros((n, self.m,  self.w, self.f)) 
        Y = torch.zeros((n, self.m))
        n_valid = 0
        for i in range(0,n,self.hw):
        # for i in range(0,n):
            end = idx_set[i] - self.h + 1
            start = end - self.w 
            start_y = end  + self.h -1
            end_y = start_y + self.hw
            # print('i =',i,start, end, start_y,end_y)
            # print(X[i,:,:,:].shape, Y[i,:,:].shape)
            X[i,:,:,:]  = torch.from_numpy(self.data[:,start:end, :]) 
            tmp_y = self.y[:,start_y:end_y].sum(-1)
            # print(tmp_y.shape,'tmp_y')
            tmp_y = np.where(tmp_y > 0, 1., 0.)
            Y[i]  = torch.from_numpy(tmp_y)
            # if i > 10:
            #     break
            n_valid+=1
        X = X[:n_valid] 
        Y = Y[:n_valid].double()
        print('X',X.shape, 'Y',Y.shape)
        return [X, Y] 

    def get_batches(self, data, batch_size, shuffle=True):
        [X, Y] = data
        length = len(X)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            yy = Y[excerpt]
            xx = X[excerpt]
            if (self.cuda):  
                xx = xx.cuda()
                yy = yy.cuda()
            data = [xx, Variable(yy)]
            yield data
            start_idx += batch_size
    
    def shuffle(self):
        [X_tr, Y_tr] = self.train
        [X_va, Y_va] = self.val
        [X_te, Y_te] = self.test
        X = torch.cat((X_tr, X_va, X_te), 0)
        Y = torch.cat((Y_tr, Y_va, Y_te), 0)
        # p_pos = Y[:,0].mean()
        p_pos = Y.mean(0)
        print(Y.shape,'Y')
        print('------ p_pos =',p_pos, 'Y',Y.shape)
        # exit()

        idx = list(range(Y.size(0)))
        random.shuffle(idx)
        idx_tr = idx[:Y_tr.size(0)]
        idx_va = idx[Y_tr.size(0):Y_tr.size(0)+Y_va.size(0)]
        idx_te = idx[-Y_te.size(0):]
        # print(len(idx_tr),len(idx_va),len(idx_te))
        
        self.train = [X[idx_tr], Y[idx_tr]]
        self.val = [X[idx_va], Y[idx_va]]
        self.test = [X[idx_te], Y[idx_te]]

class DataLoaderFreqPropensity(object):
    def __init__(self, args):
        self.w = args.window # window 10 / 7
        self.h = args.horizon # 1
        self.hw = args.pred_window
        self.cuda = args.cuda
        self.dataset = args.dataset
        self.datafile = args.datafile
        self.train = args.train
        self.val = args.val
        self.test = args.test
        with open('../data/{}/propensity.pkl'.format(self.dataset),'rb') as f:
            self.connection = pkl.load(f) #(n,m)
        
        with open('../data/{}/geo_raw.pkl'.format(self.dataset),'rb') as f:
            self.distance = pkl.load(f) #(m,)
        # self.distance = self.distance + 1e-2
        # print((1-self.distance)**0.5+1e-12,'~~~~~~~~')
        # print(self.distance,'~~~~~~~~')
        if args.model in ['nei_p2','nei_p2']:
            print('calculate p')
            self.distance = self.distance / (self.distance.max() + 100)
            self.connection = self.connection/(1-self.distance)  # **0.5
        self.connection = np.swapaxes(self.connection,0,1)
        print(self.distance,'self.distance',self.connection.max(),self.connection.min())
        with open('../data/{}/{}'.format(self.dataset,self.datafile),'rb') as f:
            self.data = pkl.load(f)     #(m,n,f)
        print('connection',self.connection.shape,'distance',self.distance.shape,'data',self.data.shape)
        # self.data = self.data[:6]
        self.m, self.t, self.f = self.data.shape
        print("m = {} \t t = {} \t f = {}".format(self.m, self.t, self.f))
        if len(self.dataset) > 3:
            print(' ------    9,18,19,14,24  --------')
            # self.y = self.data[:,:,18]
            self.y = self.data[:,:,[9,18,19,14,24]].sum(-1) 
            # print(self.y)
            print(self.y.shape,'===========')
            '''
            ['Abduction/forced disappearance', 'Agreement', 'Air/drone strike', 'Armed clash', 'Arrests', 'Attack', 
            'Change to group/activity', 'Chemical weapon', 'Disrupted weapons use', 'Excessive force against protesters', 
            'Government regains territory', 'Grenade', 'Headquarters or base established', 'Looting/property destruction', 
            'Mob violence', 'Non-state actor overtakes territory', 'Non-violent transfer of territory', 'Other', 
            'Peaceful protest', 'Protest with intervention', 'Remote explosive/landmine/IED', 'Sexual violence', 
            'Shelling/artillery/missile attack', 'Suicide bomb', 'Violent demonstration']
            '''
        else:
            self.y = self.data[:,:,13]
        self.y = np.where(self.y > 0., 1., 0.)
        print("data {} \t y {}".format(self.data.shape, self.y.shape))
        print(self.y.mean(1))
        # distance_repreat = 
        # propensity p(i affected by j) = p(i connect j)/p(i observe j) = similarity [0-1] / distance [0-1]
        self._split(int(self.train * self.t), int((self.train + self.val) * self.t), self.t)
        
        self.distance = torch.from_numpy(self.distance)
        if (self.cuda):  
            self.distance = self.distance.cuda()

    def _split(self, train, val, test):
        self.train_set = range(self.w+self.h-1, train)
        self.val_set = range(train, val)
        self.test_set = range(val, self.t)

        self.train = self._batchify(self.train_set) # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.val_set)
        self.test = self._batchify(self.test_set)

        ## standardize
        x_train_2d = self.train[0].view(-1,self.f)
        scaler = preprocessing.StandardScaler().fit(x_train_2d)
        # print(scaler.mean_.shape,scaler.scale_.shape,'===')
        x_train_2d = torch.from_numpy(scaler.transform(x_train_2d))
        print(x_train_2d.shape,'x_train_2d',type(x_train_2d))
        self.train[0] = x_train_2d.view(self.train[0].shape) 

        x_val_2d = self.val[0].view(-1,self.f)
        x_val_2d = torch.from_numpy(scaler.transform(x_val_2d))
        self.val[0] = x_val_2d.view(self.val[0].shape) 

        x_test_2d = self.test[0].view(-1,self.f)
        x_test_2d = torch.from_numpy(scaler.transform(x_test_2d))
        self.test[0] = x_test_2d.view(self.test[0].shape) 


    def _batchify(self, idx_set):  
        n = len(idx_set)
        X = torch.zeros((n, self.m,  self.w, self.f)) 
        Y = torch.zeros((n, self.m))
        C = torch.zeros((n, self.m,  self.w))
        n_valid = 0
        for i in range(0,n,self.hw):
        # for i in range(0,n):
            end = idx_set[i] - self.h + 1
            start = end - self.w 
            start_y = end  + self.h -1
            end_y = start_y + self.hw
            # print('i =',i,start, end, start_y,end_y)
            # print(X[i,:,:,:].shape, Y[i,:,:].shape)
            X[i,:,:,:]  = torch.from_numpy(self.data[:,start:end, :]) 
            C[i,:,:]  = torch.from_numpy(self.connection[:,start:end])
            tmp_y = self.y[:,start_y:end_y].sum(-1)
            # print(tmp_y.shape,'tmp_y')
            tmp_y = np.where(tmp_y > 0, 1., 0.)
            Y[i]  = torch.from_numpy(tmp_y)
            # if i > 10:
            #     break
            n_valid+=1
        X = X[:n_valid] 
        Y = Y[:n_valid]
        C = C[:n_valid]
        print('X',X.shape, 'Y',Y.shape, 'C',C.shape)
        print(C.max(),C.min(),C.mean(),C.std(),'C - max - min - mean - std')
        # print(C)
        # exit()
        return [X, Y, C] 
# 
    def get_batches(self, data, batch_size, shuffle=True):
        [X, Y, C] = data
        length = len(X)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            yy = Y[excerpt]
            xx = X[excerpt]
            cc = C[excerpt]
            if (self.cuda):  
                xx = xx.cuda()
                yy = yy.cuda()
                cc = cc.cuda()
            data = [Variable(xx), Variable(yy), Variable(cc)]
            yield data
            start_idx += batch_size
    
    def shuffle(self):
        [X_tr, Y_tr, C_tr] = self.train
        [X_va, Y_va, C_va] = self.val
        [X_te, Y_te, C_te] = self.test
        X = torch.cat((X_tr, X_va, X_te), 0)
        Y = torch.cat((Y_tr, Y_va, Y_te), 0)
        C = torch.cat((C_tr, C_va, C_te), 0)
        # p_pos = Y[:,0].mean()
        p_pos = Y.mean(0)
        print(Y.shape,'Y')
        print('------ p_pos =',p_pos, 'Y',Y.shape)
        # exit()

        idx = list(range(Y.size(0)))
        random.shuffle(idx)
        idx_tr = idx[:Y_tr.size(0)]
        idx_va = idx[Y_tr.size(0):Y_tr.size(0)+Y_va.size(0)]
        idx_te = idx[-Y_te.size(0):]
        # print(len(idx_tr),len(idx_va),len(idx_te))
        
        self.train = [X[idx_tr], Y[idx_tr], C[idx_tr]]
        self.val = [X[idx_va], Y[idx_va], C[idx_va]]
        self.test = [X[idx_te], Y[idx_te], C[idx_te]]


def split_data(size, train=.7, val=.15, test=.15, shuffle=True):
    idx = list(range(size))
    if shuffle:
        np.random.shuffle(idx)
    split_idx = np.split(idx, [int(train * len(idx)), int((train+val) * len(idx))])
    train_idx, val_idx, test_idx = split_idx[0], split_idx[1], split_idx[2]
    return train_idx, val_idx, test_idx



def eval_classifier(y_true, y_pred):
    # print(y_true.shape, y_pred.shape)
    r = {}
    r['auc'] = metrics.roc_auc_score(y_true, y_pred)
    # r['aupr'] = metrics.average_precision_score(y_true, y_pred)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    auc_score = metrics.auc(recall, precision)
    r['aupr'] = auc_score
    y_bi = np.where(y_pred>0.5, 1, 0)
    r['bacc'] = metrics.balanced_accuracy_score(y_true, y_bi)
    r['prec'] = metrics.precision_score(y_true, y_bi)
    r['rec'] = metrics.recall_score(y_true, y_bi)
    r['f1'] = metrics.f1_score(y_true, y_bi)
    return r