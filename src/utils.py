import sys
import torch
import numpy as np
import random
import pickle
from math import sqrt
import scipy.sparse as sp
from torch.autograd import Variable
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

class CountDataLoader(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.cuda = args.cuda
        self.window = args.window
        self.horizon = args.horizon
        self.pred_window = args.pred_window
        self.treat_idx = args.treat_idx
        # load labels: Y treatments X
        with open('{}/{}/cf_data.pkl'.format(args.data_path, self.dataset),'rb') as f:
            data_dict = pickle.load(f)
        data_time = data_dict['TIME'] # n
        self.data_Y = data_dict['Y'] # n
        data_treat = data_dict['C'] # n * #c
        self.data_Y_cf = data_dict['CF_Y'] # n 
        data_treat_cf = data_dict['CF_C'] # n
        if not (data_treat == 1 - data_treat_cf):
            print('treatment error. check')
            exit()
        self.data_X = data_dict['X'] # load features: count data X 
        self.data_Xsm = data_dict['X_sm']
        self.treatment = data_treat[:,self.treat_idx]
        # self.treatment_cf = data_treat_cf[:,self.treat_idx]
        self.Y1 = self.treatment * self.data_Y + (1-self.treatment) * self.data_Y_cf
        self.Y0 = (1-self.treatment) * self.data_Y + self.treatment * self.data_Y_cf
        print('<<< data processed >>>')
        # self.realization_and_split(args.train,args.val,args.test)
        # load graph features TODO
    def realization_and_split(self, train, valid, test):
        # generate treatments and corresponding outcomes
        # then split data into train, val and test
        self.rea_treat = torch.randint(0, 2, self.treatment.shape)*1.0 #np.random
        self.rea_y = torch.tensor(self.rea_treat * self.Y1 + (1-self.rea_treat) * self.Y0)
        self.rea_x = torch.tensor(self.data_X)
        self.Y1 = torch.tensor(self.Y1)
        self.Y0 = torch.tensor(self.Y0)
        # self._split()

        idx = list(range(len(self.rea_y)))
        random.shuffle(idx) # set random.seed(42)

        ind_train = int(round(self.rea_y.shape[0]*train)) 
        ind_test = int(round(self.rea_y.shape[0]*(train+test))) 
        train_idx = idx[:ind_train]
        val_idx = idx[ind_train:ind_test]
        test_idx = idx[ind_test:]

        self.train = [self.rea_treat[train_idx], self.rea_y[train_idx], self.rea_x[train_idx], self.Y1[train_idx], self.Y0[train_idx]]
        self.val = [self.rea_treat[val_idx], self.rea_y[val_idx], self.rea_x[val_idx], self.Y1[val_idx], self.Y0[val_idx]]
        self.test = [self.rea_treat[test_idx], self.rea_y[test_idx], self.rea_x[test_idx], self.Y1[test_idx], self.Y0[test_idx]]

    # def _split(self, train, valid, test):
    #     idx = list(range(len(self.data_Y)))
    #     random.shuffle(idx) # set random.seed(42)

    #     ind_train = int(round(self.data_Y.shape[0]*train)) 5
    #     ind_test = int(round(self.data_Y.shape[0]*(train+test))) 8
    #     train_idx = idx[:ind_train]
    #     val_idx = idx[ind_train:ind_test]
    #     test_idx = idx[ind_test:]

    #     self.train = [self.rea_treat[train_idx], self.rea_y[train_idx], self.rea_x[train_idx]]
    #     self.val = [self.rea_treat[val_idx], self.rea_y[val_idx], self.rea_x[val_idx]]
    #     self.test = [self.rea_treat[test_idx], self.rea_y[test_idx], self.rea_x[test_idx]]
 

    def get_batches(self, data, batch_size, shuffle=True):
        [C, Y, X, Y1, Y0] = data
        length = len(C)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            c = C[excerpt,:]
            y = Y[excerpt,:]
            x = X[excerpt,:]
            y1 = Y1[excerpt,:]
            y0 = Y0[excerpt,:]
            if self.cuda:  
                c = c.cuda()
                y = y.cuda()
                x = x.cuda()
                y1 = y1.cuda()
                y0 = y0.cuda()
            data = [Variable(c), Variable(y), Variable(x), Variable(y1), Variable(y0)]
            yield data
            start_idx += batch_size
         
# TODO
def eval_causal_effect_cf(treatment, yf, y1_pred, y0_pred, y_both, ymax=None, ymin=None, lam=.0, y_pred=None,eval_pred=True):
    r = {}
    # not discriminate locations
    treatment = treatment.reshape(-1)
    yf = yf.reshape(-1)
    y1_pred = y1_pred.reshape(-1)
    y0_pred = y0_pred.reshape(-1)
    y_both = y_both.reshape(-1,2)
    if eval_pred:
        if ymax is None and ymin is None:  # AUC binary
            if y_pred is not None:
                y_pred = y_pred.reshape(-1)
            else:
                y_pred = np.where(treatment > 0, y1_pred, y0_pred) 
                        
            tmp = eval_bi_classifier(yf, y_pred,y_pred_bi=np.where(y_pred>0.5,1,0))
            r = {**tmp, **r}

        else: # real value and norm
            y1_pred = y1_pred * (ymax - ymin + 1e-12) + ymin
            y0_pred = y0_pred * (ymax - ymin + 1e-12) + ymin
            y_both = y_both * (ymax - ymin + 1e-12) + ymin
            y_pred = np.where(treatment > 0, y1_pred, y0_pred) 
            tmp = eval_regression(yf, y_pred)
            r = {**tmp, **r}

    # counterfactual auc
    y0y1_pred = np.concatenate((y0_pred, y1_pred), axis=0)
    y0y1_label = np.concatenate((y_both[:,0], y_both[:,1]), axis=0)
    r['cf_auc'] = metrics.roc_auc_score(y0y1_label, y0y1_pred)

    eff_pred = y1_pred - y0_pred # y1 - y0
    r['ate_err'] = ate_eval_cf(eff_pred, y_both) 
    r['pehe'] = pehe_eval(eff_pred, y_both)  
    
    # print('policy_risk', 1-policy_value,  policy_risk)
    return r