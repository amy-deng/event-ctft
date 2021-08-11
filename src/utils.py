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
        self.model = args.model
        self.cuda = args.cuda
        self.window = args.window
        self.horizon = args.horizon
        self.pred_window = args.pred_window
        self.treat_idx = args.treat_idx
        self.aggr_feat = args.aggr_feat

        # load labels: Y treatments X
        with open('{}/{}/cf_data.pkl'.format(args.data_path, self.dataset),'rb') as f:
            data_dict = pickle.load(f)
        data_time = data_dict['TIME'] # n
        self.data_Y = np.array(data_dict['Y']) # n
        data_treat = data_dict['C'] # n * #c
        print(data_treat.mean(0))
        # exit()
        self.data_Y_cf = np.array(data_dict['CF_Y']) # n 
        data_treat_cf = data_dict['CF_C'] # n
        if not (data_treat == 1 - data_treat_cf).any():
            print('treatment error. check')
            exit()
        self.data_X = data_dict['X'] # load features: count data X 
        self.data_Xsm = data_dict['X_sm']
        if self.aggr_feat:
            self.f = self.data_X.shape[-1]
        else:
            self.f = self.data_X.shape[2]*self.data_X.shape[1]
        self.treatment = data_treat[:,self.treat_idx]
        print('<<< original treated proportion {:.4f} >>>'.format(self.treatment.mean()))
        self.data_Y_cf = self.data_Y_cf[:,self.treat_idx]
        # self.treatment_cf = data_treat_cf[:,self.treat_idx]
        self.Y1 = self.treatment * self.data_Y + (1-self.treatment) * self.data_Y_cf
        self.Y0 = (1-self.treatment) * self.data_Y + self.treatment * self.data_Y_cf
        print('< y_f {:.4f}  y_cf {:.4f} >'.format(self.data_Y.mean(),self.data_Y_cf.mean()))
        print('<<< data processed >>>')
        # self.realization_and_split(args.train,args.val,args.test)
        # load graph features TODO

    def realization_and_split(self, train, valid, test):
        # generate treatments and corresponding outcomes
        self.rea_treat = torch.randint(0, 2, self.treatment.shape)*1.0 #np.random
        self.rea_y = torch.tensor(self.rea_treat * self.Y1 + (1-self.rea_treat) * self.Y0)
        
        self.Y1 = torch.tensor(self.Y1)
        self.Y0 = torch.tensor(self.Y0)
        if self.aggr_feat:
            self.rea_x = torch.FloatTensor(self.data_X.sum(1)).unsqueeze(1)
            print('< aggregate event counts in historical days>')
        else:
            self.rea_x = torch.FloatTensor(self.data_X)
        # split data into train, val and test
        idx = list(range(len(self.rea_y)))
        random.shuffle(idx) # set random.seed(42)
        ind_train = int(round(self.rea_y.shape[0]*train)) 
        ind_test = int(round(self.rea_y.shape[0]*(train+test))) 
        train_idx = idx[:ind_train]
        val_idx = idx[ind_train:ind_test]
        test_idx = idx[ind_test:]
        train_val_idx = idx[:ind_test]
        train_treat = self.rea_treat[train_idx]
        val_treat = self.rea_treat[val_idx]
        test_treat = self.rea_treat[test_idx]
        train_val_treat = self.rea_treat[train_val_idx]

        # scale data/count features
        train_x = self.rea_x[train_idx]
        train_x_merge01 = train_x.view(-1,train_x.shape[-1])
        means = train_x_merge01.mean(dim=0)
        stds = train_x_merge01.std(dim=0)
        means_repeated = means.repeat(train_x.shape[0],train_x.shape[1],1)
        stds_repeated = stds.repeat(train_x.shape[0],train_x.shape[1],1)
        # print('self.train[2]',self.train[2].shape)
        # print('means_repeated',means_repeated.shape,stds_repeated.shape)
        train_x  = (train_x  - means_repeated) / (stds_repeated+1e-12)

        val_x = self.rea_x[val_idx]
        means_repeated = means.repeat(val_x.shape[0],val_x.shape[1],1)
        stds_repeated = stds.repeat(val_x.shape[0],val_x.shape[1],1)
        val_x  = (val_x  - means_repeated) / (stds_repeated+1e-12)

        test_x = self.rea_x[test_idx]
        means_repeated = means.repeat(test_x.shape[0],test_x.shape[1],1)
        stds_repeated = stds.repeat(test_x.shape[0],test_x.shape[1],1)
        test_x  = (test_x  - means_repeated) / (stds_repeated+1e-12)

        train_val_x = self.rea_x[train_val_idx]
        means_repeated = means.repeat(train_val_x.shape[0],train_val_x.shape[1],1)
        stds_repeated = stds.repeat(train_val_x.shape[0],train_val_x.shape[1],1)
        train_val_x  = (train_val_x  - means_repeated) / (stds_repeated+1e-12)
        if torch.any(train_val_x.isnan()):
            print('train_val has nan')
            exit()
        if torch.any(test_x.isnan()):
            print('test has nan')
            exit()

        if self.model == 'site':
            
            # print('x_train',x_train.shape,'y_train',y_train.shape)
            train_x_merge10 = train_x.view(train_x.shape[0],-1)
            val_x_merge10 = val_x.view(val_x.shape[0],-1)
            test_x_merge10 = test_x.view(test_x.shape[0],-1)
            train_val_x_merge10 = train_val_x.view(train_val_x.shape[0],-1)

            clf = LogisticRegression().fit(train_x_merge10, self.rea_treat[train_idx])

            train_p = clf.predict_proba(train_x_merge10)[:,1]
            val_p = clf.predict_proba(val_x_merge10)[:,1]
            test_p = clf.predict_proba(test_x_merge10)[:,1]
            train_val_p = clf.predict_proba(train_val_x_merge10)[:,1]
            # print(train_p.shape,'train_p',self.rea_treat[train_idx].shape)
            train_p = torch.from_numpy(train_p).double()
            val_p = torch.from_numpy(val_p).double()
            test_p = torch.from_numpy(test_p).double()
            train_val_p = torch.from_numpy(train_val_p).double()
            print("<< propensity scores predicted >>")
        else:
            train_p = torch.zeros(train_treat.shape) 
            val_p = torch.zeros(val_treat.shape) 
            test_p = torch.zeros(test_treat.shape) 
            train_val_p = torch.zeros(train_val_treat.shape) 
        
        
        self.train = [train_treat, self.rea_y[train_idx], train_x, self.Y1[train_idx], self.Y0[train_idx], train_p]
        self.val = [val_treat, self.rea_y[val_idx], val_x, self.Y1[val_idx], self.Y0[val_idx], val_p]
        self.test = [test_treat, self.rea_y[test_idx], test_x, self.Y1[test_idx], self.Y0[test_idx], test_p]
        self.train_val = [train_val_treat, self.rea_y[train_val_idx], train_val_x, self.Y1[train_val_idx], self.Y0[train_val_idx], train_val_p]

        print("<< the proportion of treated units: >>")
        print("  < train {:.4f} >".format(self.train[0].mean()))
        print("  < val {:.4f} >".format(self.val[0].mean()))
        print("  < test {:.4f} >".format(self.test[0].mean()))
        print("  < train_val {:.4f} >".format(self.train_val[0].mean()))

        print('<< realization, spliting and scaling done >>')
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
        [C, Y, X, Y1, Y0, P] = data
        # print(C.shape,Y.shape,X.shape,Y1.shape)
        # torch.Size([329]) torch.Size([329]) torch.Size([329, 10, 24]) torch.Size([329])
        length = len(C)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            if len(excerpt) < batch_size:
                random_pad = torch.randperm(length)[:(batch_size-len(excerpt))]
                excerpt = torch.cat([excerpt,random_pad])
            c = C[excerpt]
            y = Y[excerpt]
            x = X[excerpt,:]
            y1 = Y1[excerpt]
            y0 = Y0[excerpt]
            p = P[excerpt]
            if self.cuda:  
                c = c.cuda()
                y = y.cuda()
                x = x.cuda()
                y1 = y1.cuda()
                y0 = y0.cuda()
                p = p.cuda()
            data = [Variable(c), Variable(y), Variable(x), Variable(y1), Variable(y0), Variable(p)]
            yield data
            start_idx += batch_size
         
def eval_causal_effect_cf(y1_true, y0_true, y1_pred, y0_pred):
    r = {}
    # not discriminate locations
    # treatment = treatment.reshape(-1)
    # yf = yf.reshape(-1)
    # y1_pred = y1_pred.reshape(-1)
    # y0_pred = y0_pred.reshape(-1)
    # y_both = y_both.reshape(-1,2)
     
    # counterfactual auc
    y1y0_pred = np.concatenate((y1_pred, y0_pred), axis=0)
    y1y0_true = np.concatenate((y1_true, y0_true), axis=0)
    r['cf_auc'] = metrics.roc_auc_score(y1y0_true, y1y0_pred)

    ite_est = y1_pred - y0_pred # y1 - y0
    r['ate_err'] = ate(ite_est, y1_true, y0_true) 
    r['pehe'] = pehe(ite_est, y1_true, y0_true)  
    # print('policy_risk', 1-policy_value,  policy_risk)
    return r

def ate(ite_est, y1_true, y0_true):
    ate_true = np.mean(y1_true - y0_true)
    ate_est = np.mean(ite_est)
    return np.absolute(ate_true - ate_est)

def pehe(ite_est, y1_true, y0_true):
    ite_true = y1_true - y0_true
    return np.mean((ite_est - ite_true)**2)

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def mmd2_rbf(X,t,p=0.5,sig=0.1):
    """ Computes the l2-RBF MMD for X given t """
    it = 1*(t>0).nonzero().view(-1)
    ic = 1*(t<1).nonzero().view(-1)
    Xc = X[ic]
    Xt = X[it]
    Kcc = torch.exp(-pdist(Xc,Xc)/(sig**2))
    Kct = torch.exp(-pdist(Xc,Xt)/(sig**2))
    Ktt = torch.exp(-pdist(Xt,Xt)/(sig**2))
    m = Xc.size(0) 
    n = Xt.size(0) 
    mmd = (1.0-p)**2/(m*(m-1.0))*(Kcc.sum()-m)
    mmd = mmd + p**2/(n*(n-1.0))*(Ktt.sum()-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*Kct.sum()
    mmd = 4.0*mmd
    return mmd

def wasserstein_ht(X,t,p=0.5,lam=10,its=10,sq=False,backpropT=False,device=torch.device('cpu')):
    """return W dist between x and y"""
    '''distance matrix M'''
    # device = torch.device('cuda' if cuda else 'cpu')
    it = 1*(t>0).nonzero().view(-1)
    ic = 1*(t<1).nonzero().view(-1)
    Xc = X[ic]
    Xt = X[it]
    n = Xt.size(0) 
    m = Xc.size(0) 
    nx = n
    ny = m
    M = pdist(Xt,Xc) #distance_matrix(x,y,p=2)
    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    # print('nx=',nx,'ny=',ny,'M_mean',M_mean,'M',M.shape)

    # try:
    M_drop = F.dropout(M,10.0/(nx*ny))
    # except:
    #     M_drop = M
        # M_drop = F.dropout(M,0.01)
        # return torch.tensor(0.0).to(device)
    # print('M_drop',M_drop.shape)
    delta = torch.max(M_drop).detach()
    eff_lam = (lam/M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
     
    row = delta*torch.ones(M[0:1,:].shape).to(device)
    col = torch.cat([delta*torch.ones(M[:,0:1].shape).to(device),torch.zeros((1,1)).to(device)],0)
    row = row.to(device)
    col = col.to(device)
    Mt = torch.cat([M,row],0)
    Mt = torch.cat([Mt,col],1)

    '''compute marginal'''
    a = torch.cat([p*torch.ones((nx,1)).to(device)/nx,(1-p)*torch.ones((1,1)).to(device)],0)
    b = torch.cat([(1-p)*torch.ones((ny,1)).to(device)/ny, p*torch.ones((1,1)).to(device)],0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1)*1e-6
    temp_term = temp_term.to(device)
    a = a.to(device)
    b = b.to(device)
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K/a
    u = a
    for i in range(its):
        u = 1.0/(ainvK.matmul(b/torch.t(torch.t(u).matmul(K)))).to(device)
        # if cuda:
        #     u = u.cuda()
    v = b/(torch.t(torch.t(u).matmul(K)))
    v = v.to(device)

    upper_t = u*(torch.t(v)*K).detach()

    E = upper_t*Mt
    D = 2*torch.sum(E)
    D = D.to(device)
    return D, Mlam


def find_middle_pair(x, y):
    x = torch.abs(x-0.5)
    y = torch.abs(y-0.5)
    # print(x.shape,y.shape,'x y')
    index_1 = torch.argmin(x).item()
    index_2 = torch.argmin(y).item()
    # print('index_1',index_1,'index_2',index_2)
    # min = x[0] + y[0]
    # index_1, index_2 = 0, 0
    # for i in range(x.shape[0]):
    #     for j in range(y.shape[0]):
    #         value = x[i] + y[j]
    #         if value < min:
    #             min = value
    #             index_1, index_2 = i, j
    return index_1, index_2

def find_nearest_point(x, p):
    diff = torch.abs(x-p)
    diff_1 = diff[diff>0]
    if len(diff_1) < 1:
        # print('error')
        diff_1 = diff[diff>=0]
    # print('diff',diff.shape,diff_1.shape,'diff_1')
    # print p
    min_val = torch.min(diff_1)
    # print min_val
    I_diff = torch.where(diff == min_val)[0]
    # print(I_diff,I_diff.shape,'I_diff')
    I_diff = I_diff[0]
    if len(I_diff.size())>1:
        I_diff = I_diff[0]
    # print x[I_diff]
    return I_diff.item()