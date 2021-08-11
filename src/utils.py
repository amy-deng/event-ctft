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
 

class CountDataLoader(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.cuda = args.cuda
        self.window = args.window
        self.horizon = args.horizon
        self.pred_window = args.pred_window
        self.treat_idx = args.treat_idx
        # load labels: Y treatments
        with open('../data/{}/cf_data.pkl'.format(self.dataset),'rb') as f:
            data_dict = pickle.dump(f)
        data_time = data_dict['data_time'] # n
        data_Y = data_dict['Y'] # n
        data_treat = data_dict['C'] # n * #c
        data_Y_cf = data_dict['CF_Y'] # n 
        data_treat_cf = data_dict['CF_C'] # n
        if not (data_treat == 1 - data_treat_cf):
            print('treatment error. check')
            exit()
        treatment = data_treat[:,self.treat_idx]
        treatment_cf = data_treat_cf[:,self.treat_idx]

        # load features: count data X 
        with open('../data/{}/count_dataset.pkl'.format(self.dataset),'rb') as f:
            data_dict = pickle.dump(f)