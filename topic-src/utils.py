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
 

def eval_bi_classifier(y_true, y_score, y_pred_bi=None):
    y_true = y_true.reshape(-1)
    y_score = y_score.reshape(-1)
    r = {}
    try:
        r['auroc'] = metrics.roc_auc_score(y_true, y_score)
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
        aupr = metrics.auc(recall, precision)
        r['aupr'] = aupr
    except:
        pass
    if y_pred_bi is None:# or True:
        y_pred_bi = np.where(y_score>0.5,1,0)
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred_bi, average="binary")  
    # acc = metrics.accuracy_score(y_true, y_pred_bi)
    r['prec'] = prec
    r['rec'] = rec
    r['f1'] = f1
    r['bacc'] = metrics.balanced_accuracy_score(y_true,y_pred_bi)
    r['acc'] = metrics.accuracy_score(y_true,y_pred_bi)
    return r