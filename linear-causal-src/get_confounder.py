# TODO use the trained model to output the embedding

# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import time
import os
import numpy as np
import random
import argparse

'''
python train_freq.py -m nei_weight --loop 1
'''
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='')
parser.add_argument('-d','--dataset', type=str, default='THA')
parser.add_argument('--datafile', type=str, default='collab-t10-2010-freq.pkl')
parser.add_argument('-m','--model', type=str, default='gruf', help='')
parser.add_argument('--loop', type=int, default=10)
parser.add_argument('--aggr_feat', action="store_true")
# parser.add_argument('--treat_idx', type=int, default=0, help='not used')

parser.add_argument('--outdir', type=str, default="search_results")
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--epochs_ft', type=int, default=50, help='Number of epochs to fine-tune')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters)')
# parser.add_argument('--h_dim', type=int, default=128, help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability)')
# parser.add_argument('--alpha', type=float, default=1e-4, help='trade-off of representation balancing')
# parser.add_argument('--clip', type=float, default=100., help='gradient clipping')
parser.add_argument("-b",'--batch', type=int, default=64)
parser.add_argument('-p','--patience', type=int, default=25)
parser.add_argument('--train', type=float, default=0.7)
parser.add_argument('--val', type=float, default=0.15)
parser.add_argument('--test', type=float, default=0.15)
# parser.add_argument('--p_alpha', type=float, default=0.01)
parser.add_argument('--rep_layer', type=int, default=2)
parser.add_argument('--hyp_layer', type=int, default=2)
parser.add_argument('--rep_dim', type=int, default=64)
parser.add_argument('--hyp_dim', type=int, default=64)
# parser.add_argument('--imb_func', type=str, default='mmd')

parser.add_argument('-w','--window', type=int, default=7)
parser.add_argument('-ho','--horizon', type=int, default=2)
parser.add_argument('-pw','--pred_window', type=int, default=2)

parser.add_argument('--enc', type=str, default='gru') # or gru

parser.add_argument('--realy', action="store_true", help='real value comes with normalization')
parser.add_argument('--shuffle', action="store_false")

parser.add_argument('--token', type=str)
parser.add_argument('--i', type=int)


args = parser.parse_args()
assert args.val > .0, print('args.val should be greater than 0')

if args.model in ['cevae']:
    args.h_dim = 64

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal
import pandas as pd
import csv
from models import *
from utils import *

args.cuda = args.gpu >= 0 and torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor

# alpha = Tensor([args.alpha])

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    # alpha = alpha.cuda()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('args.device:',args.device,' args.cuda:',args.cuda)
 
data_loader = DataLoaderFreq(args)
# data_loader = DataLoaderFreqPropensity(args)
# print('check data. exit()')
# exit()
os.makedirs('models', exist_ok=True)
os.makedirs('models/' + args.dataset, exist_ok=True)

os.makedirs('results', exist_ok=True)
os.makedirs('results/' + args.dataset, exist_ok=True)

# args.treat_idx = 0
os.makedirs(args.outdir, exist_ok=True)
search_path = "{}/{}/{}_w{}h{}p{}".format(args.outdir,args.dataset,args.model,args.window,args.horizon,args.pred_window)
os.makedirs(search_path, exist_ok=True)

def prepare(args): 
    if args.model == 'gruf':
        args.enc = 'gru'
        model = DNN_F(args, data_loader)
    elif args.model == 'nei_mean':
        model = Nei_mean(args, data_loader)
    elif args.model == 'nei_weight':
        model = Nei_weight(args, data_loader)
    # elif args.model == 'nei_p':
    #     model = Nei_p(args, data_loader)
    else: 
        raise LookupError('can not find the model')
    model_name = model.__class__.__name__
    # print(model)
    token = args.model + '-lr'+str(args.lr)[1:] +  'w' + str(args.window) + 'h'+str(args.horizon) + 'pw'+str(args.pred_window)  \
        + 'agg'+str(int(args.aggr_feat)) + 'rep'+str(args.rep_layer) + '*'+str(args.rep_dim) + 'hyp'+str(args.hyp_layer) +'*'+ str(args.hyp_dim) 
        # + 'enc'+str(args.enc) 
 
    os.makedirs('models/{}/{}'.format(args.dataset, token), exist_ok=True)
    result_file = 'results/{}/{}.csv'.format(args.dataset,token)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model:{} \t # params {}'.format(model_name,total_params))
    print('Token:', token)
    if args.cuda:
        model.cuda()
    model = model.double()
    if args.shuffle:
        data_loader.shuffle()
        print('<<< data shuffled >>>')
    return model, optimizer, result_file, token


def getEmb(data_loader, data, tag='val'):
    model.eval()
    # n_samples = 0.
    # total_loss = 0.  
    # y_true, y_pred = [], []
    emb_list = []
    for inputs in data_loader.get_batches(data, args.batch, False):
        [X, Y] = inputs 
        # if args.model in ['dnnf']:
        # loss, y  = model(X, Y) 
        emb = model.getConfounder(X)
        emb_list.append(emb.cpu().detach().numpy())
    emb_list = np.concatenate(emb_list)
    print(emb_list.shape,'emb_list',type(emb_list))
    return emb_list

 

# for i in range(args.loop):
#     print('============== Loop i = {} on Dataset {} =============='.format(i,args.dataset))

model, optimizer, result_file, token = prepare(args)
model_state_file = 'models/{}/{}/{}.pth'.format(args.dataset, args.token, args.i)
print('model_state_file',model_state_file)
checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])

print("getEmb using best epoch: {}".format(checkpoint['epoch']))

train_emb = getEmb(data_loader, data_loader.train, 'train')

val_emb = getEmb(data_loader, data_loader.val, 'val')

test_emb = getEmb(data_loader, data_loader.test, 'test')

all_confounder_emb = np.concatenate((train_emb,val_emb,test_emb))
print('all_confounder_emb',all_confounder_emb.shape,type(all_confounder_emb))
with open('{}/{}/nei_weight_emb.pkl'.format(args.data_path, args.dataset), 'wb') as f:
    pkl.dump(all_confounder_emb,f)
# # cauculate mean and std, and save it to res_stat
# with open(result_file, 'r') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     arr = []
#     for row in csv_reader:
#         arr.append(list(map(float, row))) 
# arr = np.array(arr)
# arr = np.nan_to_num(arr)
# line_count = arr.shape[0]
# mean = [round(float(v),3) for v in arr.mean(0)]
# std = [round(float(v),3) for v in arr.std(0)]
# res = [str(mean[i]) +' ' + str(std[i]) for i in range(len(mean))]
# print(res)

# # os.remove(result_file)

# all_res_file = '{}/search_{}_{}.csv'.format(search_path,args.dataset,args.model)
# f = open(all_res_file,'a')
# wrt = csv.writer(f)
# wrt.writerow([token] + [line_count] + res)
# f.close()