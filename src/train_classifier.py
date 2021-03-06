# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import time
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='')
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('-d','--dataset', type=str, default='Afghanistan')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--h_dim', type=int, default=128, help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability)')
# parser.add_argument('--alpha', type=float, default=1e-4, help='trade-off of representation balancing')
parser.add_argument('--clip', type=float, default=100., help='gradient clipping')
parser.add_argument("-b",'--batch', type=int, default=64)
parser.add_argument('-w','--window', type=int, default=10)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--pred_window', type=int, default=3)
parser.add_argument('-p','--patience', type=int, default=15)
parser.add_argument('--train', type=float, default=0.6)
parser.add_argument('--val', type=float, default=0.2)
parser.add_argument('--test', type=float, default=0.2)
parser.add_argument('-m','--model', type=str, default='ols1', help='deconf')
parser.add_argument('--loop', type=int, default=2)
parser.add_argument('--realy', action="store_true", help='real value comes with normalization')
# parser.add_argument('--normy', action="store_false")
parser.add_argument('--shuffle', action="store_false")
parser.add_argument('--aggr_feat', action="store_true")

# parser.add_argument('--perloc', action="store_true", help='evaluate per location')
# parser.add_argument('--usegeo', action="store_true")
# parser.add_argument('--generate', action="store_true")
# parser.add_argument('--treat_idx', type=int, default=24, help='index of treatment')
# parser.add_argument('--rdm', action="store_true", help="apply randomization for evaluation")
# parser.add_argument('--mainfeat', action="store_true", help="main events as features")
# parser.add_argument('--noise', type=float, default=0.0, help="train possion noise averag")
# parser.add_argument('--tn', type=float, default=0.0, help="test noise possion noise averag")
# parser.add_argument('--dyn', action="store_true")
# parser.add_argument('--cri', type=str, default="p_r0", help="criteria for early stopping [p_r0,att_err]")
# parser.add_argument('--sig', action="store_true", help="significant increase in treatment")
# parser.add_argument('--ttype', type=int, default=1, help="treatment type, 0:Inc. 1:50%Inc. 2: 50%Dec")
# parser.add_argument('--hw', type=int, default=0, help="predict protest in a future window h=1-3 [horizon=1]")
# parser.add_argument('--cf', action="store_true", help="cf")
parser.add_argument('--balance1', type=float, default=0.01)
parser.add_argument('--balance2', type=float, default=0.001)
parser.add_argument('--rep_layer', type=int, default=2)
parser.add_argument('--hyp_layer', type=int, default=2)
parser.add_argument('--rep_dim', type=int, default=100)
parser.add_argument('--hyp_dim', type=int, default=100)


args = parser.parse_args()
assert args.val > .0, print('args.val should be greater than 0')
 

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal
import pandas as pd
import csv
from cevae_net import p_x_z, p_t_z, p_y_zt, q_t_x, q_y_xt, q_z_tyx, init_qz, CEVAE
from models import *
from utils import *

args.cuda = args.gpu >= 0 and torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor

# alpha = Tensor([args.alpha])

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    # alpha = alpha.cuda()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('args.device:',args.device,' args.cuda:',args.cuda)
 

# if args.model == 'site':
#     print('dataloader for SITE TODO')
#     pass
#     # data_loader = EventDataSITEBasicLoader(args)
# else:
data_loader = CountFactDataLoader(args)


os.makedirs('models', exist_ok=True)
os.makedirs('models/' + args.dataset, exist_ok=True)

os.makedirs('results', exist_ok=True)
os.makedirs('results/' + args.dataset, exist_ok=True)


def prepare(args):
    if args.model == 'dnn':
        model = DNN(data_loader.f, rep_hid=args.rep_dim, hyp_hid=args.hyp_dim, rep_layer=args.rep_layer, hyp_layer=args.hyp_layer, binary=(not args.realy),  dropout=args.dropout, device=args.device)
    else:
        raise LookupError('can not find the model')
    model_name = model.__class__.__name__
    # print(model)
    # token = args.model + '-lr'+str(args.lr)[1:] + 'wd'+str(args.weight_decay) + 'hd' + str(args.h_dim) \
    #     + 'dp' + str(args.dropout)[1:] \
    #     + 'b' + str(args.batch) + 'w' + str(args.window) + 'h'+str(args.horizon) + 'pw'+str(args.pred_window) + 'p' + str(args.patience) \
    #     + 'tr'+str(args.train)[1:] + 'va'+str(args.val)[1:] + 'agg'+str(args.aggr_feat) +'treat'+str(args.treat_idx)
    token = args.model + '-lr'+str(args.lr)[1:] +  'w' + str(args.window) + 'h'+str(args.horizon) + 'pw'+str(args.pred_window)  \
        + 'agg'+str(args.aggr_feat) + 'b1'+str(args.balance1) + 'rep'+str(args.rep_layer) + '*'+str(args.rep_dim) + 'hyp'+str(args.hyp_layer) +'*'+ str(args.hyp_dim) 
    

    os.makedirs('models/{}/{}'.format(args.dataset, token), exist_ok=True)
    result_file = 'results/{}/{}.csv'.format(args.dataset,token)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model:{} \t # params {}'.format(model_name,total_params))
    print('Token:', token)
    if args.cuda:
        model.cuda()
    data_loader.realization_and_split(args.train,args.val,args.test)
 
    print('<<< model and data are ready >>>')
    return model, optimizer, result_file, token
    # TODO train file, model file, data, run results, treatment effect


def eval(data_loader, data, tag='val'):
    model.eval()
    total = 0.
    n_samples = 0.
    total_loss = 0.  
    y_true, y_pred = [], []
    for inputs in data_loader.get_batches(data, args.batch, False):
        [Y, X] = inputs 
        if args.model in ['dnn']:
            loss, y = model(X, Y)
        total_loss += loss.item()
        # n_samples += (Y.view(-1).size(0))
        y_true.append(Y)
        y_pred.append(y)
        n_samples += 1
        # eval
    y_true = torch.cat(y_true).cpu().detach().numpy() 
    y_pred = torch.cat(y_pred).cpu().detach().numpy() 
    # print(y1_true.shape,'y1_true', y1_pred.shape,'y1_pred')
    eval_dict = eval_classifier(y_true, y_pred)
    return float(total_loss / n_samples), eval_dict


def train(data_loader, data, epoch, tag='train'):
    model.train()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    total_loss = 0.
    n_samples = 0.
    for inputs in data_loader.get_batches(data, args.batch, True):
        [Y, X]   = inputs
        if args.model in ['dnn']:
            loss, y_pred = model(X, Y)
        
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward() 
        # torch.nn.utils.clip_grad_norm(model.parameters(),args.clip)
        optimizer.step()
        # n_samples += (Y_norm.view(-1).size(0))
        n_samples += 1
    return float(total_loss / n_samples)


for i in range(args.loop):
    print('============== Loop i = {} on Dataset {} =============='.format(i,args.dataset))
    model, optimizer, result_file, token = prepare(args)
    model_state_file = 'models/{}/{}/{}.pth'.format(args.dataset, token, i)
    if i == 0 and os.path.exists(result_file):  # if result_file exist
        os.remove(result_file)
    
    bad_counter = 0
    stop_criteria = float('inf') # loss as criteria for early stop
    try:
        print('Begin training');
        for epoch in range(0, args.epochs):
            train_loss = train(data_loader, data_loader.train, epoch)
            valid_loss, eval_dict = eval(data_loader, data_loader.val, tag='val')
            if valid_loss < stop_criteria:
                stop_criteria = valid_loss
                bad_counter = 0
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                print('Epo {} tr_los:{:.5f} val_los:{:.5f} '.format(epoch, train_loss, valid_loss),'\t'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
        print("training done")
            
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early, epoch',epoch)
    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    f = open(result_file,'a')
    wrt = csv.writer(f)
    print("Test using best epoch: {}".format(checkpoint['epoch']))

    val_loss, eval_dict = eval(data_loader, data_loader.val, 'val')
    print('Val      ','\t'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))
 

    _, eval_dict = eval(data_loader, data_loader.test, 'test')
    print('Test     ','\t'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))
    test_res = [eval_dict[k] for k in eval_dict]
    wrt.writerow([val_loss] + [0] + test_res)
    f.close()

# cauculate mean and std, and save it to res_stat
with open(result_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    arr = []
    for row in csv_reader:
        arr.append(list(map(float, row))) 
arr = np.array(arr)
arr = np.nan_to_num(arr)
line_count = arr.shape[0]
mean = [round(float(v),3) for v in arr.mean(0)]
std = [round(float(v),3) for v in arr.std(0)]
res = [str(mean[i]) +' ' + str(std[i]) for i in range(len(mean))]
print(res)

all_res_file = 'results/{}/res_stat_{}_{}.csv'.format(args.dataset,args.dataset,args.model)
f = open(all_res_file,'a')
wrt = csv.writer(f)
wrt.writerow([token] + [line_count] + res)
f.close()