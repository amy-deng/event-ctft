def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import numpy as np
import time
# import utils
import os
from sklearn.utils import shuffle
from models import *
from data import *
import pickle
# from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

 
parser = argparse.ArgumentParser(description='')
parser.add_argument("--dp", type=str, default="../data", help="data path")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
parser.add_argument("--n-hidden", type=int, default=32, help="number of hidden units")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
parser.add_argument("-d", "--dataset", type=str, default='THA_w7h7_minday3', help="dataset to use")
parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument("--max-epochs", type=int, default=20, help="maximum epochs")
parser.add_argument("--seq-len", type=int, default=7)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--rnn-layers", type=int, default=1)
# parser.add_argument("--maxpool", type=int, default=1)
parser.add_argument("--patience", type=int, default=5)
# parser.add_argument("--use-gru", type=int, default=1, help='1 use gru 0 rnn')
# parser.add_argument("--attn", type=str, default='', help='dot/add/genera; default general')
parser.add_argument("--seed", type=int, default=42, help='random seed')
parser.add_argument("--runs", type=int, default=5, help='number of runs')

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
use_cuda = args.gpu >= 0 and torch.cuda.is_available()
print("cuda",use_cuda)
np.random.seed(args.seed)
torch.manual_seed(args.seed) 

train_dataset_loader = StaticGraphData(args.dp, args.dataset,set_name='train')
# valid_dataset_loader = StaticGraphData(args.dp, args.dataset,set_name='valid')
test_dataset_loader = StaticGraphData(args.dp, args.dataset,set_name='test')

train_loader = DataLoader(train_dataset_loader, batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_2)
# valid_loader = DataLoader(valid_dataset_loader, batch_size=1,
#                         shuffle=False, collate_fn=collate_2)
test_loader = DataLoader(test_dataset_loader, batch_size=1,
                        shuffle=False, collate_fn=collate_2)

model = static_heto_graph(h_dim=args.n_hidden)

optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('#params:', total_params)

# for i, batch in enumerate(train_loader):
#     g_data, y_data = batch
#     # g_data = torch.stack(g_data, dim=0)
#     y_data = torch.stack(y_data, dim=0)
#     print(i,y_data.shape,'y_data',y_data)
#     print(len(g_data),'g_data')

    
    # loss = model(batch_data, true_r)


def train(data_loader, dataset_loader=None):
    model.train()
    total_loss = 0
    t0 = time.time()
    for i, batch in enumerate(train_loader):
        g_data, y_data = batch
        # g_data = torch.stack(g_data, dim=0)
        y_data = torch.stack(y_data, dim=0)
        print(i,y_data.shape,'y_data',y_data)
        print(len(g_data),'g_data')
        loss = model(g_data, y_data)
    # for i, batch in enumerate(data_loader):
    #     batch_data, true_s, true_r, true_o = batch
    #     batch_data = torch.stack(batch_data, dim=0)
    #     true_r = torch.stack(true_r, dim=0)
    #     true_s = torch.stack(true_s, dim=0)
    #     true_o = torch.stack(true_o, dim=0)
        # loss = model(batch_data, true_r)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    # t2 = time.time()
    # reduced_loss = total_loss / (dataset_loader.len / args.batch_size)
    # print("Epoch {:04d} | Loss {:.6f} | time {:.2f} {}".format(
    #     epoch, reduced_loss, t2 - t0, time.ctime()))
    # return reduced_loss

train(train_dataset_loader, dataset_loader=None)