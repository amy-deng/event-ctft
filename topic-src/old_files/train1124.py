def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse, os, time, csv, pickle
import numpy as np
from sklearn.utils import shuffle
# from tqdm import tqdm

 
parser = argparse.ArgumentParser(description='')
parser.add_argument("--dp", type=str, default="../data", help="data path")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
parser.add_argument("--n-hidden", type=int, default=32, help="number of hidden units")
parser.add_argument("--n-layers", type=int, default=2, help="number of hidden layers")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight for L2 loss")
parser.add_argument("-d", "--dataset", type=str, default='THA_w7h7_minday3', help="dataset to use")
parser.add_argument("-df", "--datafiles", type=str, default='data_static_2014-02-01_2014-02-05_tt85_sentpmi_1k', help="dataset to use")
parser.add_argument("-cf", "--causalfiles", type=str, default='', help="causality to use")
parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument("--n-epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--seq-len", type=int, default=7)
parser.add_argument("--horizon", type=int, default=7)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--rnn-layers", type=int, default=1)
parser.add_argument("--pool", type=str, default='mean')
parser.add_argument("--patience", type=int, default=15)
# parser.add_argument("--use-gru", type=int, default=1, help='1 use gru 0 rnn')
# parser.add_argument("--attn", type=str, default='', help='dot/add/genera; default general')
parser.add_argument("--seed", type=int, default=999, help='random seed')
parser.add_argument("--runs", type=int, default=5, help='number of runs')
parser.add_argument("-m","--model", type=str, default="m0", help="model name")
parser.add_argument("--train", type=float, default=0.6, help="")
parser.add_argument("--val", type=float, default=0.2, help="")
parser.add_argument('--shuffle', action="store_false")
parser.add_argument("--cau_setup", type=str, default="pos1", help="pos1,sign1,raw")
parser.add_argument("--note", type=str, default="", help="")
parser.add_argument("--n-topics", type=int, default=50, help='number of topics')
parser.add_argument("--n-heads", type=int, default=4, help='number of attention heads')

args = parser.parse_args()
print(args)


os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
# os.environ["CUDA_LAUNCH_BLOCKING"]='1'
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from models import *
from model_gcn import *
from model_gat import *
from model_hgt import *
from model_gcn_hetero import *
from model_han import *
from model_hetero import *
from model_rgcn import *
from model_cau import *
from utils import *
from data import *

use_cuda = args.gpu >= 0 and torch.cuda.is_available()


print("cuda",use_cuda)

torch.manual_seed(args.seed) 
if use_cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')
print('device',device)
with open('{}/{}/word_emb_300.pkl'.format(args.dp,args.dataset), 'rb') as f:
    word_embeds = pickle.load(f)
print('load word_emb_300.pkl')
word_embeds = torch.FloatTensor(word_embeds)
vocab_size = word_embeds.size(0)
emb_size = word_embeds.size(1)


# train_dataset_loader = StaticGraphData(args.dp, args.dataset,set_name='train')
# valid_dataset_loader = StaticGraphData(args.dp, args.dataset,set_name='valid')
# test_dataset_loader = StaticGraphData(args.dp, args.dataset,set_name='test')

static_graph_dataset = StaticGraphDataNew(args.dp, args.dataset,args.datafiles, args.horizon,args.causalfiles)

dataset_size = len(static_graph_dataset)
indices = list(range(dataset_size))
split1 = int(np.floor(args.train * dataset_size))
split2 = int(np.floor((args.val+args.train) * dataset_size))
if args.shuffle:
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]
else:
    test_indices = indices[split2:]
    train_val_indices = indices[:split2]
    np.random.seed(args.seed)
    np.random.shuffle(train_val_indices)
    train_indices, val_indices = train_val_indices[:split1], train_val_indices[split1:split2]
    
# print((train_indices),(val_indices),(test_indices))
print(len(train_indices),len(val_indices),len(test_indices))
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(static_graph_dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_2,sampler=train_sampler)
valid_loader = DataLoader(static_graph_dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_2,sampler=valid_sampler)
test_loader = DataLoader(static_graph_dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_2, sampler=test_sampler)
train_loader.len = len(train_indices)
valid_loader.len = len(val_indices)
test_loader.len = len(test_indices)

# print(train_loader,train_loader.len)

# train_loader = DataLoader(train_dataset_loader, batch_size=args.batch_size,
#                         shuffle=True, collate_fn=collate_2)
# valid_loader = DataLoader(valid_dataset_loader, batch_size=args.batch_size,
#                         shuffle=False, collate_fn=collate_2)
# test_loader = DataLoader(test_dataset_loader, batch_size=args.batch_size,
                        # shuffle=False, collate_fn=collate_2)

def prepare(args,word_embeds,device): 
    if args.model == 'gcn':
        model = GCN(in_feats=emb_size, n_hidden=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, device=device, dropout=args.dropout,pool=args.pool)
    if args.model == 'gat':
        model = GAT(in_feats=emb_size, n_hidden=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, heads=4,device=device, dropout=args.dropout,pool=args.pool)
    elif args.model == 'gcnet':
        model = GCNHet(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, device=device, dropout=args.dropout,pool=args.pool)
    elif args.model == 'gcnetall':
        model = GCNHetAll(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, device=device, dropout=args.dropout,pool=args.pool) 
    elif args.model == 'gcnetall2':
        model = GCNHetAll2(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, device=device, dropout=args.dropout,pool=args.pool)
    elif args.model == 'gcnetall3':
        model = GCNHetAll3(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, device=device, dropout=args.dropout,pool=args.pool)
    elif args.model == 'gcnetall4':
        model = GCNHetAll4(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, device=device, dropout=args.dropout,pool=args.pool) 
    elif args.model == 'han':
        model = HAN(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, device=device, dropout=args.dropout,pool=args.pool) 
    elif args.model == 'hanall':
        model = HANAll(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, device=device, dropout=args.dropout,pool=args.pool) 
    elif args.model == 'hetero':
        model = Hetero(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, device=device, dropout=args.dropout,pool=args.pool) 
    elif args.model == 'heteroall':
        model = HeteroAll(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, device=device, dropout=args.dropout,pool=args.pool) 
    elif args.model == 'rgcnall':
        model = RGCNAll(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, device=device, dropout=args.dropout,pool=args.pool) 
    elif args.model == 'topic':
        model = static_topic_graph(h_inp=emb_size, vocab_size=vocab_size, h_dim=args.n_hidden, device=device, pool=args.pool)
    elif args.model == 'tcau0':
        model = static_topic_cau0(h_inp=emb_size, vocab_size=vocab_size, h_dim=args.n_hidden, device=device, pool=args.pool)
    elif args.model == 'hetero2':
        model = static_heto_graph2(h_inp=emb_size, vocab_size=vocab_size, h_dim=args.n_hidden, device=device, pool=args.pool)
    elif args.model == 'cau0':
        model = static_heto_cau0(h_inp=emb_size, vocab_size=vocab_size, h_dim=args.n_hidden, device=device, pool=args.pool)
    elif args.model == 'cau11':
        model = static_heto_cau1(h_inp=emb_size, vocab_size=vocab_size, h_dim=args.n_hidden, device=device, pool=args.pool) 
    elif args.model == 'word':
        model = static_word_graph(h_inp=emb_size, vocab_size=vocab_size, h_dim=args.n_hidden, device=device,pool=args.pool)
    elif args.model == 'word2':
        model = static_word_graph2(h_inp=emb_size, vocab_size=vocab_size, h_dim=args.n_hidden, device=device,pool=args.pool)
    elif args.model == 'temp_hetero':
        model = temp_heto_graph(h_inp=emb_size, vocab_size=vocab_size, h_dim=args.n_hidden, device=device,pool=args.pool)
    elif args.model == 'temp_word':
        model = temp_word_graph2(h_inp=emb_size, vocab_size=vocab_size, h_dim=args.n_hidden, device=device,pool=args.pool)
    elif args.model == 'hgt':
        model = HGT(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=args.n_heads, device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout,pool=args.pool, use_norm = True)
    elif args.model == 'hgtall':
        model = HGTAll(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=args.n_heads, device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout,pool=args.pool, use_norm = True)
    elif args.model == 'hgtallcau':
        model = HGTAllcau(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=args.n_heads, device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout,pool=args.pool, use_norm = True)
    elif args.model == 'temphgtall':
        model = TempHGTAll(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=4, seq_len=args.seq_len,device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout,pool=args.pool, use_norm = True)
    elif args.model == 'ours':
        model = tempMP(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=args.n_heads, activation=F.relu, seq_len=args.seq_len,device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout,pool=args.pool, use_norm = True)
    # elif args.model == 'ours2':
    #     model = tempMP2(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=4, activation=F.relu, seq_len=args.seq_len,device=device, 
    #     num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout,pool=args.pool, use_norm = True)
    elif args.model == 'ours3':
        model = tempMP3(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=args.n_heads, activation=F.relu, seq_len=args.seq_len,device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout,pool=args.pool, use_norm = True)
    elif args.model == 'ours4':
        model = tempMP4(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=args.n_heads, activation=F.relu, seq_len=args.seq_len,device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout,pool=args.pool, use_norm = True)
    elif args.model == 'ours6':
        model = tempMP6(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=args.n_heads, activation=F.relu, seq_len=args.seq_len,device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout,pool=args.pool, use_norm = True)
    elif args.model == 'cau6':
        model = tempMP6cau(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=args.n_heads, activation=F.relu, seq_len=args.seq_len,device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout,pool=args.pool, use_norm = True)
        # model = static_hgt(h_inp=emb_size, vocab_size=vocab_size, h_dim=args.n_hidden, device=device,pool=args.pool)
    # elif args.model == 'temp_word_hetero':
    #     model = temp_word_hetero(h_inp=emb_size, vocab_size=vocab_size, h_dim=args.n_hidden, device=device,pool=args.pool)
    model_name = model.__class__.__name__
    # print(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#params:', total_params)
    token = '{}_seed{}topic{}w{}h{}lr{}bs{}p{}hid{}l{}tr{}va{}po{}nh{}'.format(model_name, args.seed, args.n_topics, args.seq_len, args.horizon,args.lr,args.batch_size,args.patience,args.n_hidden,args.n_layers,args.train,args.val,args.pool,args.n_heads)
    if args.shuffle is False:
        token += '_noshuf'
    if args.note != "":
        token += args.note
    # if args.model in ['cus3','cus4']:
    #     token += args.cau_setup
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/' + args.dataset, exist_ok=True)
    os.makedirs('models/{}/{}'.format(args.dataset, token), exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/' + args.dataset, exist_ok=True)

    result_file = 'results/{}/{}.csv'.format(args.dataset,token)
    model_state_file = 'models/{}/{}.pth'.format(args.dataset, token)
    # model_graph_file = 'models/{}/{}_graph.pth'.format(args.dataset, token)
    # outf = 'models/{}/{}.result'.format(args.dataset, token)

    if use_cuda:
        model.cuda()
        word_embeds = word_embeds.cuda()
    model.word_embeds = word_embeds
    return model, optimizer, result_file, token
 

epoch = 0
def train(train_loader):
    model.train()
    total_loss = 0
    t0 = time.time()
    for i, batch in enumerate(train_loader):
        g_data, y_data = batch
        # g_data = torch.stack(g_data, dim=0)
        y_data = torch.stack(y_data, dim=0).to(device)
        # print(i,y_data.shape,'y_data',y_data)
        # print(len(g_data),'g_data')
        loss, y_pred = model(g_data, y_data) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    t2 = time.time()
    reduced_loss = total_loss / (train_loader.len / args.batch_size)
    print("Epo {:04d} | Loss {:.6f} | time {:.2f} {}".format(epoch, reduced_loss, t2 - t0, time.ctime()))
    return reduced_loss

@torch.no_grad()
def eval(data_loader, set_name='valid'):
    model.eval()
    y_true_l, y_pred_l = [], []
    total_loss = 0
    for i, batch in enumerate(data_loader):
        g_data, y_data = batch
        # g_data = torch.stack(g_data, dim=0)
        y_data = torch.stack(y_data, dim=0).to(device)
        loss, y_pred = model(g_data, y_data) 
        y_true_l.append(y_data)
        y_pred_l.append(y_pred)
        total_loss += loss.item()

    # print('{} results'.format(set_name)) 
    y_true_l = torch.cat(y_true_l,0).cpu().detach().numpy() 
    # print(y_true_l.shape,'y_true_l')
    y_pred_l = torch.cat(y_pred_l,0).cpu().detach().numpy() 
    eval_dict = eval_bi_classifier(y_true_l, y_pred_l)
    reduced_loss = total_loss / (data_loader.len / args.batch_size)
    # print("{} Loss: {:.6f}".format(set_name, reduced_loss))
    return reduced_loss, eval_dict

 

for i in range(args.runs):
    model, optimizer, result_file, token = prepare(args, word_embeds, device)
    print('============== Run i = {} on Dataset {} {} =============='.format(i,args.dataset,token))
    # result_file = 'results/{}/{}.csv'.format(args.dataset,token)
    model_state_file = 'models/{}/{}/{}.pth'.format(args.dataset, token, i)
    if i == 0 and os.path.exists(result_file):  # if result_file exist
        os.remove(result_file)
        
    bad_counter = 0
    loss_small = float('inf')
    value_large = float('-inf')
    try:
        print('begin training ...')
        for epoch in range(0, args.n_epochs):
            epoch_start_time = time.time()
            train_loss = train(train_loader)
            valid_loss, eval_dict = eval(valid_loader, set_name='val')
            small_value = valid_loss 
            if small_value < loss_small:
                loss_small = small_value
                bad_counter = 0
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                print('Epo {:04d} tr_los:{:.5f} val_los:{:.5f} '.format(epoch, train_loss, valid_loss),'|'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))
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
    val_loss, eval_dict = eval(valid_loader, 'val')
    print('Val','|'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))
    # val_res = [eval_dict[k] for k in eval_dict]

    _, eval_dict = eval(test_loader, 'test')
    print('Test','|'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))
    test_res = [eval_dict[k] for k in eval_dict]
    wrt.writerow([val_loss] + [0] + test_res)
    f.close()


with open(result_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    arr = []
    for row in csv_reader:
        arr.append(list(map(float, row)))
arr = np.array(arr)
# print('arr',arr,arr.shape)
# arr = np.nan_to_num(arr)
line_count = arr.shape[0]
mean = [round(float(v),3) for v in arr.mean(0)]
std = [round(float(v),3) for v in arr.std(0)]
res = [str(mean[i]) +' ' + str(std[i]) for i in range(len(mean))]
print(res)


all_res_file = 'results/{}/{}.csv'.format(args.dataset,args.model)
# all_res_file = 'results/{}/{}.csv'.format(args.dataset,args.res)
f = open(all_res_file,'a')
wrt = csv.writer(f)
wrt.writerow([token] + [line_count] + res)
f.close()
print(token)
 