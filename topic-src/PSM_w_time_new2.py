import pandas as pd
import numpy as np
import sys, os, json, time, collections, pickle
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import scipy

'''
python PSM_w_time_new2.py ../data THA_topic check_topic_causal_data_w14h14_from2013_minprob0.05 14 1 0
python PSM_w_time_new2.py ../data THA_topic check_topic_causal_data_w14h14_from2013_minprob0.05 3,7,14 1 0

for each event find causes
'''
# out_path='../data'
# dataset_name='THA_topic'
# raw_data_name='check_topic_causal_data_w14h14_from2013_minprob0.05'
# pred_window=14
# target_binary=1
# check=0
try:
    # event_path = sys.argv[1] # /home/sdeng/data/icews/detailed_event_json/THA_2010_w14h7_city.json
    out_path = sys.argv[1]
    dataset_name = sys.argv[2] # THA_topic
    raw_data_name = sys.argv[3] 
    pred_window = int(sys.argv[4])
    target_binary = int(sys.argv[5])
    check = int(sys.argv[6])
    # event_code = int(sys.argv[4])
except:
    print("usage: <out_path> <dataset_name `THA_topic`> <raw_data_name `check_topic_causal_data_w7h7`> <pred_window 5> <target_binary 0> <check 1/0>")
    exit()


class Net(nn.Module):
    def __init__(self, h_inp, h_hid):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(h_inp, h_hid)  
        self.fc2 = nn.Linear(h_hid, h_hid)
        self.fc3 = nn.Linear(h_hid, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, x, y, optimizer, criterion):
    model.train()
    model.zero_grad()
    output = model(x)
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()
    return loss, output

@torch.no_grad()
def evaluate(model, x, y, optimizer, criterion):
    model.eval()
    # model.zero_grad()
    output = model(x)
    loss = criterion(output,y)
    # loss.backward()
    # optimizer.step()
    return loss, output
 

class OurDataset(data.Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

criterion = nn.BCEWithLogitsLoss()
EPOCHS = 200
BATCH_SIZE = 64


if check == 1:
    file_list = glob.glob('{}/{}/{}/check2_topic*.pkl'.format(out_path, dataset_name, raw_data_name))
else:
    file_list = glob.glob('{}/{}/{}/nocheck2_topic*.pkl'.format(out_path, dataset_name, raw_data_name))

file_list.sort()
save_path = '{}/{}/{}/causal_effect'.format(out_path, dataset_name, raw_data_name)
os.makedirs(save_path, exist_ok=True)
effect_dict = {}
for file in file_list:
    file_name = file.split('/')[-1]
    tmp = file_name.split('.')[0].split('_')
    topic_id = int(tmp[2])
    split_date = tmp[3]
    # topic_id = file_name.split('.')[0].split('_')[-1]
    # with open('{}/{}/{}/topic_{}.pkl'.format(out_path, dataset_name, raw_data_name, topic_id),'rb') as f:
    with open(file,'rb') as f:
        dataset = pickle.load(f)
    treatment = dataset['treatment']
    treatment = treatment
    treatment = np.where(treatment > 0, 1, 0)
    covariate = dataset['covariate']
    covariate = np.concatenate([v.toarray() for v in covariate],0) 
    
    # print("dataset['outcome']",dataset['outcome'].shape)
    # outcome = dataset['outcome'][:,:pred_window,].sum(1) # number of events; sum of all days
    # outcome_sep_day = dataset['outcome'][:,:pred_window,] # number of events; sum of all days
    outcome3 = dataset['outcome'][:,:3,].sum(1)
    outcome7 = dataset['outcome'][:,:7,].sum(1)
    outcome14 = dataset['outcome'][:,:14,].sum(1)

    if target_binary == 1:
        print('Convert outcome to binary')
        outcome3 = np.where(outcome3 > 0, 1, 0)
        outcome7 = np.where(outcome7 > 0, 1, 0)
        outcome14 = np.where(outcome14 > 0, 1, 0)
        # exit()

    print('topic {} data loaded \t {}'.format(topic_id,split_date))
    print('outcome3',outcome3.shape) 
    
    # train propensity scoring function
    # logistic regression
    time1= time.time()
    scaler = StandardScaler()
    X = scaler.fit_transform(covariate)
    print('X',type(X),X.shape)
    print('StandardScaler time',time.time()-time1)
    # """
    # build a nn
    net = Net(X.shape[-1],128)
    net = net.cuda()
    # target = torch.randn(10)  # a dummy target, for example
    # target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.BCEWithLogitsLoss()
    optm = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(treatment).int()
    print(X_torch.type(),'X_torch','y_torch',y_torch.type())
    our_dataset = OurDataset(X_torch,y_torch)
    train_dataloader = DataLoader(our_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for bidx, batch in enumerate(train_dataloader):
            x_train, y_train = batch[0], batch[1]
            x_train = x_train.cuda()
            y_train = y_train.unsqueeze(-1).cuda()
            print(x_train.shape,y_train.shape)
            loss, predictions = train(net,x_train,y_train, optm, criterion)
            epoch_loss+=loss
        print('Epoch {} Loss : {}'.format((epoch+1),epoch_loss))

    net.eval()
    pred = net(X_torch,y_torch)
    pred = torch.sigmoid()
    print('pred.shape',pred.shape)
    propensity = pred
    # """
    # 
    cls = LogisticRegression(random_state=42,max_iter=2000)
    cls = CalibratedClassifierCV(cls)
    cls.fit(X, treatment)
    print('propensity scoring model trained')
    
    propensity = cls.predict_proba(covariate)
    propensity = propensity[:,1]
    print(type(propensity),propensity.shape,'propensity')
    # caliper = propensity.std()*0.2
    propensity_logit = scipy.special.logit(propensity)
    caliper = propensity_logit.std()* 0.2

    # get pairs and calculate average treatment effect 
    # for each treatment ele, find a control, most similar
    controlled_indices = np.where(treatment == 0)[0]
    treatment_idices = treatment.nonzero()[0]
    np.random.shuffle(treatment_idices)
    # treatment_idices
    eff_list3 = [] 
    eff_list7 = []
    eff_list14 = []
    used_control_indices = []
    n_pairs = 0
    for i in treatment_idices:
        curr = propensity_logit[controlled_indices]
        diff = np.abs(curr-propensity_logit[i])
        min_idx = np.argmin(diff, axis=0)
        min_diff = diff[min_idx]
        if min_diff < caliper:
            # get treatment effect?
            # outcome_control = outcome[controlled_indices[min_idx]]
            # outcome_treatment = outcome[i]
            eff = outcome3[controlled_indices[min_idx]]-outcome3[i]
            eff_list3.append(eff) 
            eff = outcome7[controlled_indices[min_idx]]-outcome7[i]
            eff_list7.append(eff) 
            eff = outcome14[controlled_indices[min_idx]]-outcome14[i]
            eff_list14.append(eff) 
            n_pairs += 1
            used_control_indices.append(controlled_indices[min_idx])
        else:
            print('min diff is larger than the caliper {:.5f}; skip'.format(caliper))

    eff_list3 = np.stack(eff_list3,0) 
    eff_list7 = np.stack(eff_list7,0) 
    eff_list14 = np.stack(eff_list14,0) 
    print('eff_list3',eff_list3.shape)
    # ATE3 = eff_list3.mean(0)
    effect_dict[(int(topic_id),split_date)] = [eff_list3.mean(0),eff_list7.mean(0),eff_list14.mean(0)]
    # top3 = ATE.argsort()[-3:][::-1]
    exit()

print(len(effect_dict),'len effect_dict')
with open('{}/effect_dict_pw{}_biy{}_nocheck.pkl'.format(save_path,'3714',target_binary),'wb') as f:
    pickle.dump(effect_dict,f)
print(save_path,'/effect_dict.pkl saved')