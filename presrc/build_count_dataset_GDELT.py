import pandas as pd
import numpy as np
import sklearn
import glob
from datetime import date, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score,balanced_accuracy_score,precision_recall_curve,auc
import sys,os
import pickle

try:
    CONTRY = sys.argv[1]
    # DELTA = int(sys.argv[2])
    WINDOW = int(sys.argv[2])
    HORIZON = int(sys.argv[3])
    PREDWINDOW = int(sys.argv[4]) 
    TARGETEVENT = str(sys.argv[5]) 
except:
    print("Usage: CONTRY(NI,EG), WINDOW, HORIZON, PREDWINDOW, TARGETEVENT (protest(p), riot(r)) (delta=1) ")
    exit()

mydir = '/home/sdeng/workspace/gdelt_data_preprocess/event/'
# file_list = glob.glob(mydir + "*NI*.json")
# file_list = glob.glob(mydir + "*CA*.json")
# country_name = 'EG'
country_name = CONTRY

file_list = glob.glob(mydir + "*{}*.json".format(country_name))
file_list
df_list = []
for f in file_list:
    cur_df = pd.read_json(f,lines=True)
    df_list.append(cur_df)
#     print(cur_df.head())
#     break
    
path = '../data/{}/'.format(CONTRY)
os.makedirs(path, exist_ok=True)
print('path',path)

df = pd.concat(df_list, ignore_index=True)
df['event_date'] = pd.to_datetime(df['event_date'],format='%Y%m%d' )
df.sort_values(by=['event_date'],inplace=True) 
df = df.loc[df['IsRootEvent'] == 1]

df['event_date'] = df.event_date.dt.strftime('%Y-%m-%d')
df = df.loc[df['event_date']>='2015-01-01']


def getRoot(x):
    x = int(x)
    if len(str(x)) == 4: # 1128
        return x // 100
    elif len(str(x)) == 3:
        if x // 10 < 20: # 190
            return x // 10
        else:
            return x // 100
    else:
        return x // 10
    
def movingaverage(a, n=3) :
    padding = []
    for i in range(n-1):
        padding.append(a[:i+1].mean())
    padding = np.array(padding)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.concatenate((padding, ret[n - 1:] / n),0)

df = df.loc[df['EventCode'] != '---'] 
df['RootEventCode'] = df['EventCode'].apply(lambda x: getRoot(x) )
 
start_year = 2015
start_month = 1
start_day = 1
end_year = 2020
end_month = 12
end_day = 31
event_type_column = 'EventCode'
event_type_column = 'RootEventCode'
delta_value = 1
if delta_value == 1:
    level = 'day'
elif delta_value == 7:
    level = 'week'
elif delta_value == 14:
    level = 'biweek'
elif delta_value == 30:
    level = 'month'
subevents = df[event_type_column].unique()
print(len(subevents),subevents)
subevent_count_dict = {}
start_date = date(start_year, start_month, start_day)
end_date = date(end_year, end_month, end_day)
delta = timedelta(days=delta_value)
n_days = 0
last_date = start_date - delta
while start_date <= end_date:
#     print('last_date',last_date,'start_date',start_date )
    last_date = start_date
    start_date += delta
    n_days += 1
print('n_days =',n_days)
# print('n_days =',len(df['event_date'].unique()))
for v in subevents:
    subevent_count_dict[v] = np.array([0 for i in range(n_days)])

riots_count = np.array([0 for i in range(n_days)])

# for loop day.... save count of each subevent.
start_date = date(start_year, start_month, start_day)
end_date = date(end_year, end_month, end_day)
delta = timedelta(days=delta_value)
day_i = 0
last_date = start_date - delta
# print('last_date',last_date,'start_date',start_date,'end_date',end_date)
while start_date <= end_date:
#     print('last_date',last_date,'start_date',start_date )
    last_date_str = last_date.strftime("%Y-%m-%d") #("%d %B %Y")
    date_str = start_date.strftime("%Y-%m-%d")
#     print('last_date_str',last_date_str,' --- date_str',date_str)
    df_day = df.loc[(df['event_date'] > last_date_str) & (df['event_date'] <= date_str)]
    if day_i%200==0:
        print('#',len(df_day),len(df))
#         print(df_day['sub_event_type'] )
    df_count = df_day[event_type_column].value_counts().rename_axis('unique_values').reset_index(name='counts')
#     print('df_count',df_count,df)
    for i,row in df_count.iterrows():
        subevent_count_dict[row['unique_values']][day_i] = row['counts']

    df_riots = df_day.loc[df_day['EventCode'].isin([145,1451,1452,1453,1454])]
    riots_count[day_i] = len(df_riots)
    last_date = start_date
    start_date += delta
    day_i += 1
print('day_i =',day_i)

subevent_count_dict
SUBEVENTS = [i+1 for i in range(20)]

# build sequence data
X = []
for k in SUBEVENTS:
    try:
        v = subevent_count_dict[k].tolist()
        X.append(v)
    except:
        pass
    
X = np.array(X)
X = np.swapaxes(X,0,1)
if TARGETEVENT == 'p':
    print('exit for p as target event')
    Y = subevent_count_dict[14]
    y_threshod = np.percentile(Y, 95)#Y.mean()
    exit()
elif TARGETEVENT == 'r':
    Y = riots_count
    y_threshod = 0
    y_threshod = np.percentile(Y, 70)#Y.mean()
    pass

print('X',X.shape,'Y',Y.shape, 'y_threshod',y_threshod)


ii = 0
data_X = []
data_Y = []
for i in range(0,len(X),HORIZON+PREDWINDOW-1): # no overlap of pre_window
# for i in range(0,len(X),HORIZON): # overlap 1
#     print('x',i,i+window,' y',i+window,i+window+pred_window)
    data_X.append(X[i:i+WINDOW])
    protest = Y[i+WINDOW:i+WINDOW+PREDWINDOW].sum()
#     print(Y[i+window:i+window+pred_window])
#     print(X[i:i+window],Y[i+window:i+window+pred_window-1])
    data_Y.append(1 if protest > y_threshod else 0)
    if i+WINDOW >=len(X) or i+WINDOW+PREDWINDOW-1 >= len(X):
        break
    ii+=1
print(ii)


data_X = np.array(data_X)
data_Y = np.array(data_Y)
print('data_X',data_X.shape,'data_Y',data_Y.shape,'Y mean',data_Y.mean())

all_datasets = {
    'temporal':None,
    'static':None
}

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, stratify=data_Y, test_size=0.25,
                                                    shuffle = True,
                                                    random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
all_datasets['temporal'] = [X_train, X_test, y_train, y_test]

flat_data_X = data_X.reshape(data_X.shape[0],-1)
X_train, X_test, y_train, y_test = train_test_split(flat_data_X, data_Y, stratify=data_Y, test_size=0.25,
                                                    shuffle = True,
                                                    random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
all_datasets['static'] = [X_train, X_test, y_train, y_test]
                       
with open(path+'count_dataset.pkl','wb') as f:
    pickle.dump(all_datasets,f)


# python build_count_dataset_GDELT.py NI 14 1 3 r 