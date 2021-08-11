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
    RAWDATA = sys.argv[1]
    # DELTA = int(sys.argv[2])
    WINDOW = int(sys.argv[2])
    HORIZON = int(sys.argv[3])
    PREDWINDOW = int(sys.argv[4]) 
    TARGETEVENT = str(sys.argv[5]) 
except:
    print("Usage: DATASET, WINDOW, HORIZON, PREDWINDOW, TARGETEVENT (protest(p), riot(r)) (delta=1) ")
    exit()


DATASET = RAWDATA.split('-')[7][:-4]
print(DATASET,'dataset_name')

filename = RAWDATA.split('/')[-1]
print('path',RAWDATA,'filename',filename)
start_year = int(filename.split('-')[0])
start_month = int(filename.split('-')[1])
start_day = int(filename.split('-')[2])
end_year = int(filename.split('-')[3])
end_month = int(filename.split('-')[4])
end_day = int(filename.split('-')[5])


path = '../data/{}/'.format(DATASET)
os.makedirs(path, exist_ok=True)
print('path',path)

df = pd.read_csv(RAWDATA,sep=';')
df = df.drop_duplicates(subset=['data_id'], keep='first')
df['event_date'] = pd.to_datetime(df['event_date'])
print(df.columns)
df.sort_values(by=['event_date'],inplace=True ) 

event_type_column = 'sub_event_type'
# event_type_column = 'event_type'

subevents = df[event_type_column].unique()
print(len(subevents),subevents)
delta_value = 1
if delta_value == 1:
    level = 'day'
elif delta_value == 7:
    level = 'week'
elif delta_value == 14:
    level = 'biweek'
elif delta_value == 30:
    level = 'month'

subevent_count_dict = {}
start_date = date(start_year, start_month, start_day)
end_date = date(end_year, end_month, end_day)
delta = timedelta(days=delta_value)
n_days = 0
last_date = start_date - delta
while start_date <= end_date:
    last_date = start_date
    start_date += delta
    n_days += 1
print('n_days =',n_days)
for v in subevents:
    subevent_count_dict[v] = np.array([0 for i in range(n_days)])


start_date = date(start_year, start_month, start_day)
end_date = date(end_year, end_month, end_day)
delta = timedelta(days=delta_value)
day_i = 0
last_date = start_date - delta
while start_date <= end_date:
    last_date_str = last_date.strftime("%Y-%m-%d") #("%d %B %Y")
    date_str = start_date.strftime("%Y-%m-%d")
    df_day = df.loc[(df['event_date'] > last_date_str) & (df['event_date'] <= date_str)]
    if day_i%300==0:
        print('#',day_i,len(df_day),len(df))
    df_count = df_day[event_type_column].value_counts().rename_axis('unique_values').reset_index(name='counts')
    for i,row in df_count.iterrows():
        subevent_count_dict[row['unique_values']][day_i] = row['counts']
    last_date = start_date
    start_date += delta
    day_i += 1
print('day_i =',day_i)

def movingaverage(a, n=3) :
    padding = []
    for i in range(n-1):
        padding.append(a[:i+1].mean())
    padding = np.array(padding)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.concatenate((padding, ret[n - 1:] / n),0)


if TARGETEVENT == 'p':
    event_set_protest = ['Protest with intervention','Excessive force against protesters','Peaceful protest']
    subevent_count_dict['Protests'] = subevent_count_dict['Protest with intervention'] + subevent_count_dict['Peaceful protest'] + subevent_count_dict['Excessive force against protesters']
    # del subevent_count_dict['Protest with intervention']
    # del subevent_count_dict['Excessive force against protesters']
    # del subevent_count_dict['Peaceful protest']
elif TARGETEVENT == 'r':
    event_set_protest = ['Mob violence','Violent demonstration']
    subevent_count_dict['Protests'] = subevent_count_dict['Mob violence'] + subevent_count_dict['Violent demonstration'] 
else:
    print('please choose p or r for TARGETEVENT')
    exit()


SUBEVENTS = ['Abduction/forced disappearance', 'Agreement', 'Air/drone strike',
       'Armed clash', 'Arrests', 'Attack', 'Change to group/activity',
       'Chemical weapon', 'Disrupted weapons use',
       'Excessive force against protesters',
       'Government regains territory', 'Grenade',
       'Headquarters or base established', 'Looting/property destruction',
       'Mob violence', 'Non-state actor overtakes territory',
       'Non-violent transfer of territory', 'Other', 'Peaceful protest',
       'Protest with intervention', 'Remote explosive/landmine/IED',
       'Sexual violence', 'Shelling/artillery/missile attack',
       'Suicide bomb', 'Violent demonstration']


# build sequence data, consider all
X = []
for k in SUBEVENTS:
    try:
        v = subevent_count_dict[k].tolist()
        X.append(v)
    except:
        pass
    
X = np.array(X)
X = np.swapaxes(X,0,1)
Y = subevent_count_dict['Protests']

 
ii = 0
data_X = []
data_Y = []
data_time = []
for i in range(0,len(X),HORIZON+PREDWINDOW-1): # no overlap of pre_window
    if i+WINDOW >=len(X) or i+WINDOW+PREDWINDOW-1 >= len(X):
        break
# for i in range(0,len(X),HORIZON): # overlap 1
#     print('x',i,i+window,' y',i+window,i+window+pred_window)
    data_X.append(X[i:i+WINDOW])
    protest = Y[i+WINDOW:i+WINDOW+PREDWINDOW].sum()
#     print(Y[i+window:i+window+pred_window])
#     print(X[i:i+window],Y[i+window:i+window+pred_window-1])
    data_Y.append(1 if protest > 0 else 0)
    # if i+WINDOW >=len(X) or i+WINDOW+PREDWINDOW-1 >= len(X):
    #     break
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
