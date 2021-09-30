import pandas as pd
import numpy as np
import sys, os, json, time, collections, pickle
# import glob
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

import scipy
'''
python propensity_score_matching.py ../data THA_topic raw_w7h7 0
'''
try:
    # event_path = sys.argv[1] # /home/sdeng/data/icews/detailed_event_json/THA_2010_w14h7_city.json
    out_path = sys.argv[1]
    dataset_name = sys.argv[2] # THA_topic
    raw_data_name = sys.argv[3] 
    topic_id = int(sys.argv[4])
except:
    print("usage: <out_path> <dataset_name `THA_topic`> <raw_data_name `raw_w10h7`> <topic_id>")
    exit()


with open('{}/{}/{}/topic_{}.pkl'.format(out_path, dataset_name, raw_data_name, topic_id),'rb') as f:
    dataset = pickle.load(f)



plot_path = '{}/{}/{}/plot'.format(out_path, dataset_name, raw_data_name)
os.makedirs(plot_path, exist_ok=True)
# exit()

treatment = dataset['treatment']
treatment = treatment
treatment = np.where(treatment > 0, 1, 0)

covariate = dataset['covariate']
covariate = np.concatenate([v.toarray() for v in covariate],0)

outcome = dataset['outcome'].sum(1) # number of events; sum of all days
# not binary vector
outcome_sep_day = dataset['outcome'] # number of events; sum of all days
print('data loaded')

# train propensity scoring function
# logistic regression
scaler = StandardScaler()
X = scaler.fit_transform(covariate)
print(X.shape)

cls = LogisticRegression(random_state=42,max_iter=2000)
cls = CalibratedClassifierCV(cls)
cls.fit(X, treatment)
print('propensity scoring model trained')

propensity = cls.predict_proba(covariate)
propensity = propensity[:,1]

# caliper = propensity.std()*0.2
propensity_logit = scipy.special.logit(propensity)
caliper = propensity_logit.std()* 0.2

# get pairs and calculate average treatment effect 
# for each treatment ele, find a control, most similar
controlled_indices = np.where(treatment == 0)[0]
treatment_idices = treatment.nonzero()[0]
np.random.shuffle(treatment_idices)
# treatment_idices

eff_list = []
all_outcome_treatment = []
all_control_outcome = []
all_control_outcome_day = []
all_treatment_outcome_day = []
used_control_indices = []
n_pairs = 0
for i in treatment_idices:
    curr = propensity_logit[controlled_indices]
    diff = np.abs(curr-propensity_logit[i])
    min_idx = np.argmin(diff, axis=0)
    min_diff = diff[min_idx]
    if min_diff < caliper:
        # get treatment effect?
        outcome_control = outcome[controlled_indices[min_idx]]
        outcome_treatment = outcome[i]
        # print('outcome_control',outcome_control,'outcome_treatment',outcome_treatment)
        eff = outcome_treatment-outcome_control

        eff_list.append(eff)
        all_outcome_treatment.append(outcome_treatment)
        all_control_outcome.append(outcome_control)
        all_control_outcome_day.append(outcome_sep_day[controlled_indices[min_idx]])
        all_treatment_outcome_day.append(outcome_sep_day[i])
        n_pairs += 1
        used_control_indices.append(controlled_indices[min_idx])
    else:
        print('min diff is larger than the caliper {:.5f}; skip'.format(caliper))

eff_list = np.stack(eff_list,0)
all_control_outcome = np.stack(all_control_outcome,0)
all_outcome_treatment = np.stack(all_outcome_treatment,0)
all_control_outcome_day = np.stack(all_control_outcome_day,0)
all_treatment_outcome_day = np.stack(all_treatment_outcome_day,0)

all_control_outcome_max = all_control_outcome.max(0)
all_outcome_treatment_max = all_outcome_treatment.max(0)
all_control_outcome_day_max = all_control_outcome_day.max(0)
all_treatment_outcome_day_max = all_treatment_outcome_day.max(0)

ATE = eff_list.mean(0)
top3 = ATE.argsort()[-3:][::-1]
# ate 
fig, axes = plt.subplots(7, 3, figsize=(10,16), sharex=True, sharey=False)
labels = ['statement', 'appeal','express cooperate','consult','diplomatic cooperation','material cooperation','provide aid','yield','investigate','demand','disapprove','reject','threaten','protest','minitary','reduce relation','coerce','assault','fight','mass violence']
for i, ax in enumerate(axes.flatten()):
    if i >= 20:
        break
    fig.add_subplot(ax)
    if ATE[i]>0:
        lw = 2
        title_color = 'red'
    else:
        lw = 1
        title_color = 'black'
    if i in top3:
        top = '*'
    else:
        top = ''
    ax.set_title(top+labels[i]+top+ ' ATE={:.4f}'.format(ATE[i]),color=title_color)

    ax.plot(all_treatment_outcome_day_max[:,i],label='treated',linewidth=lw,color='blue')#,marker='o')
    ax.plot(all_control_outcome_day_max[:,i],label='controlled',linewidth=lw,color='orange')
    max_y = max(all_control_outcome_day_max[:,i].max(), all_treatment_outcome_day_max[:,i].max()) + 2
    ax.set_ylim(0, max_y) 
    
    ax.margins(x=0, y=0)
    ax.set_ylabel('# of events')#, color=cols[i])
    ax.set_xlabel('Lead time')
    ax.legend()


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.suptitle('Topic '+str(topic_id), fontsize=16, y=1.0)  
plt.margins(x=0, y=0)
plt.legend()
plt.tight_layout()
fig.savefig("{}/topic_{}.pdf".format(plot_path,topic_id), bbox_inches='tight', dpi=300, transparent=True)
# plt.show()
print('plot saved')

# t  

# save figure
# topic_{}_event_{}.pdf

# 0.2 of the standard deviation

