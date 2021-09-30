import pandas as pd
import numpy as np
import sys, os, json, time, collections, pickle
# import glob
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import scipy

try:
    # event_path = sys.argv[1] # /home/sdeng/data/icews/detailed_event_json/THA_2010_w14h7_city.json
    out_path = sys.argv[1]
    dataset = sys.argv[2] # THA_topic
    raw_data_name = sys.argv[3] 
    topic_id = int(sys.argv[4])
    event_code = int(sys.argv[5])
    # start_year = int(sys.argv[3])
    # end_year = int(sys.argv[4])
    # window = int(sys.argv[3])
    # horizon = int(sys.argv[4])
    # lda_name = sys.argv[5]
    # ngram_path = sys.argv[6]
    # top_k_ngram = int(sys.argv[7])
except:
    print("usage: <out_path> <dataset `THA_topic`> <raw_data_name `raw_w10h7`> <topic_id> <event_code 1-20>")
    exit()


with open('{}/{}/{}/topic_{}.pkl'.format(out_path, dataset, raw_data_name, topic_id),'rb') as f:
    dataset = pickle.load(f)

treatment = dataset['treatment']
treatment = treatment[:,topic_id]
treatment = np.where(treatment > 0, 1, 0)

covariate = dataset['covariate']
covariate = np.concatenate([v.toarray() for v in covariate],0)

outcome = dataset['outcome'].sum(1) # number of events; sum of all days
# not binary vector
outcome_sep_day = dataset['outcome'] # number of events; sum of all days


# train propensity scoring function
# logistic regression

cls = LogisticRegression(random_state=42).fit(covariate, treatment)
cls = CalibratedClassifierCV(cls)

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
all_outcome_treatment_day = []
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
        print('outcome_control',outcome_control,'outcome_treatment',outcome_treatment)
        eff = outcome_treatment-outcome_control

        eff_list.append(eff)
        all_outcome_treatment.append(outcome_treatment)
        all_control_outcome.append(outcome_control)
        all_control_outcome_day.append(outcome_sep_day[controlled_indices[min_idx]])
        all_outcome_treatment_day.append(outcome_sep_day[i])
        n_pairs += 1
    else:
        print('no')

eff_list = np.stack(eff_list,0)
all_control_outcome = np.stack(all_control_outcome,0)
all_outcome_treatment = np.stack(all_outcome_treatment,0)
all_control_outcome_day = np.stack(all_control_outcome_day,0)
all_outcome_treatment_day = np.stack(all_outcome_treatment_day,0)

# TEST TODO
all_control_outcome_max = all_control_outcome.max(0)
all_outcome_treatment_max = all_outcome_treatment.max(0)
all_control_outcome_day_max = all_control_outcome_day.max(0)
all_outcome_treatment_day_max = all_outcome_treatment_day.max(0)

ATE = eff_list.mean(0)

# ate 

for i in range(all_control_outcome_day_max.shape[-1]):
    all_outcome_treatment_day_max[:,i]
    pass
# draw treatment group, control group max()

# t  

# save figure
# topic_{}_event_{}.pdf

# 0.2 of the standard deviation

