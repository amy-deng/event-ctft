import pandas as pd
import numpy as np
import sys, os, json, time, collections, pickle
import glob
# from sklearn.linear_model import LogisticRegression
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.preprocessing import StandardScaler
# from matplotlib import pyplot as plt
import pickle
import scipy
'''
python causal_sig_test.py ../data THA_topic check_topic_causal_data_w7h7 effect_dict_pw5_biy0
for each event find causes
'''
try:
    out_path = sys.argv[1]
    dataset_name = sys.argv[2] # THA_topic
    raw_data_name = sys.argv[3] 
    effect_dict = sys.argv[4] 
except:
    print("usage: <out_path> <dataset_name `THA_topic`> <raw_data_name `check_topic_causal_data_w7h7`> <effect_dict>")
    exit()

file_path = "{}/{}/{}/causal_effect/{}.pkl".format(out_path,dataset_name,raw_data_name,effect_dict)

with open(file_path,'rb') as f:
    data = pickle.load(f)


# for each type of events find significant causes topic >0.02