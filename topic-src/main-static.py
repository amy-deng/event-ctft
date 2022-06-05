import sys
import os
import numpy as np
from subprocess import call

'''
python main-static.py gcn,gat,rgcn EGY_w7h7_mind3n7df0.01 sta_tf_2014-2015_900,sta_tf_2015-2016_900,sta_tf_2016-2017_900 5 64 3 0  "--train 0.4"
python main-static.py gcn,gat,rgcn AFG_w7h7_mind3n7df0.01 sta_tf_2014-2015_900,sta_tf_2015-2016_900,sta_tf_2016-2017_900 5 64 3 6    "--train 0.4"
python main-static.py gcn,gat,rgcn RUS_w7h7_mind3n10df0.01 sta_tf_2014-2015_900,sta_tf_2015-2016_900,sta_tf_2016-2017_900 5 64 3 8    "--train 0.4"
python main-static.py gcn,gat,rgcn THA_w7h7_mind3n7df0.01 sta_tf_2014-2015_900,sta_tf_2015-2016_900,sta_tf_2016-2017_900 5 64 3 7    "--train 0.4"


/Users/amy/anaconda3/bin/python main-static.py gcn,gat,rgcn EGY_w7h7_mind3n7df0.01,AFG_w7h7_mind3n7df0.01,THA_w7h7_mind3n7df0.01,RUS_w7h7_mind3n10df0.01 sta_tf_2014-2015_900,sta_tf_2015-2016_900,sta_tf_2016-2017_900 5 64 3 7    "--train 0.4"

'''

try:
    models = sys.argv[1]
    datasets = sys.argv[2]
    datafiles = sys.argv[3]
    horizon = sys.argv[4]
    hidden = sys.argv[5]
    layer = sys.argv[6]
    # note = sys.argv[7]
    gpu = sys.argv[7]
    others = sys.argv[8]
except:
    print('Usage: models, datasets, datafiles, horizon, hidden, layer, gpu, others')
    exit()
available_models = ['gcn','gat','rgcn']

model_l = models.split(',')
for m in model_l:
    if m not in available_models:
        print(m,'is unavailable')
        exit()
dataset_l = datasets.split(',')
# horizon_l = horizons.split(',')
for mod in model_l:
    for dataset in dataset_l:
    # for horizon in horizon_l:
        command = "python train.py --dataset {} --datafiles {} --horizon {} --gpu {} -m {}  --n-hidden {} --n-layers {} {}".format(\
        dataset,datafiles,horizon,gpu,mod,hidden,layer,others)  
        print(command)
        call(command, shell=True)