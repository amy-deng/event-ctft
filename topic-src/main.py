import sys
import os
import numpy as np
from subprocess import call



try:
    models = sys.argv[1]
    dataset = sys.argv[2]
    datafiles = sys.argv[3]
    horizons = sys.argv[4]
    hidden = sys.argv[5]
    layer = sys.argv[6]
    note = sys.argv[7]
    gpu = sys.argv[8]
    others = sys.argv[9]
except:
    print('Usage: models, dataset, datafiles, horizons, hidden, layer, note, gpu, others')
    exit()
available_models = ['gcn','gat','heteroall','gcnetall3','gcnetall4','hgtall','hanall','rgcnall','temphgtall','ours','hgtallcau','ours2','ours3','ours4','ours5']

model_l = models.split(',')
for m in model_l:
    if m not in available_models:
        print(m,'is unavailable')
        exit()


horizon_l = horizons.split(',')
for mod in model_l:
    for horizon in horizon_l:
        command = "python train.py --dataset {} --datafiles {} --horizon {} --gpu {} -m {}  --n-hidden {} --n-layers {} --note {} {}".format(\
            dataset,datafiles,horizon,gpu,mod,hidden,layer,note, others)  
        print(command)
        call(command, shell=True)