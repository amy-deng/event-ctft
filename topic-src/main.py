import sys
import os
import numpy as np
from subprocess import call

'''
python main.py gcn,gat,rgcn EGY_w7h7_mind3n7df0.01 sta_tf_2014-2015_900,sta_tf_2015-2016_900,sta_tf_2016-2017_900 3 64 2 . 4 " --train 0.3"
python main.py temphgt EGY_w7h7_mind3n7df0.01 dyn_tf_2014-2015_900,dyn_tf_2015-2016_900,dyn_tf_2016-2017_900 3 64 1 . 4 " --train 0.3  --n-topics 60 "

'''

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
available_models = ['gcn','gat','heteroall','gcnetall3','gcnetall4','hgtall','hanall','rgcnall','temphgtall',\
    'ours','hgtallcau','ours2','ours3','ours4','ours5','ours5rgcn','ours6','cau6','rgcn','hgt','temphgt',\
        'temp1','temp11','temp2','temp21','temp3','temp4','temp41','temp5','temp6','temp61','temp7','temp71','temp72','temp8','temp81','ditemp81','temp82']

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