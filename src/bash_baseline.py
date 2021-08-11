import sys
import os
import numpy as np
from subprocess import call

try:
    dataset = sys.argv[1]
    gpu = sys.argv[2]
    loop = sys.argv[3]
    models = str(sys.argv[4])
    treat = str(sys.argv[5])
    others = str(sys.argv[6])
except:
    print('Usage: dataset, gpu, loop, models[ols1,ols2,tarnet,cfrmmd,cfrwas,site,deconf,dnd,cevae,`all` - use `,`],treat[24 yemen, 23 afg, syria 7 ],others()')
    exit()

MODELS = ['ols1','ols2','tarnet','cfrmmd','cfrwass','site','deconf','dndc','cevae']
# MODELS = ['ols1','ols2','tarnet','cfrmmd','cfrwass','cevae','deconf','deconf2','dndc','dndc2']

if 'all' in models:
    model_list = MODELS
else:
    model_list = models.split(',')
    model_list = [m for m in MODELS if m in model_list]
    assert len(model_list) > 0, 'invalid model'

print(model_list)
# treat_list = [int(v) for v in treats.split(',')]
base = "python train_baseline.py"
args = ' -w 10 --horizon 1 --pred_window 3 -d {} --gpu {} --loop {} --treat_idx {}'.format(dataset,gpu,loop,treat)

args += others
for aggr in ["", "--aggr_feat"]:
    for model in model_list:
        comm = '{} {} -m {}  {}'.format(base, args, model, aggr)
        print('----------  {} ---------- '.format(comm))
        call(comm, shell=True)

