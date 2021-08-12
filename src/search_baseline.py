# [ols1,ols2]


# [tarnet,cfrwass,cfrmmd]


# [site]


import sys
import os
import numpy as np
from subprocess import call
from itertools import product

try:
    dataset = sys.argv[1]
    gpu = sys.argv[2]
    loop = sys.argv[3]
    model = str(sys.argv[4])
    treat = str(sys.argv[5])
    others = str(sys.argv[6])
except:
    print('Usage: dataset, gpu, loop, model[ols1,ols2,tarnet,cfrmmd,cfrwas,site,deconf,dnd,cevae],treat[24 yemen, 23 afg, syria 7 ],others()')
    exit()

MODELS = ['ols1','ols2','tarnet','cfrmmd','cfrwass','site','deconf','dndc','cevae']
# MODELS = ['ols1','ols2','tarnet','cfrmmd','cfrwass','cevae','deconf','deconf2','dndc','dndc2']
if model not in MODELS:
    print('invalid model',model)

args = ' python train_baseline.py -w 10 --horizon 1 --pred_window 3 -d {} --gpu {} --loop {} --treat_idx {} --aggr_feat'.format(dataset,gpu,loop,treat)
args += others

if model in ['ols1','ols2']:
    params = {
        'batch':[64]
    }
    ii=0
    keys = list(params.keys())
    for combo in product(params['batch'],repeat=1):
        arguments = {k: v for k, v in zip(keys, combo) if v is not None}
        param_set = ''
        for k in arguments:
            param_set += ' --{} {}'.format(k,arguments[k])

        cur_args = args + param_set
        print(' ---- {} -m {} ---- '.format(cur_args, model))
        call('{} -m {}'.format(cur_args, model), shell=True)
        ii+=1
    print('# ', ii)

elif model in ['tarnet','cfrwass','cfrmmd']:
    params = {
        'balance1':[round(10**(i/2),3) for i in range(-10,2)],
        'rep_layer':[1,2,3],
        'hyp_layer':[1,2,3],
        'rep_dim':[50, 100],
        'hyp_dim':[50, 100],
        'batch':[64]
    }
    ii=0
    keys = list(params.keys())
    for combo in product(params['balance1'], params['rep_layer'],params['hyp_layer'],params['rep_dim'],params['hyp_dim'],params['batch'], repeat=1):
        arguments = {k: v for k, v in zip(keys, combo) if v is not None}
        param_set = ''
        for k in arguments:
            param_set += ' --{} {}'.format(k,arguments[k])
        cur_args = args + param_set
        print(' ---- {} -m {} ---- '.format(cur_args, model))
        call('{} -m {}'.format(cur_args, model), shell=True)
        ii+=1
    print('# ', ii)

elif model in ['site']:
    params = {
        'balance1':[round(10**(i/2),3) for i in range(-10,2)],
        'balance2':[round(10**(i/2),3) for i in range(-10,2)],
        'rep_layer':[1,2,3],
        'hyp_layer':[1,2,3],
        'rep_dim':[50, 100],
        'hyp_dim':[50, 100],
        'batch':[64]
    }
    ii=0
    keys = list(params.keys())
    for combo in product(params['balance1'], params['balance2'], params['rep_layer'],params['hyp_layer'],params['rep_dim'],params['hyp_dim'],params['batch'], repeat=1):
        arguments = {k: v for k, v in zip(keys, combo) if v is not None}
        param_set = ''
        for k in arguments:
            param_set += ' --{} {}'.format(k,arguments[k])

        cur_args = args + param_set
        print(' ---- {} -m {} ---- '.format(cur_args, model))
        call('{} -m {}'.format(cur_args, model), shell=True)
        ii+=1
    print('# ', ii)



