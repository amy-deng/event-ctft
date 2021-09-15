import sys
import os
import numpy as np
from subprocess import call

def load_config(cfg_file):
    cfg = {}

    with open(cfg_file,'r') as f:
        for l in f:
            l = l.strip()
            if len(l)>0 and not l[0] == '#':
                vs = l.split('=')
                if len(vs)>0:
                    k,v = (vs[0], eval(vs[1]))
                    if not isinstance(v,list):
                        v = [v]
                    cfg[k] = v
    return cfg

def sample_config(configs):
    cfg_sample = {}
    for k in configs.keys():
        opts = configs[k]
        c = np.random.choice(len(opts),1)[0]
        cfg_sample[k] = opts[c]
    print('-----------cfg_sample----------',cfg_sample)
    return cfg_sample

def cfg_string(cfg):
    ks = sorted(cfg.keys())
    cfg_str = ','.join(['%s:%s' % (k, str(cfg[k])) for k in ks])
    return cfg_str.lower()

def is_used_cfg(cfg, used_cfg_file):
    cfg_str = cfg_string(cfg)
    used_cfgs = read_used_cfgs(used_cfg_file)
    return cfg_str in used_cfgs

def read_used_cfgs(used_cfg_file):
    used_cfgs = set()
    with open(used_cfg_file, 'r') as f:
        for l in f:
            used_cfgs.add(l.strip())
    return used_cfgs

def save_used_cfg(cfg, used_cfg_file):
    with open(used_cfg_file, 'a') as f:
        cfg_str = cfg_string(cfg)
        f.write('%s\n' % cfg_str)

def run(cfg_file, num_runs, model, dataset, datafile, gpu):
    configs = load_config(cfg_file)
    
    """ add configs """
    # define path
    outdir = configs['outdir'][0]
    # configs['model'] = str(model)
    # configs['dataset'] = str(dataset)
    # configs['window'] = window
    # configs['horizon'] = horizon
    # configs['pred_window'] = pred_window
    append_config = " -m {} -d {} --datafile {} --gpu {} ".format(model,dataset,datafile,gpu)
    # print(configs)
    # model = configs['model'][0]
    # dataset = configs['dataset'][0]
    window = configs['window'][0]
    horizon = configs['horizon'][0]
    pred_window = configs['pred_window'][0]
    used_cfg_path = '{}/{}/{}_w{}h{}p{}'.format(outdir,dataset,model,window,horizon,pred_window)
    os.makedirs(used_cfg_path, exist_ok=True)
    used_cfg_file = '{}/used_configs.txt'.format(used_cfg_path,dataset)

    if not os.path.isfile(used_cfg_file):
        f = open(used_cfg_file, 'w')
        f.close()

    for i in range(num_runs):
        cfg = sample_config(configs)
        if is_used_cfg(cfg, used_cfg_file):
            print ('Configuration used, skipping')
            continue

        # save_used_cfg(cfg, used_cfg_file)

        print ('------------------------------')
        print ('Run %d of %d:' % (i+1, num_runs))
        print ('------------------------------')
        print ('\n'.join(['%s: %s' % (str(k), str(v)) for k,v in cfg.items() if len(configs[k])>1]))

        flags = ' '.join('--%s %s' % (k,str(v)) for k,v in cfg.items())
        flags += append_config
        print('python train_freq_p.py {}'.format(flags))
        call('python train_freq_p.py {}'.format(flags), shell=True)

        save_used_cfg(cfg, used_cfg_file)

if __name__ == "__main__":
    # print(len(sys.argv))
    try:
        config_file = sys.argv[1]
        n_runs = int(sys.argv[2])
        model_name = sys.argv[3]
        dataset = sys.argv[4]
        datafile = sys.argv[5]
        gpu = sys.argv[6]
    except:

        print ('Usage: python param_search.py <config file> <num runs> <model name> <dataset> <datafile> <gpu>')
        exit()
    run(config_file, n_runs, model_name, dataset, datafile, gpu)
    