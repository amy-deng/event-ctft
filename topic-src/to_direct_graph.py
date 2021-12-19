import pandas as pd
import pickle
import sys
import dgl

try:
    dataset = sys.argv[1]
except:
    print('Usage: dataset-name')
    exit()


filenames = ['dyn_tf_2014-2015_600.pkl','dyn_tf_2015-2016_600.pkl','dyn_tf_2016-2017_600.pkl', \
            'dyn_tf_2014-2015_900.pkl','dyn_tf_2015-2016_900.pkl','dyn_tf_2016-2017_900.pkl']

# edges = ['wd','dw','td','tt','ww','tw']
edges = [('word', 'wd', 'doc'), ('topic', 'td', 'doc'), ('topic', 'tt', 'topic'), ('word', 'ww', 'word'), ('topic', 'tw', 'word'), ('doc', 'dw', 'word')]
for file in filenames:
    new_data = []
    print(file)
    with open('../data/' + dataset + '/' + file,'rb') as f:
        data = pickle.load(f)
    print('len',len(data))
    for g in data:
        sub_g = dgl.edge_type_subgraph(g,edges)
        new_data.append(sub_g)
    print(len(data),len(new_data))
    with open('../data/' + dataset + '/' + 'didyn'+file[3:],'wb') as f:
        pickle.dump(new_data,f)
    # break
print('saved')